"""Document indexes: a dense vector store paired with a BM25 index over the same chunks.

:class:`DocumentIndex` holds the backend-independent logic (per-source replacement,
skipping unchanged sources, tenant stamping) and :class:`FaissIndex` persists the vectors
to a directory on disk. Writers build a new :class:`IndexSnapshot` and swap it in under
a lock; readers take the current snapshot and never see a half-applied update.
"""

from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from rank_bm25 import BM25Okapi

from rag_service.filters import TENANT_KEY, MetadataFilter
from rag_service.text import tokenize

MANIFEST_FILE = "manifest.json"
FAISS_FILES = ("index.faiss", "index.pkl")
DEFAULT_TENANT = "default"

#: Turns a :class:`MetadataFilter` into the ``filter`` argument of a vector store search.
DenseFilter = Callable[[MetadataFilter], Any]


class IndexCompatibilityError(RuntimeError):
    """The stored index was built with a different embedding model."""


def faiss_filter(metadata_filter: MetadataFilter) -> Callable[[dict[str, Any]], bool]:
    """FAISS accepts a predicate over chunk metadata."""
    return metadata_filter.matches


@dataclass(frozen=True)
class IndexSnapshot:
    """An immutable view of the indexed chunks.

    ``version`` is derived from the chunk ids, so two snapshots with the same content
    have the same version on every replica; caches use it to invalidate stale answers.
    """

    vectorstore: VectorStore | None = None
    chunks: tuple[Document, ...] = ()
    bm25: BM25Okapi | None = field(default=None, compare=False)
    dense_filter: DenseFilter = field(default=faiss_filter, compare=False)
    version: str = "empty"

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def source_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(str(c.metadata["source"]) for c in self.chunks).items()))


@dataclass(frozen=True)
class IngestResult:
    sources: list[str]
    chunks_added: int
    chunks_removed: int
    unchanged: list[str] = field(default_factory=list)


def content_version(chunks: Sequence[Document]) -> str:
    """Stable fingerprint of a set of chunks, independent of their order."""
    if not chunks:
        return "empty"
    ids = sorted(str(chunk.metadata["chunk_id"]) for chunk in chunks)
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def build_snapshot(
    vectorstore: VectorStore | None,
    chunks: Sequence[Document],
    dense_filter: DenseFilter = faiss_filter,
) -> IndexSnapshot:
    """Pair ``chunks`` with a fresh BM25 index."""
    if vectorstore is None or not chunks:
        return IndexSnapshot(dense_filter=dense_filter)
    bm25 = BM25Okapi([tokenize(chunk.page_content) or [""] for chunk in chunks])
    return IndexSnapshot(
        vectorstore=vectorstore,
        chunks=tuple(chunks),
        bm25=bm25,
        dense_filter=dense_filter,
        version=content_version(chunks),
    )


class DocumentIndex(ABC):
    """Vector store plus BM25 for one tenant.

    ``embedding_id`` identifies the embedding model (for example
    ``"openai:text-embedding-3-small"``). It is stored next to the vectors so that an
    index built with one model is never queried with vectors from another. Every chunk
    added is stamped with ``tenant_id`` in its metadata.
    """

    def __init__(
        self, embeddings: Embeddings, embedding_id: str, tenant_id: str = DEFAULT_TENANT
    ) -> None:
        self.embeddings = embeddings
        self.embedding_id = embedding_id
        self.tenant_id = tenant_id
        self._lock = threading.Lock()
        self._snapshot = IndexSnapshot(dense_filter=self.dense_filter)

    @property
    def snapshot(self) -> IndexSnapshot:
        return self._snapshot

    @property
    @abstractmethod
    def dense_filter(self) -> DenseFilter:
        """Translation of metadata filters for this backend's vector search."""

    @abstractmethod
    def load(self) -> None:
        """Load previously stored chunks, if there are any."""

    @abstractmethod
    def _write(self, changed: list[Document], stale: list[str]) -> IndexSnapshot:
        """Delete ``stale`` chunk ids, add ``changed`` chunks and return the new snapshot.

        Called with the lock held.
        """

    @abstractmethod
    def _drop(self) -> None:
        """Delete everything stored for this index. Called with the lock held."""

    # Mutation ------------------------------------------------------------------

    def _stamp(self, chunk: Document) -> Document:
        return Document(
            page_content=chunk.page_content,
            metadata={**chunk.metadata, TENANT_KEY: self.tenant_id},
        )

    def _ids_for_sources(self, sources: set[str]) -> list[str]:
        return [
            str(chunk.metadata["chunk_id"])
            for chunk in self._snapshot.chunks
            if chunk.metadata["source"] in sources
        ]

    def add(self, chunks: Sequence[Document]) -> IngestResult:
        """Index ``chunks``, replacing any chunks previously indexed for the same sources.

        A source whose chunk ids all match what is already indexed is left alone, so
        re-ingesting an unchanged corpus does not call the embedding model.
        """
        by_source: dict[str, list[Document]] = {}
        for chunk in chunks:
            by_source.setdefault(str(chunk.metadata["source"]), []).append(chunk)
        sources = sorted(by_source)
        with self._lock:
            unchanged = sorted(
                source
                for source, new in by_source.items()
                if set(self._ids_for_sources({source}))
                == {str(chunk.metadata["chunk_id"]) for chunk in new}
            )
            changed = [
                self._stamp(chunk) for chunk in chunks if chunk.metadata["source"] not in unchanged
            ]
            if not changed:
                return IngestResult(sources, chunks_added=0, chunks_removed=0, unchanged=unchanged)
            stale = self._ids_for_sources(set(sources) - set(unchanged))
            self._snapshot = self._write(changed, stale)
        return IngestResult(
            sources, chunks_added=len(changed), chunks_removed=len(stale), unchanged=unchanged
        )

    def remove_source(self, source: str) -> int:
        """Remove every chunk of ``source``; returns how many chunks were removed."""
        with self._lock:
            stale = self._ids_for_sources({source})
            if not stale:
                return 0
            self._snapshot = self._write([], stale)
        return len(stale)

    def clear(self) -> None:
        """Drop the whole index."""
        with self._lock:
            self._drop()
            self._snapshot = IndexSnapshot(dense_filter=self.dense_filter)


class FaissIndex(DocumentIndex):
    """FAISS vectors and their docstore persisted in ``directory``.

    Writes update a copy of the FAISS store, so a request keeps searching the snapshot
    it started with.
    """

    def __init__(
        self,
        directory: Path,
        embeddings: Embeddings,
        embedding_id: str,
        tenant_id: str = DEFAULT_TENANT,
    ) -> None:
        self.directory = directory
        super().__init__(embeddings, embedding_id, tenant_id)

    @property
    def dense_filter(self) -> DenseFilter:
        return faiss_filter

    def _snapshot_of(self, vectorstore: FAISS | None) -> IndexSnapshot:
        if vectorstore is None or vectorstore.index.ntotal == 0:
            return IndexSnapshot(dense_filter=faiss_filter)
        ids = [vectorstore.index_to_docstore_id[i] for i in range(vectorstore.index.ntotal)]
        return build_snapshot(vectorstore, vectorstore.get_by_ids(ids), faiss_filter)

    # Persistence ---------------------------------------------------------------

    def load(self) -> None:
        manifest_path = self.directory / MANIFEST_FILE
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("embedding_id") != self.embedding_id:
            raise IndexCompatibilityError(
                f"index in {self.directory} was built with {manifest.get('embedding_id')!r}, "
                f"but the service is configured for {self.embedding_id!r}; "
                "delete the directory or point RAG_INDEX_DIR elsewhere"
            )
        # The pickle half of a FAISS index can run code when loaded. It is only read
        # from the directory this class writes to, never from user uploads.
        vectorstore = FAISS.load_local(
            str(self.directory),
            self.embeddings,
            allow_dangerous_deserialization=True,
            normalize_L2=True,
        )
        snapshot = self._snapshot_of(vectorstore)
        with self._lock:
            if self._adopt_untagged_chunks(snapshot):
                self._save(snapshot)
            self._snapshot = snapshot

    def _adopt_untagged_chunks(self, snapshot: IndexSnapshot) -> bool:
        """Stamp chunks saved before tenancy existed with this index's tenant.

        The documents are the freshly loaded docstore entries, so updating their
        metadata in place updates what FAISS searches return.
        """
        untagged = [chunk for chunk in snapshot.chunks if TENANT_KEY not in chunk.metadata]
        for chunk in untagged:
            chunk.metadata[TENANT_KEY] = self.tenant_id
        return bool(untagged)

    def _save(self, snapshot: IndexSnapshot) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if not isinstance(snapshot.vectorstore, FAISS):
            self._drop()
            return
        snapshot.vectorstore.save_local(str(self.directory))
        manifest = {"embedding_id": self.embedding_id, "chunks": len(snapshot.chunks)}
        (self.directory / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")

    def _drop(self) -> None:
        for name in (*FAISS_FILES, MANIFEST_FILE):
            (self.directory / name).unlink(missing_ok=True)

    # Mutation ------------------------------------------------------------------

    def _clone_vectorstore(self) -> FAISS | None:
        current = self._snapshot.vectorstore
        if not isinstance(current, FAISS):
            return None
        return FAISS.deserialize_from_bytes(
            current.serialize_to_bytes(),
            self.embeddings,
            allow_dangerous_deserialization=True,
            normalize_L2=True,
        )

    def _write(self, changed: list[Document], stale: list[str]) -> IndexSnapshot:
        ids = [str(chunk.metadata["chunk_id"]) for chunk in changed]
        vectorstore = self._clone_vectorstore()
        if vectorstore is None:
            if not changed:
                return self._snapshot
            vectorstore = FAISS.from_documents(changed, self.embeddings, ids=ids, normalize_L2=True)
        else:
            if stale:
                vectorstore.delete(stale)
            if changed:
                vectorstore.add_documents(changed, ids=ids)
        snapshot = self._snapshot_of(vectorstore)
        self._save(snapshot)
        return snapshot
