"""FAISS-backed document index with a matching BM25 index, persisted to disk.

Writers build a new :class:`IndexSnapshot` and swap it in under a lock; readers take the
current snapshot and never see a half-applied update.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from rag_service.text import tokenize

MANIFEST_FILE = "manifest.json"
FAISS_FILES = ("index.faiss", "index.pkl")


class IndexCompatibilityError(RuntimeError):
    """The index on disk was built with a different embedding model."""


@dataclass(frozen=True)
class IndexSnapshot:
    """An immutable view of the indexed chunks."""

    vectorstore: FAISS | None = None
    chunks: tuple[Document, ...] = ()
    bm25: BM25Okapi | None = field(default=None, compare=False)

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


def _build_snapshot(vectorstore: FAISS | None) -> IndexSnapshot:
    if vectorstore is None:
        return IndexSnapshot()
    ids = [vectorstore.index_to_docstore_id[i] for i in range(vectorstore.index.ntotal)]
    chunks = tuple(vectorstore.get_by_ids(ids))
    if not chunks:
        return IndexSnapshot()
    bm25 = BM25Okapi([tokenize(chunk.page_content) or [""] for chunk in chunks])
    return IndexSnapshot(vectorstore=vectorstore, chunks=chunks, bm25=bm25)


class DocumentIndex:
    """Owns the FAISS store and BM25 index for one directory on disk.

    ``embedding_id`` identifies the embedding model (for example
    ``"openai:text-embedding-3-small"``). It is written next to the index so that an
    index built with one model is never queried with vectors from another.
    """

    def __init__(self, directory: Path, embeddings: Embeddings, embedding_id: str) -> None:
        self.directory = directory
        self.embeddings = embeddings
        self.embedding_id = embedding_id
        self._lock = threading.Lock()
        self._snapshot = IndexSnapshot()

    @property
    def snapshot(self) -> IndexSnapshot:
        return self._snapshot

    # Persistence ---------------------------------------------------------------

    def load(self) -> None:
        """Load a previously saved index, if there is one."""
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
        with self._lock:
            self._snapshot = _build_snapshot(vectorstore)

    def _save(self, snapshot: IndexSnapshot) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if snapshot.vectorstore is None:
            for name in (*FAISS_FILES, MANIFEST_FILE):
                (self.directory / name).unlink(missing_ok=True)
            return
        snapshot.vectorstore.save_local(str(self.directory))
        manifest = {"embedding_id": self.embedding_id, "chunks": len(snapshot.chunks)}
        (self.directory / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")

    # Mutation ------------------------------------------------------------------

    def _clone_vectorstore(self) -> FAISS | None:
        current = self._snapshot.vectorstore
        if current is None:
            return None
        return FAISS.deserialize_from_bytes(
            current.serialize_to_bytes(),
            self.embeddings,
            allow_dangerous_deserialization=True,
            normalize_L2=True,
        )

    def _ids_for_sources(self, sources: set[str]) -> list[str]:
        return [
            str(chunk.metadata["chunk_id"])
            for chunk in self._snapshot.chunks
            if chunk.metadata["source"] in sources
        ]

    def add(self, chunks: Sequence[Document]) -> IngestResult:
        """Index ``chunks``, replacing any chunks previously indexed for the same sources."""
        sources = sorted({str(chunk.metadata["source"]) for chunk in chunks})
        if not chunks:
            return IngestResult(sources=[], chunks_added=0, chunks_removed=0)
        ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
        with self._lock:
            stale = self._ids_for_sources(set(sources))
            vectorstore = self._clone_vectorstore()
            if vectorstore is None:
                vectorstore = FAISS.from_documents(
                    list(chunks), self.embeddings, ids=ids, normalize_L2=True
                )
            else:
                if stale:
                    vectorstore.delete(stale)
                vectorstore.add_documents(list(chunks), ids=ids)
            snapshot = _build_snapshot(vectorstore)
            self._save(snapshot)
            self._snapshot = snapshot
        return IngestResult(sources=sources, chunks_added=len(chunks), chunks_removed=len(stale))

    def remove_source(self, source: str) -> int:
        """Remove every chunk of ``source``; returns how many chunks were removed."""
        with self._lock:
            stale = self._ids_for_sources({source})
            if not stale:
                return 0
            vectorstore = self._clone_vectorstore()
            if vectorstore is None:
                return 0
            vectorstore.delete(stale)
            snapshot = _build_snapshot(vectorstore if vectorstore.index.ntotal else None)
            self._save(snapshot)
            self._snapshot = snapshot
        return len(stale)

    def clear(self) -> None:
        """Drop the whole index, in memory and on disk."""
        with self._lock:
            self._snapshot = IndexSnapshot()
            self._save(self._snapshot)
