"""Qdrant backend for :class:`~rag_service.store.DocumentIndex`.

Each index owns one Qdrant collection. Chunks are stored the way ``langchain-qdrant``
expects (``page_content`` and ``metadata`` in the payload), so metadata filters become
payload conditions on ``metadata.<key>``. A small shared meta collection records, per
collection, the embedding model and a content version; a replica compares that version
with its own to notice writes made elsewhere and reload its BM25 copy of the chunks.
"""

from __future__ import annotations

import time
import uuid
import warnings
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from rag_service.filters import TENANT_KEY, MetadataFilter
from rag_service.store import (
    DEFAULT_TENANT,
    DenseFilter,
    DocumentIndex,
    IndexCompatibilityError,
    IndexSnapshot,
    build_snapshot,
    content_version,
)

CONTENT_KEY = QdrantVectorStore.CONTENT_KEY
METADATA_KEY = QdrantVectorStore.METADATA_KEY
META_SUFFIX = "meta"
SCROLL_PAGE = 256


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be UUIDs or integers; chunk ids are 128-bit hex digests."""
    return str(uuid.UUID(hex=chunk_id))


def _condition(key: str, values: tuple[Any, ...]) -> models.Condition:
    field = f"{METADATA_KEY}.{key}"
    if len(values) == 1:
        return models.FieldCondition(key=field, match=models.MatchValue(value=values[0]))
    return models.Filter(
        should=[
            models.FieldCondition(key=field, match=models.MatchValue(value=value))
            for value in values
        ]
    )


def to_qdrant_filter(metadata_filter: MetadataFilter) -> models.Filter | None:
    """Translate a metadata filter into a Qdrant payload filter (``None`` when empty)."""
    if metadata_filter.is_empty:
        return None
    return models.Filter(
        must=[_condition(key, values) for key, values in metadata_filter.conditions]
    )


def qdrant_filter(metadata_filter: MetadataFilter, depth: int) -> dict[str, Any]:
    """Qdrant filters during the search itself, so no extra candidate depth is needed."""
    return {"filter": to_qdrant_filter(metadata_filter)}


def build_qdrant_client(url: str | None, api_key: str | None, path: str) -> QdrantClient:
    """Connect to a server, an in-memory store (``":memory:"``) or a local directory."""
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if url:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(path=path)


class QdrantIndex(DocumentIndex):
    """Chunks of one tenant in the Qdrant collection ``collection``.

    Unlike FAISS, the vectors live outside the process, so an older snapshot shares the
    vector store with newer ones; only its chunk list and BM25 index stay fixed.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embeddings: Embeddings,
        embedding_id: str,
        tenant_id: str = DEFAULT_TENANT,
        *,
        meta_collection: str = f"rag_{META_SUFFIX}",
        sync_seconds: float = 5.0,
    ) -> None:
        self.client = client
        self.collection = collection
        self.meta_collection = meta_collection
        self.sync_seconds = sync_seconds
        self._checked_at = 0.0
        super().__init__(embeddings, embedding_id, tenant_id)

    @property
    def dense_filter(self) -> DenseFilter:
        return qdrant_filter

    # Meta records ----------------------------------------------------------------

    @property
    def _meta_id(self) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-service/{self.collection}"))

    def _ensure_meta_collection(self) -> None:
        if not self.client.collection_exists(self.meta_collection):
            self.client.create_collection(
                self.meta_collection,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
            )

    def _read_meta(self) -> dict[str, Any] | None:
        if not self.client.collection_exists(self.meta_collection):
            return None
        points = self.client.retrieve(self.meta_collection, ids=[self._meta_id])
        return dict(points[0].payload or {}) if points else None

    def _write_meta(self, version: str) -> None:
        self._ensure_meta_collection()
        payload = {
            "collection": self.collection,
            "embedding_id": self.embedding_id,
            "version": version,
        }
        self.client.upsert(
            self.meta_collection,
            points=[models.PointStruct(id=self._meta_id, vector=[1.0], payload=payload)],
        )

    # Loading ---------------------------------------------------------------------

    def _vectorstore(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client, collection_name=self.collection, embedding=self.embeddings
        )

    def _scroll_chunks(self) -> list[Document]:
        chunks: list[Document] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                chunks.append(
                    Document(
                        page_content=str(payload.get(CONTENT_KEY, "")),
                        metadata=dict(payload.get(METADATA_KEY) or {}),
                    )
                )
            if offset is None:
                break
        chunks.sort(key=lambda c: (str(c.metadata["source"]), int(c.metadata["chunk_index"])))
        return chunks

    def load(self) -> None:
        meta = self._read_meta()
        if meta is not None and meta.get("embedding_id") != self.embedding_id:
            raise IndexCompatibilityError(
                f"Qdrant collection {self.collection!r} was built with "
                f"{meta.get('embedding_id')!r}, but the service is configured for "
                f"{self.embedding_id!r}; drop the collection or change the collection prefix"
            )
        if not self.client.collection_exists(self.collection):
            snapshot = IndexSnapshot(dense_filter=qdrant_filter)
        else:
            snapshot = build_snapshot(self._vectorstore(), self._scroll_chunks(), qdrant_filter)
        with self._lock:
            self._snapshot = snapshot
            self._checked_at = time.monotonic()

    def sync(self) -> bool:
        """Reload when another replica changed the collection; returns whether it did.

        The meta record is read at most once every ``sync_seconds``.
        """
        now = time.monotonic()
        if now - self._checked_at < self.sync_seconds:
            return False
        self._checked_at = now
        meta = self._read_meta()
        remote = str(meta.get("version", "empty")) if meta else "empty"
        if remote == self._snapshot.version:
            return False
        self.load()
        return True

    # Mutation --------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        size = len(self.embeddings.embed_query("dimension probe"))
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
        )
        with warnings.catch_warnings():
            # Local mode has no payload indexes and warns that they are ignored.
            warnings.simplefilter("ignore")
            for key in (TENANT_KEY, "source"):
                self.client.create_payload_index(
                    self.collection,
                    field_name=f"{METADATA_KEY}.{key}",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    def _write(self, changed: list[Document], stale: list[str]) -> IndexSnapshot:
        self._ensure_collection()
        if stale:
            self.client.delete(
                self.collection,
                points_selector=models.PointIdsList(points=[point_id(i) for i in stale]),
            )
        vectorstore = self._vectorstore()
        if changed:
            ids = [point_id(str(chunk.metadata["chunk_id"])) for chunk in changed]
            vectorstore.add_documents(changed, ids=ids)
        removed = set(stale)
        chunks = [c for c in self._snapshot.chunks if c.metadata["chunk_id"] not in removed]
        chunks.extend(changed)
        chunks.sort(key=lambda c: (str(c.metadata["source"]), int(c.metadata["chunk_index"])))
        self._write_meta(content_version(chunks))
        return build_snapshot(vectorstore, chunks, qdrant_filter)

    def _drop(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self._write_meta("empty")
