"""Construction of the configured :class:`~rag_service.store.DocumentIndex` backend."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.embeddings import Embeddings

from rag_service.config import Settings
from rag_service.store import DocumentIndex, FaissIndex

IndexFactory = Callable[[str], DocumentIndex]


def build_index_factory(
    settings: Settings, embeddings: Embeddings, embedding_id: str
) -> IndexFactory:
    """Return a function that creates the index of one tenant."""
    if settings.vector_store == "faiss":

        def faiss_index(tenant_id: str) -> DocumentIndex:
            return FaissIndex(settings.index_dir, embeddings, embedding_id, tenant_id)

        return faiss_index

    from rag_service.qdrant_store import META_SUFFIX, QdrantIndex, build_qdrant_client

    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    client = build_qdrant_client(settings.qdrant_url, api_key, str(settings.index_dir / "qdrant"))
    prefix = settings.qdrant_collection_prefix

    def qdrant_index(tenant_id: str) -> DocumentIndex:
        return QdrantIndex(
            client,
            f"{prefix}{tenant_id}",
            embeddings,
            embedding_id,
            tenant_id,
            meta_collection=f"{prefix}{META_SUFFIX}",
            sync_seconds=settings.qdrant_sync_seconds,
        )

    return qdrant_index
