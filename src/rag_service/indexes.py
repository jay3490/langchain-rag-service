"""Per-tenant document indexes for the configured vector-store backend.

Every tenant gets its own index: a FAISS directory ``<index_dir>/<tenant>`` or a Qdrant
collection ``<prefix><tenant>``. Retrieval additionally filters on the ``tenant_id``
stamped on each chunk, so a chunk is only returned to the tenant that ingested it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from langchain_core.embeddings import Embeddings

from rag_service.auth import validate_tenant_id
from rag_service.config import Settings
from rag_service.store import DEFAULT_TENANT, FAISS_FILES, MANIFEST_FILE, DocumentIndex, FaissIndex

logger = logging.getLogger(__name__)

IndexFactory = Callable[[str], DocumentIndex]


def migrate_single_index_layout(index_dir: Path, tenant_id: str = DEFAULT_TENANT) -> bool:
    """Move an index saved directly in ``index_dir`` (before tenancy) into its tenant folder."""
    if not (index_dir / MANIFEST_FILE).exists():
        return False
    target = index_dir / tenant_id
    if (target / MANIFEST_FILE).exists():
        return False
    target.mkdir(parents=True, exist_ok=True)
    for name in (*FAISS_FILES, MANIFEST_FILE):
        if (index_dir / name).exists():
            (index_dir / name).replace(target / name)
    logger.info("moved the index in %s to %s", index_dir, target)
    return True


def build_index_factory(
    settings: Settings, embeddings: Embeddings, embedding_id: str
) -> IndexFactory:
    """Return a function that creates the index of one tenant."""
    if settings.vector_store == "faiss":
        migrate_single_index_layout(settings.index_dir, settings.default_tenant)

        def faiss_index(tenant_id: str) -> DocumentIndex:
            return FaissIndex(settings.index_dir / tenant_id, embeddings, embedding_id, tenant_id)

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


class IndexRegistry:
    """Creates and loads each tenant's index on first use."""

    def __init__(self, factory: IndexFactory) -> None:
        self._factory = factory
        self._indexes: dict[str, DocumentIndex] = {}
        self._lock = threading.Lock()

    def get(self, tenant_id: str) -> DocumentIndex:
        index = self._indexes.get(tenant_id)
        if index is not None:
            return index
        validate_tenant_id(tenant_id)
        with self._lock:
            index = self._indexes.get(tenant_id)
            if index is None:
                index = self._factory(tenant_id)
                index.load()
                logger.info(
                    "loaded index for tenant %s with %d chunks",
                    tenant_id,
                    len(index.snapshot.chunks),
                )
                self._indexes[tenant_id] = index
        return index

    def loaded(self) -> dict[str, DocumentIndex]:
        with self._lock:
            return dict(self._indexes)
