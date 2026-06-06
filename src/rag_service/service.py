"""Wires settings, models, the tenant indexes and session memory into one object."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, NamedTuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from rag_service.chain import (
    RagResult,
    build_answer_chain,
    build_rag_chain,
    build_retrieval_chain,
    source_refs,
)
from rag_service.config import Settings
from rag_service.filters import TENANT_KEY
from rag_service.indexes import IndexRegistry, build_index_factory
from rag_service.ingestion import load_bytes, load_directory, safe_source_name, split_documents
from rag_service.memory import SessionStore
from rag_service.providers import build_chat_model, build_embeddings
from rag_service.retrieval import RetrievalOptions, build_retriever
from rag_service.store import DocumentIndex, IndexSnapshot, IngestResult

logger = logging.getLogger(__name__)


class EmptyIndexError(RuntimeError):
    """Raised when a question is asked before anything has been ingested."""


class PathOutsideRootError(ValueError):
    """Raised when a directory to ingest lies outside ``settings.ingest_root``."""


class StreamEvent(NamedTuple):
    """One event of a streamed answer: ``sources``, ``token`` or ``done``."""

    event: str
    data: Any


def embedding_id(settings: Settings) -> str:
    """Identify the embedding model so that indexes are never mixed across models."""
    if settings.provider == "fake":
        return f"fake:hashing-{settings.fake_embedding_dim}"
    endpoint = settings.openai_base_url or "https://api.openai.com/v1"
    return f"openai:{endpoint}:{settings.embedding_model}"


class RagService:
    def __init__(
        self,
        settings: Settings,
        *,
        embeddings: Embeddings | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings or build_embeddings(settings)
        self.llm = llm or build_chat_model(settings)
        self.indexes = IndexRegistry(
            build_index_factory(settings, self.embeddings, embedding_id(settings))
        )
        self.sessions = SessionStore(max_messages=settings.max_history_messages)
        self.default_options = RetrievalOptions.from_settings(settings)

    @property
    def default_tenant(self) -> str:
        return self.settings.default_tenant

    def index_for(self, tenant_id: str) -> DocumentIndex:
        return self.indexes.get(tenant_id)

    def snapshot_for(self, tenant_id: str) -> IndexSnapshot:
        """The tenant's current snapshot, after picking up writes from other replicas."""
        index = self.index_for(tenant_id)
        index.sync()
        return index.snapshot

    def start(self) -> None:
        """Load the default tenant's index and run the startup ingest, if configured."""
        self.index_for(self.default_tenant)
        if self.settings.ingest_on_startup is not None:
            tenant = self.settings.startup_tenant or self.default_tenant
            result = self.ingest_directory(
                self.settings.ingest_on_startup, tenant_id=tenant, restrict=False
            )
            logger.info(
                "startup ingest of %s for tenant %s: %d chunks added, %d sources unchanged",
                self.settings.ingest_on_startup,
                tenant,
                result.chunks_added,
                len(result.unchanged),
            )

    # Ingestion -------------------------------------------------------------------

    def _index(self, documents: list[Document], tenant_id: str) -> IngestResult:
        chunks = split_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        return self.index_for(tenant_id).add(chunks)

    def resolve_ingest_path(self, path: Path) -> Path:
        """Resolve ``path`` against ``ingest_root`` and refuse anything outside it."""
        root = self.settings.ingest_root.resolve()
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise PathOutsideRootError(f"{path} is outside the ingest root")
        return candidate

    def ingest_directory(
        self, path: Path, *, tenant_id: str | None = None, restrict: bool = True
    ) -> IngestResult:
        """Ingest every supported file below ``path`` into the tenant's index.

        With ``restrict`` (the API default) the path is resolved inside ``ingest_root``.
        """
        directory = self.resolve_ingest_path(path) if restrict else path
        return self._index(load_directory(directory), tenant_id or self.default_tenant)

    def ingest_files(
        self, files: list[tuple[str, bytes]], *, tenant_id: str | None = None
    ) -> IngestResult:
        """Ingest uploaded ``(filename, content)`` pairs into the tenant's index."""
        documents: list[Document] = []
        for filename, data in files:
            documents.extend(load_bytes(safe_source_name(filename), data))
        return self._index(documents, tenant_id or self.default_tenant)

    # Question answering ------------------------------------------------------------

    def options(self, **overrides: object) -> RetrievalOptions:
        return self.default_options.with_overrides(**overrides)

    @staticmethod
    def session_key(tenant_id: str, session_id: str) -> str:
        """Sessions are namespaced by tenant, so equal session ids never collide."""
        return f"{tenant_id}/{session_id}"

    async def answer(
        self,
        question: str,
        *,
        tenant_id: str | None = None,
        session_id: str | None = None,
        options: RetrievalOptions | None = None,
    ) -> RagResult:
        """Answer ``question``; with a ``session_id`` the turn is added to its history."""
        tenant = tenant_id or self.default_tenant
        retriever = self._retriever(tenant, options)
        session = self.session_key(tenant, session_id) if session_id else None
        history = self.sessions.get(session) if session else []
        result = await build_rag_chain(self.llm, retriever).ainvoke(
            {"question": question, "chat_history": history}
        )
        if session:
            self.sessions.append_turn(session, question, result["answer"])
        return result

    async def stream_answer(
        self,
        question: str,
        *,
        tenant_id: str | None = None,
        session_id: str | None = None,
        options: RetrievalOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Answer ``question`` as a sequence of events.

        Emits one ``sources`` event once retrieval is done, a ``token`` event per chunk
        of generated text, and a final ``done`` event with the full answer. The turn is
        stored in the session only after generation completes.
        """
        tenant = tenant_id or self.default_tenant
        retriever = self._retriever(tenant, options)
        session = self.session_key(tenant, session_id) if session_id else None
        history = self.sessions.get(session) if session else []
        state = await build_retrieval_chain(retriever, self.llm).ainvoke(
            {"question": question, "chat_history": history}
        )
        yield StreamEvent(
            "sources",
            {
                "question": question,
                "standalone_question": state["standalone_question"],
                "sources": source_refs(state["documents"]),
            },
        )
        parts: list[str] = []
        async for token in build_answer_chain(self.llm).astream(state):
            parts.append(token)
            yield StreamEvent("token", token)
        answer = "".join(parts)
        if session:
            self.sessions.append_turn(session, question, answer)
        yield StreamEvent("done", {"answer": answer, "session_id": session_id})

    def _retriever(self, tenant_id: str, options: RetrievalOptions | None) -> BaseRetriever:
        snapshot = self.snapshot_for(tenant_id)
        if snapshot.is_empty:
            raise EmptyIndexError("the index is empty; ingest documents first")
        options = options or self.default_options
        # Enforced on every query, whatever filter the client sent.
        enforced = options.metadata_filter.with_condition(TENANT_KEY, tenant_id)
        return build_retriever(snapshot, options.with_overrides(metadata_filter=enforced))
