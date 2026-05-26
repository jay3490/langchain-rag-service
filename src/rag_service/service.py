"""Wires settings, models, the index and session memory into one application object."""

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
from rag_service.ingestion import load_bytes, load_directory, safe_source_name, split_documents
from rag_service.memory import SessionStore
from rag_service.providers import build_chat_model, build_embeddings
from rag_service.retrieval import RetrievalOptions, build_retriever
from rag_service.store import DocumentIndex, IngestResult

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
        self.index = DocumentIndex(settings.index_dir, self.embeddings, embedding_id(settings))
        self.sessions = SessionStore(max_messages=settings.max_history_messages)
        self.default_options = RetrievalOptions.from_settings(settings)

    def start(self) -> None:
        """Load the persisted index and ingest the startup directory, if configured."""
        self.index.load()
        logger.info("loaded index with %d chunks", len(self.index.snapshot.chunks))
        if self.settings.ingest_on_startup is not None:
            result = self.ingest_directory(self.settings.ingest_on_startup, restrict=False)
            logger.info(
                "startup ingest of %s: %d chunks added, %d sources unchanged",
                self.settings.ingest_on_startup,
                result.chunks_added,
                len(result.unchanged),
            )

    # Ingestion -------------------------------------------------------------------

    def _index(self, documents: list[Document]) -> IngestResult:
        chunks = split_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        return self.index.add(chunks)

    def resolve_ingest_path(self, path: Path) -> Path:
        """Resolve ``path`` against ``ingest_root`` and refuse anything outside it."""
        root = self.settings.ingest_root.resolve()
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise PathOutsideRootError(f"{path} is outside the ingest root")
        return candidate

    def ingest_directory(self, path: Path, *, restrict: bool = True) -> IngestResult:
        """Ingest every supported file below ``path``.

        With ``restrict`` (the API default) the path is resolved inside ``ingest_root``.
        """
        directory = self.resolve_ingest_path(path) if restrict else path
        return self._index(load_directory(directory))

    def ingest_files(self, files: list[tuple[str, bytes]]) -> IngestResult:
        """Ingest uploaded ``(filename, content)`` pairs."""
        documents: list[Document] = []
        for filename, data in files:
            documents.extend(load_bytes(safe_source_name(filename), data))
        return self._index(documents)

    # Question answering ------------------------------------------------------------

    def options(self, **overrides: object) -> RetrievalOptions:
        return self.default_options.with_overrides(**overrides)

    async def answer(
        self,
        question: str,
        *,
        session_id: str | None = None,
        options: RetrievalOptions | None = None,
    ) -> RagResult:
        """Answer ``question``; with a ``session_id`` the turn is added to its history."""
        retriever = self._retriever(options)
        history = self.sessions.get(session_id) if session_id else []
        result = await build_rag_chain(self.llm, retriever).ainvoke(
            {"question": question, "chat_history": history}
        )
        if session_id:
            self.sessions.append_turn(session_id, question, result["answer"])
        return result

    async def stream_answer(
        self,
        question: str,
        *,
        session_id: str | None = None,
        options: RetrievalOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Answer ``question`` as a sequence of events.

        Emits one ``sources`` event once retrieval is done, a ``token`` event per chunk
        of generated text, and a final ``done`` event with the full answer. The turn is
        stored in the session only after generation completes.
        """
        retriever = self._retriever(options)
        history = self.sessions.get(session_id) if session_id else []
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
        if session_id:
            self.sessions.append_turn(session_id, question, answer)
        yield StreamEvent("done", {"answer": answer, "session_id": session_id})

    def _retriever(self, options: RetrievalOptions | None) -> BaseRetriever:
        snapshot = self.index.snapshot
        if snapshot.is_empty:
            raise EmptyIndexError("the index is empty; ingest documents first")
        return build_retriever(snapshot, options or self.default_options)
