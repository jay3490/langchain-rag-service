"""Wires settings, models, the tenant indexes and session memory into one object."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, NamedTuple, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, Status, StatusCode

from rag_service.budget import ContextBudget, get_token_counter
from rag_service.cache import AnswerCache, Cache, CachedEmbeddings, CacheStats, build_cache
from rag_service.chain import (
    RagResult,
    build_answer_chain,
    build_rag_chain,
    build_retrieval_chain,
    source_refs,
)
from rag_service.config import Settings
from rag_service.filters import TENANT_KEY
from rag_service.guardrails import Guardrails
from rag_service.indexes import IndexRegistry, build_index_factory
from rag_service.ingestion import load_bytes, load_directory, safe_source_name, split_documents
from rag_service.memory import SessionStore
from rag_service.metrics import MetricsCallbackHandler, ServiceMetrics
from rag_service.providers import build_chat_model, build_embeddings
from rag_service.rerank import Reranker, RerankingRetriever, build_reranker
from rag_service.retrieval import RetrievalOptions, build_retriever
from rag_service.store import DocumentIndex, IndexSnapshot, IngestResult
from rag_service.tracing import TracingCallbackHandler, build_tracer_provider, get_tracer

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
        reranker: Reranker | None = None,
        cache: Cache | None = None,
        tracer_provider: TracerProvider | None = None,
    ) -> None:
        self.settings = settings
        self._owned_tracer_provider = (
            build_tracer_provider(settings) if tracer_provider is None else None
        )
        self.tracer = get_tracer(tracer_provider or self._owned_tracer_provider)
        self.cache = cache or (build_cache(settings) if settings.cache_enabled else None)
        base_embeddings = embeddings or build_embeddings(settings)
        self.embeddings: Embeddings = (
            CachedEmbeddings(
                base_embeddings,
                self.cache,
                embedding_id(settings),
                ttl=settings.embedding_cache_ttl_seconds,
                prefix=settings.cache_prefix,
            )
            if self.cache is not None
            else base_embeddings
        )
        self.answers = (
            AnswerCache(
                self.cache, ttl=settings.answer_cache_ttl_seconds, prefix=settings.cache_prefix
            )
            if self.cache is not None and settings.answer_cache_ttl_seconds
            else None
        )
        self.llm = llm or build_chat_model(settings)
        self.indexes = IndexRegistry(
            build_index_factory(settings, self.embeddings, embedding_id(settings))
        )
        self.sessions = SessionStore(max_messages=settings.max_history_messages)
        self.default_options = RetrievalOptions.from_settings(settings)
        self.reranker = reranker or build_reranker(settings, self.llm)
        self.guardrails = Guardrails(
            min_confidence=settings.min_retrieval_confidence,
            citation_policy=settings.citation_policy,
        )
        self.tokens = get_token_counter(settings.tokenizer_encoding)
        self.metrics = ServiceMetrics()
        self.callbacks: list[BaseCallbackHandler] = [
            MetricsCallbackHandler(self.metrics, self.tokens)
        ]
        self.metrics.watch_caches(self._cache_stats)
        self.budget = (
            ContextBudget(self.tokens, settings.max_context_tokens)
            if settings.max_context_tokens
            else None
        )

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

    def _cache_stats(self) -> list[CacheStats]:
        stats: list[CacheStats] = []
        if isinstance(self.embeddings, CachedEmbeddings):
            stats.append(self.embeddings.stats)
        if self.answers is not None:
            stats.append(self.answers.stats)
        return stats

    def run_config(self, span: Span) -> RunnableConfig:
        """Config for a chain run: metrics, and tracing under ``span``."""
        tracing = TracingCallbackHandler(self.tracer, trace.set_span_in_context(span))
        return {"callbacks": [*self.callbacks, tracing]}

    def close(self) -> None:
        """Release external connections (the cache) and flush spans."""
        if self.cache is not None:
            self.cache.close()
        if self._owned_tracer_provider is not None:
            self._owned_tracer_provider.shutdown()

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
        with self.tracer.start_as_current_span(
            "rag.index", attributes={"rag.tenant": tenant_id, "rag.documents": len(documents)}
        ) as span:
            chunks = split_documents(
                documents, self.settings.chunk_size, self.settings.chunk_overlap
            )
            result = self.index_for(tenant_id).add(chunks)
            span.set_attribute("rag.chunks_added", result.chunks_added)
            span.set_attribute("rag.chunks_removed", result.chunks_removed)
            span.set_attribute("rag.sources_unchanged", len(result.unchanged))
            return result

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
        """Answer ``question``; with a ``session_id`` the turn is added to its history.

        A question without prior history is looked up in the answer cache first.
        """
        tenant = tenant_id or self.default_tenant
        span = self._answer_span(tenant, question, streaming=False)
        try:
            result = await self._answer(question, tenant, session_id, options, span)
        except Exception as exc:
            _record_error(span, exc)
            raise
        else:
            span.set_attribute("rag.confidence", result["confidence"])
            _annotate(span, result["cached"], result["refusal"], len(result["sources"]))
            return result
        finally:
            span.end()

    async def _answer(
        self,
        question: str,
        tenant: str,
        session_id: str | None,
        options: RetrievalOptions | None,
        span: Span,
    ) -> RagResult:
        snapshot, effective = self._prepare(tenant, options)
        session = self.session_key(tenant, session_id) if session_id else None
        history = self.sessions.get(session) if session else []
        cache_key = None if history else self._answer_key(tenant, question, snapshot, effective)
        cached = self._cached_answer(cache_key)
        if cached is not None:
            result = cached
        else:
            retriever = self._build_retriever(snapshot, effective)
            chain = build_rag_chain(self.llm, retriever, self.budget, self.guardrails)
            result = await chain.ainvoke(
                {"question": question, "chat_history": history}, config=self.run_config(span)
            )
            self._store_answer(cache_key, result)
        if session:
            self.sessions.append_turn(session, question, result["answer"])
        self.metrics.observe_answer(result["refusal"], result["cached"])
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
        stored in the session only after generation completes. An answer from the cache
        is sent as a single ``token`` event.
        """
        tenant = tenant_id or self.default_tenant
        span = self._answer_span(tenant, question, streaming=True)
        sources = 0
        try:
            async for event in self._stream_answer(question, tenant, session_id, options, span):
                if event.event == "sources":
                    sources = len(event.data["sources"])
                    span.set_attribute("rag.confidence", event.data["confidence"])
                elif event.event == "done":
                    _annotate(span, event.data["cached"], event.data["refusal"], sources)
                yield event
        except Exception as exc:
            _record_error(span, exc)
            raise
        finally:
            span.end()

    async def _stream_answer(
        self,
        question: str,
        tenant: str,
        session_id: str | None,
        options: RetrievalOptions | None,
        span: Span,
    ) -> AsyncIterator[StreamEvent]:
        snapshot, effective = self._prepare(tenant, options)
        session = self.session_key(tenant, session_id) if session_id else None
        history = self.sessions.get(session) if session else []
        cache_key = None if history else self._answer_key(tenant, question, snapshot, effective)
        cached = self._cached_answer(cache_key)
        if cached is not None:
            for event in self._replay(cached, session_id):
                yield event
            if session:
                self.sessions.append_turn(session, question, cached["answer"])
            self.metrics.observe_answer(cached["refusal"], cached=True)
            return
        retriever = self._build_retriever(snapshot, effective)
        state = await build_retrieval_chain(retriever, self.llm, self.budget).ainvoke(
            {"question": question, "chat_history": history}, config=self.run_config(span)
        )
        yield StreamEvent(
            "sources",
            {
                "question": question,
                "standalone_question": state["standalone_question"],
                "sources": source_refs(state["documents"]),
                "confidence": round(float(state["confidence"]), 6),
            },
        )
        parts: list[str] = []
        answer_chain = build_answer_chain(self.llm, self.guardrails)
        async for token in answer_chain.astream(state, config=self.run_config(span)):
            parts.append(token)
            yield StreamEvent("token", token)
        reviewed = self.guardrails.review("".join(parts), state["documents"], state["confidence"])
        self.metrics.observe_answer(reviewed.refusal, cached=False)
        if session:
            self.sessions.append_turn(session, question, reviewed.answer)
        self._store_answer(
            cache_key,
            RagResult(
                question=question,
                standalone_question=state["standalone_question"],
                answer=reviewed.answer,
                sources=source_refs(state["documents"]),
                confidence=round(float(state["confidence"]), 6),
                citations=reviewed.citations,
                invalid_citations=reviewed.invalid_citations,
                refusal=reviewed.refusal,
                cached=False,
            ),
        )
        yield StreamEvent(
            "done",
            {
                "answer": reviewed.answer,
                "session_id": session_id,
                "citations": reviewed.citations,
                "invalid_citations": reviewed.invalid_citations,
                "refusal": reviewed.refusal,
                "cached": False,
            },
        )

    def _answer_span(self, tenant: str, question: str, *, streaming: bool) -> Span:
        return self.tracer.start_span(
            "rag.answer",
            attributes={
                "rag.tenant": tenant,
                "rag.question_chars": len(question),
                "rag.streaming": streaming,
            },
        )

    @staticmethod
    def _replay(result: RagResult, session_id: str | None) -> list[StreamEvent]:
        return [
            StreamEvent(
                "sources",
                {
                    "question": result["question"],
                    "standalone_question": result["standalone_question"],
                    "sources": result["sources"],
                    "confidence": result["confidence"],
                },
            ),
            StreamEvent("token", result["answer"]),
            StreamEvent(
                "done",
                {
                    "answer": result["answer"],
                    "session_id": session_id,
                    "citations": result["citations"],
                    "invalid_citations": result["invalid_citations"],
                    "refusal": result["refusal"],
                    "cached": True,
                },
            ),
        ]

    # Answer cache ------------------------------------------------------------------

    def _answer_settings(self, options: RetrievalOptions) -> dict[str, Any]:
        """Everything besides the question and index that can change an answer."""
        s = self.settings
        return {
            "model": embedding_id(s),
            "chat_model": s.chat_model if s.provider != "fake" else "extractive-fake",
            "temperature": s.temperature,
            "mode": options.mode,
            "search_type": options.search_type,
            "k": options.k,
            "fetch_k": options.fetch_k,
            "mmr_lambda": options.mmr_lambda,
            "rrf_k": options.rrf_k,
            "weights": [options.dense_weight, options.bm25_weight],
            "reranker": s.reranker if options.rerank else "none",
            "rerank_candidates": s.rerank_candidates,
            "max_context_tokens": s.max_context_tokens,
            "min_confidence": s.min_retrieval_confidence,
            "citation_policy": s.citation_policy,
        }

    def _answer_key(
        self, tenant: str, question: str, snapshot: IndexSnapshot, options: RetrievalOptions
    ) -> str | None:
        if self.answers is None:
            return None
        return self.answers.key(
            tenant_id=tenant,
            question=question,
            metadata_filter=options.metadata_filter.as_dict(),
            settings=self._answer_settings(options),
            index_version=snapshot.version,
        )

    def _cached_answer(self, key: str | None) -> RagResult | None:
        if key is None or self.answers is None:
            return None
        hit = self.answers.get(key)
        if hit is None:
            return None
        return cast(RagResult, {**hit, "cached": True})

    def _store_answer(self, key: str | None, result: RagResult) -> None:
        if key is not None and self.answers is not None:
            self.answers.put(key, result)

    # Retrieval ---------------------------------------------------------------------

    def _prepare(
        self, tenant_id: str, options: RetrievalOptions | None
    ) -> tuple[IndexSnapshot, RetrievalOptions]:
        """The tenant's snapshot and the options with the tenant filter enforced."""
        snapshot = self.snapshot_for(tenant_id)
        if snapshot.is_empty:
            raise EmptyIndexError("the index is empty; ingest documents first")
        options = options or self.default_options
        # Enforced on every query, whatever filter the client sent.
        return snapshot, options.with_overrides(
            metadata_filter=options.metadata_filter.with_condition(TENANT_KEY, tenant_id)
        )

    def _retriever(self, tenant_id: str, options: RetrievalOptions | None) -> BaseRetriever:
        return self._build_retriever(*self._prepare(tenant_id, options))

    def _build_retriever(self, snapshot: IndexSnapshot, options: RetrievalOptions) -> BaseRetriever:
        if self.reranker is None or not options.rerank:
            return build_retriever(snapshot, options)
        candidates = max(options.k, self.settings.rerank_candidates)
        return RerankingRetriever(
            base=build_retriever(snapshot, options.with_overrides(k=candidates)),
            reranker=self.reranker,
            k=options.k,
        )


def _annotate(span: Span, cached: bool, refusal: str | None, sources: int) -> None:
    span.set_attribute("rag.cached", cached)
    span.set_attribute("rag.refusal", refusal or "none")
    span.set_attribute("rag.sources", sources)


def _record_error(span: Span, error: BaseException) -> None:
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, type(error).__name__))
