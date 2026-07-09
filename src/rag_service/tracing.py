"""OpenTelemetry tracing for answers, retrieval, model calls and indexing.

The service opens a ``rag.answer`` span per question (``rag.index`` per ingest) and a
:class:`TracingCallbackHandler` turns LangChain retriever and chat-model runs inside it
into child spans: ``rag.retrieve`` for each retriever (nested retrievers nest) and
``gen_ai.chat <model>`` for each model call, with token usage when the provider reports
it.

``RAG_OTEL_EXPORTER`` selects where spans go: ``none`` uses whatever tracer provider
the process already has (a no-op unless something such as ``opentelemetry-instrument``
installed one), ``console`` prints spans, and ``otlp`` sends them over OTLP/HTTP, which
needs the optional ``otel`` extra.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from rag_service import __version__
from rag_service.config import Settings

TRACER_NAME = "rag_service"


def build_tracer_provider(settings: Settings) -> TracerProvider | None:
    """A provider exporting as configured, or ``None`` to use the process-wide one."""
    if settings.otel_exporter == "none":
        return None
    exporter: SpanExporter
    if settings.otel_exporter == "console":
        exporter = ConsoleSpanExporter()
    else:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as exc:
            raise RuntimeError(
                "RAG_OTEL_EXPORTER=otlp needs the OTLP exporter; "
                'install it with pip install "langchain-rag-service[otel]"'
            ) from exc
        exporter = (
            OTLPSpanExporter(endpoint=settings.otel_endpoint)
            if (settings.otel_endpoint)
            else OTLPSpanExporter()
        )
    resource = Resource.create(
        {"service.name": settings.otel_service_name, "service.version": __version__}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def get_tracer(provider: trace.TracerProvider | None) -> Tracer:
    return trace.get_tracer(TRACER_NAME, __version__, tracer_provider=provider)


def _model_name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str:
    params = kwargs.get("invocation_params") or {}
    for key in ("model_name", "model"):
        if params.get(key):
            return str(params[key])
    return str((serialized or {}).get("name") or params.get("_type") or "unknown")


class TracingCallbackHandler(BaseCallbackHandler):
    """Spans for retriever and chat-model runs of one request, under ``parent``.

    LangChain reports each run with its parent run id. Chain runs get no span of
    their own but are remembered, so a model call inside a re-ranking retriever is
    attached to that retriever's span.
    """

    run_inline = True

    def __init__(self, tracer: Tracer, parent: Context | None = None) -> None:
        self.tracer = tracer
        self.parent = parent
        self._spans: dict[UUID, Span] = {}
        self._parents: dict[UUID, UUID | None] = {}
        self._lock = threading.Lock()

    def _context_for(self, parent_run_id: UUID | None) -> Context | None:
        run = parent_run_id
        while run is not None:
            span = self._spans.get(run)
            if span is not None:
                return trace.set_span_in_context(span)
            run = self._parents.get(run)
        return self.parent

    def _start(
        self,
        name: str,
        run_id: UUID,
        parent_run_id: UUID | None,
        attributes: dict[str, Any],
    ) -> None:
        with self._lock:
            self._parents[run_id] = parent_run_id
            span = self.tracer.start_span(
                name, context=self._context_for(parent_run_id), attributes=attributes
            )
            self._spans[run_id] = span

    def _end(self, run_id: UUID, error: BaseException | None = None, **attributes: Any) -> None:
        with self._lock:
            span = self._spans.pop(run_id, None)
        if span is None:
            return
        for key, value in attributes.items():
            span.set_attribute(key, value)
        if error is not None:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, type(error).__name__))
        span.end()

    # Chains: remembered for parentage only ----------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any] | Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self._parents[run_id] = parent_run_id

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            self._parents.pop(run_id, None)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            self._parents.pop(run_id, None)

    # Retrievers ----------------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        retriever = str((serialized or {}).get("name") or kwargs.get("name") or "retriever")
        self._start(
            "rag.retrieve",
            run_id,
            parent_run_id,
            {"rag.retriever": retriever, "rag.query_chars": len(query)},
        )

    def on_retriever_end(self, documents: Sequence[Any], *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, **{"rag.documents": len(documents)})

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error)

    # Chat models ---------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        model = _model_name(serialized, kwargs)
        self._start(
            f"gen_ai.chat {model}",
            run_id,
            parent_run_id,
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": model,
                "gen_ai.request.messages": sum(len(batch) for batch in messages),
            },
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        model = _model_name(serialized, kwargs)
        self._start(
            f"gen_ai.completion {model}",
            run_id,
            parent_run_id,
            {"gen_ai.operation.name": "text_completion", "gen_ai.request.model": model},
        )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        attributes: dict[str, Any] = {}
        for batch in response.generations:
            for generation in batch:
                usage = (
                    getattr(generation.message, "usage_metadata", None)
                    if isinstance(generation, ChatGeneration)
                    else None
                )
                if usage:
                    attributes["gen_ai.usage.input_tokens"] = int(usage["input_tokens"])
                    attributes["gen_ai.usage.output_tokens"] = int(usage["output_tokens"])
        self._end(run_id, **attributes)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error)
