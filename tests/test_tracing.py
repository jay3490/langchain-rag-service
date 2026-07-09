import asyncio
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from rag_service.config import Settings
from rag_service.service import RagService
from rag_service.tracing import build_tracer_provider
from tests.helpers import SAMPLE_DOCS

QUESTION = "What value is the RRF constant k commonly set to?"


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> Iterator[TracerProvider]:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield tracer_provider
    tracer_provider.shutdown()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        index_dir=tmp_path / "index",
        ingest_on_startup=SAMPLE_DOCS,
        chunk_size=600,
        chunk_overlap=80,
    )


def _service(settings: Settings, provider: TracerProvider, **kwargs: object) -> RagService:
    service = RagService(settings, tracer_provider=provider, **kwargs)  # type: ignore[arg-type]
    service.start()
    return service


def by_name(spans: tuple[ReadableSpan, ...], prefix: str) -> list[ReadableSpan]:
    return [span for span in spans if span.name.startswith(prefix)]


def test_answer_span_contains_retrieval_and_model_spans(
    settings: Settings, provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    service = _service(settings, provider)
    exporter.clear()

    asyncio.run(service.answer(QUESTION, tenant_id="default"))

    spans = exporter.get_finished_spans()
    (answer,) = by_name(spans, "rag.answer")
    retrievals = by_name(spans, "rag.retrieve")
    (llm,) = by_name(spans, "gen_ai.chat")

    assert answer.attributes["rag.tenant"] == "default"
    assert answer.attributes["rag.cached"] is False
    assert answer.attributes["rag.refusal"] == "none"
    assert answer.attributes["rag.sources"] == 4
    # Hybrid search: one outer retriever with dense and BM25 retrievers inside it.
    outer = [s for s in retrievals if s.parent.span_id == answer.context.span_id]
    assert len(outer) == 1
    inner = [s for s in retrievals if s.parent.span_id == outer[0].context.span_id]
    assert {s.attributes["rag.retriever"] for s in inner} == {
        "VectorStoreRetriever",
        "BM25Retriever",
    }
    assert outer[0].attributes["rag.documents"] == 4
    assert llm.parent.span_id == answer.context.span_id
    assert llm.attributes["gen_ai.request.model"] == "ExtractiveFakeChatModel"
    assert {s.context.trace_id for s in spans} == {answer.context.trace_id}


def test_reranker_model_calls_nest_under_the_retrieval(
    settings: Settings, provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    settings.reranker = "llm"
    settings.rerank_candidates = 5
    service = _service(settings, provider)
    exporter.clear()

    asyncio.run(service.answer(QUESTION))

    spans = exporter.get_finished_spans()
    (answer,) = by_name(spans, "rag.answer")
    (outer,) = [
        s for s in by_name(spans, "rag.retrieve") if s.parent.span_id == answer.context.span_id
    ]
    grading = [
        s for s in by_name(spans, "gen_ai.chat") if s.parent.span_id == outer.context.span_id
    ]
    assert outer.attributes["rag.retriever"] == "RerankingRetriever"
    assert len(grading) == 5


def test_cached_answers_and_token_usage_are_recorded(
    settings: Settings, provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    message = AIMessage(
        content="Sixty [1].",
        usage_metadata={"input_tokens": 50, "output_tokens": 3, "total_tokens": 53},
    )
    service = _service(settings, provider, llm=GenericFakeChatModel(messages=iter([message])))
    exporter.clear()

    asyncio.run(service.answer(QUESTION))
    asyncio.run(service.answer(QUESTION))

    spans = exporter.get_finished_spans()
    first, second = by_name(spans, "rag.answer")
    (llm,) = by_name(spans, "gen_ai.chat")
    assert llm.attributes["gen_ai.usage.input_tokens"] == 50
    assert llm.attributes["gen_ai.usage.output_tokens"] == 3
    assert (first.attributes["rag.cached"], second.attributes["rag.cached"]) == (False, True)


def test_streamed_answers_are_traced(
    settings: Settings, provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    service = _service(settings, provider)
    exporter.clear()

    async def consume() -> None:
        async for _ in service.stream_answer(QUESTION):
            pass

    asyncio.run(consume())

    (answer,) = by_name(exporter.get_finished_spans(), "rag.answer")
    assert answer.attributes["rag.streaming"] is True
    assert answer.attributes["rag.sources"] == 4
    assert len(by_name(exporter.get_finished_spans(), "gen_ai.chat")) == 1


def test_failures_mark_the_span_as_an_error(
    settings: Settings, provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    service = _service(settings, provider, llm=GenericFakeChatModel(messages=iter([])))
    exporter.clear()

    with pytest.raises(RuntimeError):
        asyncio.run(service.answer(QUESTION))

    spans = exporter.get_finished_spans()
    (answer,) = by_name(spans, "rag.answer")
    (llm,) = by_name(spans, "gen_ai.chat")
    assert answer.status.status_code is StatusCode.ERROR
    assert llm.status.status_code is StatusCode.ERROR


def test_ingest_is_traced(
    settings: Settings, provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    service = _service(settings, provider)
    exporter.clear()

    service.ingest_files([("notes.md", b"# Notes\n\nDeploys happen on Tuesday.")])

    (span,) = by_name(exporter.get_finished_spans(), "rag.index")
    assert span.attributes["rag.chunks_added"] == 1
    assert span.attributes["rag.tenant"] == "default"


def test_exporter_selection(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    assert build_tracer_provider(settings) is None
    settings.otel_exporter = "console"
    console = build_tracer_provider(settings)
    assert isinstance(console, TracerProvider)
    assert console.resource.attributes["service.name"] == "langchain-rag-service"
    console.shutdown()

    settings.otel_exporter = "otlp"
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None)
    with pytest.raises(RuntimeError, match=r"\[otel\]"):
        build_tracer_provider(settings)
