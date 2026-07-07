import asyncio
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult
from prometheus_client.parser import text_string_to_metric_families

from rag_service.api import create_app
from rag_service.budget import TokenCounter
from rag_service.cache import CacheStats
from rag_service.config import Settings
from rag_service.metrics import MetricsCallbackHandler, ServiceMetrics
from rag_service.service import RagService
from tests.helpers import SAMPLE_DOCS

QUESTION = "What value is the RRF constant k commonly set to?"


def samples(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """Flatten an exposition into {(sample name, sorted labels): value}."""
    flat: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            flat[(sample.name, tuple(sorted(sample.labels.items())))] = sample.value
    return flat


def value(text: str, name: str, **labels: str) -> float:
    return samples(text).get((name, tuple(sorted(labels.items()))), 0.0)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        index_dir=tmp_path / "index",
        ingest_on_startup=SAMPLE_DOCS,
        chunk_size=600,
        chunk_overlap=80,
        reranker="llm",
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_metrics_endpoint_serves_the_exposition_format(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    names = {family.name for family in text_string_to_metric_families(response.text)}
    assert {
        "rag_http_requests",
        "rag_http_request_duration_seconds",
        "rag_retrieval_duration_seconds",
        "rag_llm_duration_seconds",
        "rag_llm_tokens",
        "rag_cache_lookups",
        "rag_cache_hit_ratio",
        "rag_answers",
        "rag_ingest_jobs_queued",
    } <= names


def test_request_retrieval_llm_and_answer_metrics(client: TestClient) -> None:
    client.post("/query", json={"question": QUESTION})
    client.post("/query", json={"question": QUESTION})
    text = client.get("/metrics").text

    route = {"method": "POST", "route": "/query"}
    assert value(text, "rag_http_requests_total", status="200", **route) == 2
    assert value(text, "rag_http_request_duration_seconds_count", **route) == 2
    # The second question is answered from the cache, so only one retrieval ran:
    # one outer (re-ranking) retrieval, its nested hybrid/dense/BM25 runs not counted.
    assert value(text, "rag_retrieval_duration_seconds_count") == 1
    # One answer call plus one grading call per re-ranked candidate.
    llm_calls = value(text, "rag_llm_duration_seconds_count", model="ExtractiveFakeChatModel")
    assert llm_calls == 1 + 12
    assert value(text, "rag_llm_tokens_total", kind="prompt", source="estimate") > 0
    assert value(text, "rag_llm_tokens_total", kind="completion", source="estimate") > 0
    assert value(text, "rag_answers_total", outcome="answered", cached="false") == 1
    assert value(text, "rag_answers_total", outcome="answered", cached="true") == 1
    assert value(text, "rag_cache_lookups_total", cache="answer", result="hit") == 1
    assert value(text, "rag_cache_hit_ratio", cache="answer") == 0.5
    assert value(text, "rag_cache_lookups_total", cache="embedding", result="miss") > 0


def test_unmatched_routes_share_one_label(client: TestClient) -> None:
    client.get("/no/such/path")
    client.get("/another/missing/path")
    text = client.get("/metrics").text
    assert (
        value(text, "rag_http_requests_total", method="GET", route="unmatched", status="404") == 2
    )


def test_provider_reported_usage_is_preferred(settings: Settings) -> None:
    settings.reranker = "none"
    message = AIMessage(
        content="Sixty [1].",
        usage_metadata={"input_tokens": 120, "output_tokens": 7, "total_tokens": 127},
    )
    service = RagService(settings, llm=GenericFakeChatModel(messages=iter([message])))
    service.start()

    asyncio.run(service.answer(QUESTION))

    text = service.metrics.render().decode()
    assert value(text, "rag_llm_tokens_total", kind="prompt", source="provider") == 120
    assert value(text, "rag_llm_tokens_total", kind="completion", source="provider") == 7


def test_refusals_are_counted_by_reason(settings: Settings) -> None:
    settings.reranker = "none"
    settings.min_retrieval_confidence = 0.9
    service = RagService(settings)
    service.start()

    asyncio.run(service.answer("giraffe zebra fusion"))

    text = service.metrics.render().decode()
    assert value(text, "rag_answers_total", outcome="low_confidence", cached="false") == 1


def test_cache_collector_reads_live_counters() -> None:
    metrics = ServiceMetrics()
    stats = CacheStats("embedding", hits=3, misses=1)
    metrics.watch_caches(lambda: [stats])
    stats.record(hits=1, misses=0)

    text = metrics.render().decode()
    assert value(text, "rag_cache_lookups_total", cache="embedding", result="hit") == 4
    assert value(text, "rag_cache_hit_ratio", cache="embedding") == 0.8


def test_watched_gauges_can_be_replaced() -> None:
    metrics = ServiceMetrics()
    metrics.watch_gauge("rag_test_gauge", "test", lambda: 1.0)
    metrics.watch_gauge("rag_test_gauge", "test", lambda: 2.0)
    assert value(metrics.render().decode(), "rag_test_gauge") == 2.0


def test_handler_ignores_events_for_unknown_runs() -> None:
    handler = MetricsCallbackHandler(ServiceMetrics(), TokenCounter())
    handler.on_llm_end(LLMResult(generations=[]), run_id=uuid4())
    handler.on_retriever_end([], run_id=uuid4())
