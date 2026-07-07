"""Prometheus metrics, served at ``GET /metrics``.

Each application owns a :class:`ServiceMetrics` with its own registry. Request metrics
come from the request middleware, retrieval and model metrics from
:class:`MetricsCallbackHandler` (a LangChain callback handler attached to every chain
run), and cache metrics are read from the caches' counters at scrape time.

Token counts use the usage the provider reports when there is one
(``source="provider"``); otherwise prompt and completion text are counted with the
service's tokenizer (``source="estimate"``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.outputs import ChatGeneration, LLMResult
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from rag_service.budget import TokenCounter
from rag_service.cache import CacheStats

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class CacheCollector(Collector):
    """Exposes lookups and the hit ratio of each cache from its live counters."""

    def __init__(self, stats: Callable[[], Iterable[CacheStats]]) -> None:
        self._stats = stats

    def collect(self) -> Iterable[Metric]:
        lookups = CounterMetricFamily(
            "rag_cache_lookups", "Cache lookups by cache and result.", labels=["cache", "result"]
        )
        ratio = GaugeMetricFamily(
            "rag_cache_hit_ratio", "Hits divided by lookups since start.", labels=["cache"]
        )
        for stats in self._stats():
            lookups.add_metric([stats.name, "hit"], stats.hits)
            lookups.add_metric([stats.name, "miss"], stats.misses)
            ratio.add_metric([stats.name], stats.hit_ratio)
        yield lookups
        yield ratio


class GaugeCollector(Collector):
    """A gauge whose value is computed when Prometheus scrapes."""

    def __init__(self, name: str, documentation: str, value: Callable[[], float]) -> None:
        self.name = name
        self.documentation = documentation
        self._value = value

    def collect(self) -> Iterable[Metric]:
        gauge = GaugeMetricFamily(self.name, self.documentation)
        gauge.add_metric([], self._value())
        yield gauge


class ServiceMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self._gauges: dict[str, GaugeCollector] = {}
        self.http_requests = Counter(
            "rag_http_requests",
            "HTTP requests by method, route template and status.",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.http_latency = Histogram(
            "rag_http_request_duration_seconds",
            "HTTP request latency, until the response body has been sent.",
            ["method", "route"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.retrieval_latency = Histogram(
            "rag_retrieval_duration_seconds",
            "Latency of one retrieval, including fusion and re-ranking.",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.llm_latency = Histogram(
            "rag_llm_duration_seconds",
            "Latency of one chat model call.",
            ["model"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.llm_tokens = Counter(
            "rag_llm_tokens",
            "Chat model tokens by kind (prompt or completion) and how they were counted.",
            ["kind", "source"],
            registry=self.registry,
        )
        self.answers = Counter(
            "rag_answers",
            "Answers by outcome (answered or the refusal reason) and cache use.",
            ["outcome", "cached"],
            registry=self.registry,
        )

    def observe_request(self, method: str, route: str, status: int, seconds: float) -> None:
        self.http_requests.labels(method, route, str(status)).inc()
        self.http_latency.labels(method, route).observe(seconds)

    def observe_answer(self, refusal: str | None, cached: bool) -> None:
        self.answers.labels(refusal or "answered", str(cached).lower()).inc()

    def watch_caches(self, stats: Callable[[], Iterable[CacheStats]]) -> None:
        self.registry.register(CacheCollector(stats))

    def watch_gauge(self, name: str, documentation: str, value: Callable[[], float]) -> None:
        """Register (or replace, e.g. after an application restart) a computed gauge."""
        previous = self._gauges.pop(name, None)
        if previous is not None:
            self.registry.unregister(previous)
        self._gauges[name] = GaugeCollector(name, documentation, value)
        self.registry.register(self._gauges[name])

    def render(self) -> bytes:
        return generate_latest(self.registry)


def _model_name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str:
    params = kwargs.get("invocation_params") or {}
    for key in ("model_name", "model"):
        if params.get(key):
            return str(params[key])
    if serialized and serialized.get("name"):
        return str(serialized["name"])
    return str(params.get("_type", "unknown"))


class MetricsCallbackHandler(BaseCallbackHandler):
    """Times retrievals and chat model calls and counts tokens.

    Retrievers nest (hybrid search runs a dense and a BM25 retriever, re-ranking wraps
    the hybrid one); only the outermost retrieval of a run is timed.
    """

    run_inline = True

    def __init__(self, metrics: ServiceMetrics, tokens: TokenCounter) -> None:
        self.metrics = metrics
        self.tokens = tokens
        self._retrievals: dict[UUID, float | None] = {}
        self._llm_calls: dict[UUID, tuple[float, str, str]] = {}

    # Retrieval ---------------------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        nested = parent_run_id is not None and parent_run_id in self._retrievals
        self._retrievals[run_id] = None if nested else time.perf_counter()

    def _finish_retrieval(self, run_id: UUID) -> None:
        started = self._retrievals.pop(run_id, None)
        if started is not None:
            self.metrics.retrieval_latency.observe(time.perf_counter() - started)

    def on_retriever_end(self, documents: Sequence[Any], *, run_id: UUID, **kwargs: Any) -> None:
        self._finish_retrieval(run_id)

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish_retrieval(run_id)

    # Chat models -------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        prompt = "\n".join(get_buffer_string(batch) for batch in messages)
        self._llm_calls[run_id] = (time.perf_counter(), _model_name(serialized, kwargs), prompt)

    def on_llm_start(
        self, serialized: dict[str, Any] | None, prompts: list[str], *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._llm_calls[run_id] = (
            time.perf_counter(),
            _model_name(serialized, kwargs),
            "\n".join(prompts),
        )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        call = self._llm_calls.pop(run_id, None)
        if call is None:
            return
        started, model, prompt = call
        self.metrics.llm_latency.labels(model).observe(time.perf_counter() - started)
        prompt_tokens, completion_tokens, source = self._usage(response, prompt)
        self.metrics.llm_tokens.labels("prompt", source).inc(prompt_tokens)
        self.metrics.llm_tokens.labels("completion", source).inc(completion_tokens)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        call = self._llm_calls.pop(run_id, None)
        if call is not None:
            self.metrics.llm_latency.labels(call[1]).observe(time.perf_counter() - call[0])

    def _usage(self, response: LLMResult, prompt: str) -> tuple[int, int, str]:
        generations = [g for batch in response.generations for g in batch]
        for generation in generations:
            if isinstance(generation, ChatGeneration):
                usage = getattr(generation.message, "usage_metadata", None)
                if usage:
                    return int(usage["input_tokens"]), int(usage["output_tokens"]), "provider"
        completion = "".join(g.text for g in generations)
        return self.tokens.count(prompt), self.tokens.count(completion), "estimate"
