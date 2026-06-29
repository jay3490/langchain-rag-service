import asyncio
import json
from pathlib import Path
from typing import Any

import fakeredis
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from rag_service.api import create_app
from rag_service.cache import AnswerCache, MemoryCache, RedisCache, normalize_question
from rag_service.config import Settings
from rag_service.service import RagService
from tests.helpers import SAMPLE_DOCS

QUESTION = "What value is the RRF constant k commonly set to?"


def test_normalize_question() -> None:
    assert normalize_question("  What IS   bm25?? ") == "what is bm25"
    assert normalize_question("what is bm25") == "what is bm25"


def _key(cache: AnswerCache, **overrides: Any) -> str:
    fields: dict[str, Any] = {
        "tenant_id": "acme",
        "question": "What is BM25?",
        "metadata_filter": {"tenant_id": ["acme"]},
        "settings": {"k": 4},
        "index_version": "v1",
    }
    return cache.key(**{**fields, **overrides})


def test_key_covers_tenant_question_filters_settings_and_index() -> None:
    cache = AnswerCache(MemoryCache(), ttl=60)
    base = _key(cache)

    assert _key(cache, question="what is bm25") == base
    assert _key(cache, tenant_id="globex") != base
    assert _key(cache, question="What is MMR?") != base
    assert _key(cache, metadata_filter={"tenant_id": ["acme"], "source": ["a.md"]}) != base
    assert _key(cache, settings={"k": 5}) != base
    assert _key(cache, index_version="v2") != base


def test_answer_cache_round_trip_and_stats() -> None:
    cache = AnswerCache(RedisCache(fakeredis.FakeRedis()), ttl=60)
    key = _key(cache)
    assert cache.get(key) is None
    cache.put(key, {"answer": "Lexical ranking [1].", "sources": []})
    assert cache.get(key) == {"answer": "Lexical ranking [1].", "sources": []}
    assert (cache.stats.hits, cache.stats.misses) == (1, 1)


def _counting_llm(replies: int) -> GenericFakeChatModel:
    return GenericFakeChatModel(
        messages=iter([AIMessage(content=f"Answer {i} [1].") for i in range(replies)])
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        index_dir=tmp_path / "index",
        ingest_root=SAMPLE_DOCS.parent,
        ingest_on_startup=SAMPLE_DOCS,
        chunk_size=600,
        chunk_overlap=80,
    )


def _service(settings: Settings, replies: int = 5) -> RagService:
    service = RagService(settings, llm=_counting_llm(replies))
    service.start()
    return service


def test_repeated_questions_are_answered_from_the_cache(settings: Settings) -> None:
    service = _service(settings)

    first = asyncio.run(service.answer(QUESTION))
    second = asyncio.run(service.answer("  what value is the rrf constant k commonly set to "))

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["answer"] == first["answer"] == "Answer 0 [1]."
    assert second["sources"] == first["sources"]


def test_cache_respects_tenants_filters_and_options(settings: Settings) -> None:
    service = _service(settings)
    service.ingest_directory(Path("docs"), tenant_id="acme")

    answers = [
        asyncio.run(service.answer(QUESTION))["answer"],
        asyncio.run(service.answer(QUESTION, tenant_id="acme"))["answer"],
        asyncio.run(service.answer(QUESTION, options=service.options(k=2)))["answer"],
    ]

    assert answers == ["Answer 0 [1].", "Answer 1 [1].", "Answer 2 [1]."]


def test_ingesting_changes_the_key(settings: Settings) -> None:
    service = _service(settings)
    asyncio.run(service.answer(QUESTION))
    service.ingest_files([("extra.md", b"# Extra\n\nRRF k is sometimes tuned per corpus.")])

    again = asyncio.run(service.answer(QUESTION))

    assert again["cached"] is False
    assert again["answer"] == "Answer 1 [1]."


def test_follow_up_questions_bypass_the_cache(settings: Settings) -> None:
    service = _service(settings)
    asyncio.run(service.answer(QUESTION))

    first_turn = asyncio.run(service.answer(QUESTION, session_id="s"))
    follow_up = asyncio.run(service.answer(QUESTION, session_id="s"))

    assert first_turn["cached"] is True
    assert follow_up["cached"] is False
    history = service.sessions.get(service.session_key("default", "s"))
    assert len(history) == 4


def test_answer_cache_can_be_disabled(settings: Settings) -> None:
    settings.answer_cache_ttl_seconds = None
    service = _service(settings)
    assert service.answers is None
    asyncio.run(service.answer(QUESTION))
    assert asyncio.run(service.answer(QUESTION))["cached"] is False


def test_stream_replays_a_cached_answer(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        blocking = client.post("/query", json={"question": QUESTION}).json()
        with client.stream("POST", "/query/stream", json={"question": QUESTION}) as response:
            body = response.read().decode()
        again = client.post("/query", json={"question": QUESTION}).json()

    names = [
        line.removeprefix("event:").strip()
        for line in body.splitlines()
        if line.startswith("event:")
    ]
    data = [
        json.loads(line.removeprefix("data:"))
        for line in body.splitlines()
        if line.startswith("data:")
    ]
    assert names == ["sources", "token", "done"]
    assert data[1] == blocking["answer"]
    assert data[2]["cached"] is True
    assert blocking["cached"] is False
    assert again["cached"] is True
