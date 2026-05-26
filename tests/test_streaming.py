import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_service.api import create_app
from rag_service.config import Settings
from rag_service.service import RagService
from tests.helpers import SAMPLE_DOCS


def parse_sse(body: str) -> list[tuple[str, Any]]:
    """Parse a text/event-stream body into (event, decoded JSON data) pairs."""
    events: list[tuple[str, Any]] = []
    for block in body.strip().split("\n\n"):
        name, data_lines = "message", []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
        if data_lines:
            events.append((name, json.loads("\n".join(data_lines))))
    return events


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


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def stream(client: TestClient, payload: dict[str, Any]) -> list[tuple[str, Any]]:
    with client.stream("POST", "/query/stream", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return parse_sse(response.read().decode())


def test_stream_emits_sources_tokens_then_done(client: TestClient) -> None:
    question = "What value is the RRF constant k commonly set to?"
    events = stream(client, {"question": question})

    names = [name for name, _ in events]
    assert names[0] == "sources"
    assert names[-1] == "done"
    assert set(names[1:-1]) == {"token"}
    assert len(names) > 3

    sources = events[0][1]
    assert sources["standalone_question"] == question
    assert sources["sources"][0]["source"] == "rank-fusion.md"

    tokens = "".join(data for name, data in events if name == "token")
    assert tokens == events[-1][1]["answer"]


def test_stream_matches_the_blocking_endpoint(client: TestClient) -> None:
    payload = {"question": "What does the BM25 parameter k1 control?", "k": 3}
    blocking = client.post("/query", json=payload).json()
    events = stream(client, payload)

    assert events[-1][1]["answer"] == blocking["answer"]
    assert events[0][1]["sources"] == blocking["sources"]


def test_stream_records_the_turn_in_the_session(client: TestClient) -> None:
    stream(client, {"question": "What is MMR?", "session_id": "stream-1"})
    follow_up = stream(
        client, {"question": "Which parameter balances it?", "session_id": "stream-1"}
    )

    assert follow_up[0][1]["standalone_question"] == "What is MMR? Which parameter balances it?"
    assert follow_up[-1][1]["session_id"] == "stream-1"
    history = client.get("/sessions/stream-1").json()["messages"]
    assert len(history) == 4
    assert history[1]["content"] != ""


def test_stream_on_empty_index_is_a_conflict(settings: Settings) -> None:
    settings.ingest_on_startup = None
    with TestClient(create_app(settings)) as client:
        response = client.post("/query/stream", json={"question": "anything"})
    assert response.status_code == 409


def test_failure_mid_stream_becomes_an_error_event(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = RagService(settings)

    async def failing(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise RuntimeError("model unavailable")
        yield  # unreachable; makes this function an async generator

    monkeypatch.setattr(service, "stream_answer", failing)
    with TestClient(create_app(settings, service=service)) as client:
        events = stream(client, {"question": "What is BM25?"})

    assert events == [("error", {"detail": "answer generation failed"})]
