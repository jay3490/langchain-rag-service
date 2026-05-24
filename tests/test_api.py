from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_service.api import create_app
from rag_service.config import Settings
from rag_service.prompts import NO_ANSWER
from tests.helpers import SAMPLE_DOCS, build_pdf


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        index_dir=tmp_path / "index",
        ingest_root=SAMPLE_DOCS.parent,
        chunk_size=600,
        chunk_overlap=80,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def loaded(client: TestClient) -> TestClient:
    response = client.post("/ingest/directory", json={"path": "docs"})
    assert response.status_code == 200
    return client


def test_health_reports_provider_and_size(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["provider"] == "fake"
    assert body["retriever_mode"] == "hybrid"
    assert body["chunks"] == 0


def test_query_before_ingest_is_a_conflict(client: TestClient) -> None:
    response = client.post("/query", json={"question": "What is BM25?"})
    assert response.status_code == 409
    assert "ingest" in response.json()["detail"]


def test_ingest_directory(client: TestClient) -> None:
    body = client.post("/ingest/directory", json={"path": "docs"}).json()

    assert "bm25.md" in body["sources"]
    assert body["chunks_added"] == body["total_chunks"] > 0
    assert client.get("/health").json()["chunks"] == body["total_chunks"]

    again = client.post("/ingest/directory", json={"path": "docs"}).json()
    assert again["chunks_added"] == 0
    assert again["unchanged"] == body["sources"]


@pytest.mark.parametrize(("path", "code"), [("../..", 400), ("missing", 404)])
def test_ingest_directory_rejects_bad_paths(client: TestClient, path: str, code: int) -> None:
    assert client.post("/ingest/directory", json={"path": path}).status_code == code


def test_upload_text_and_pdf(client: TestClient) -> None:
    files = [
        ("files", ("notes.md", b"# Notes\n\nThe deploy window is Tuesday.", "text/markdown")),
        (
            "files",
            ("../secret/manual.pdf", build_pdf(["Rotate keys every 90 days"]), "application/pdf"),
        ),
    ]
    body = client.post("/ingest/files", files=files).json()
    assert body["sources"] == ["manual.pdf", "notes.md"]

    answer = client.post("/query", json={"question": "How often should keys rotate?"}).json()
    assert "90 days" in answer["answer"]
    cited = answer["sources"][0]
    assert cited["source"] == "manual.pdf"
    assert cited["page"] == 1
    assert cited["label"] == "manual.pdf (page 1)"


def test_upload_rejects_unsupported_types(client: TestClient) -> None:
    response = client.post("/ingest/files", files=[("files", ("x.png", b"\x89PNG", "image/png"))])
    assert response.status_code == 415


def test_upload_size_limit(settings: Settings) -> None:
    settings.max_upload_bytes = 10
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/ingest/files", files=[("files", ("a.txt", b"x" * 11, "text/plain"))]
        )
    assert response.status_code == 413


def test_query_returns_a_cited_answer(loaded: TestClient) -> None:
    body = loaded.post(
        "/query", json={"question": "What value is the RRF constant k commonly set to?"}
    ).json()

    assert body["question"] == body["standalone_question"]
    assert "60" in body["answer"]
    assert "[" in body["answer"]
    assert len(body["sources"]) == 4
    assert body["sources"][0]["source"] == "rank-fusion.md"
    assert all(s["score"] is not None for s in body["sources"])


def test_query_overrides(loaded: TestClient) -> None:
    body = loaded.post(
        "/query",
        json={"question": "chunk overlap", "k": 2, "retriever_mode": "dense", "search_type": "mmr"},
    ).json()
    assert len(body["sources"]) == 2
    assert all(s["score"] is None for s in body["sources"])


def test_unanswerable_question_declines(loaded: TestClient) -> None:
    body = loaded.post("/query", json={"question": "zebra giraffe"}).json()
    assert body["answer"] == NO_ANSWER


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "ok", "k": 0}, {"question": "ok", "session_id": "bad id!"}],
)
def test_query_validation(loaded: TestClient, payload: dict[str, object]) -> None:
    assert loaded.post("/query", json=payload).status_code == 422


def test_sessions_record_turns_and_rewrite_follow_ups(loaded: TestClient) -> None:
    first = loaded.post(
        "/query",
        json={"question": "What does the BM25 parameter b control?", "session_id": "s1"},
    ).json()
    follow_up = loaded.post(
        "/query", json={"question": "What is its usual setting?", "session_id": "s1"}
    ).json()

    assert first["standalone_question"] == first["question"]
    assert follow_up["standalone_question"] == (
        "What does the BM25 parameter b control? What is its usual setting?"
    )
    assert follow_up["sources"][0]["source"] == "bm25.md"
    assert follow_up["answer"].endswith("[1]")

    history = loaded.get("/sessions/s1").json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[2]["content"] == "What is its usual setting?"

    assert loaded.delete("/sessions/s1").status_code == 204
    assert loaded.get("/sessions/s1").json()["messages"] == []
    assert loaded.delete("/sessions/s1").status_code == 404


def test_documents_list_and_delete(loaded: TestClient) -> None:
    listing = loaded.get("/documents").json()
    assert listing["sources"]["glossary.txt"] >= 1

    assert loaded.delete("/documents/glossary.txt").status_code == 204
    assert "glossary.txt" not in loaded.get("/documents").json()["sources"]
    assert loaded.delete("/documents/glossary.txt").status_code == 404


def test_index_survives_restart(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        client.post("/ingest/directory", json={"path": "docs"})
        chunks = client.get("/health").json()["chunks"]

    with TestClient(create_app(settings)) as restarted:
        assert restarted.get("/health").json()["chunks"] == chunks


def test_ingest_on_startup(settings: Settings) -> None:
    settings.ingest_on_startup = SAMPLE_DOCS
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["chunks"] > 0
