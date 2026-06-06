from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from rag_service.api import create_app
from rag_service.auth import hash_api_key
from rag_service.config import Settings
from rag_service.indexes import migrate_single_index_layout
from rag_service.ingestion import split_documents
from rag_service.providers import HashingEmbeddings
from rag_service.retrieval import BM25Retriever, HybridRetriever
from rag_service.service import RagService
from rag_service.store import FaissIndex
from tests.helpers import SAMPLE_DOCS

KEYS = {"acme": "key-for-acme", "globex": "key-for-globex"}
QUESTION = "What value is the RRF constant k commonly set to?"


def headers(tenant: str) -> dict[str, str]:
    return {"X-API-Key": KEYS[tenant]}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        index_dir=tmp_path / "index",
        ingest_root=SAMPLE_DOCS.parent,
        chunk_size=600,
        chunk_overlap=80,
        api_keys={hash_api_key(key): tenant for tenant, key in KEYS.items()},
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/documents"),
        ("post", "/query"),
        ("post", "/query/stream"),
        ("post", "/ingest/directory"),
        ("get", "/sessions/s1"),
        ("delete", "/documents/a.md"),
    ],
)
def test_requests_without_a_valid_key_are_rejected(
    client: TestClient, method: str, path: str
) -> None:
    kwargs = {"json": {"question": "q", "path": "docs"}} if method == "post" else {}
    for extra in ({}, {"X-API-Key": "wrong"}, {"Authorization": "Bearer wrong"}):
        response = getattr(client, method)(path, headers=extra, **kwargs)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"


def test_health_stays_open(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["auth_enabled"] is True


def test_bearer_tokens_are_accepted(client: TestClient) -> None:
    bearer = {"Authorization": f"Bearer {KEYS['acme']}"}
    assert client.get("/documents", headers=bearer).status_code == 200


def test_tenants_only_see_their_own_documents(client: TestClient, settings: Settings) -> None:
    ingest = client.post("/ingest/directory", json={"path": "docs"}, headers=headers("acme"))
    assert ingest.status_code == 200

    empty = client.post("/query", json={"question": QUESTION}, headers=headers("globex"))
    assert empty.status_code == 409
    assert client.get("/documents", headers=headers("globex")).json()["total_chunks"] == 0

    plan = b"# Plan\n\nGlobex ships the reactor in March."
    files = [("files", ("plan.md", plan, "text/markdown"))]
    assert client.post("/ingest/files", files=files, headers=headers("globex")).status_code == 200

    acme = client.post("/query", json={"question": QUESTION}, headers=headers("acme")).json()
    globex = client.post(
        "/query", json={"question": "When does the reactor ship?"}, headers=headers("globex")
    ).json()
    assert "plan.md" not in {s["source"] for s in acme["sources"]}
    assert {s["source"] for s in globex["sources"]} == {"plan.md"}
    assert "March" in globex["answer"]

    assert client.delete("/documents/plan.md", headers=headers("acme")).status_code == 404
    assert (settings.index_dir / "acme" / "manifest.json").exists()
    assert (settings.index_dir / "globex" / "manifest.json").exists()


def test_sessions_are_isolated_per_tenant(client: TestClient) -> None:
    for tenant in KEYS:
        client.post("/ingest/directory", json={"path": "docs"}, headers=headers(tenant))
    client.post(
        "/query",
        json={"question": "What is MMR?", "session_id": "shared"},
        headers=headers("acme"),
    )

    assert client.get("/sessions/shared", headers=headers("globex")).json()["messages"] == []
    assert len(client.get("/sessions/shared", headers=headers("acme")).json()["messages"]) == 2
    assert client.delete("/sessions/shared", headers=headers("globex")).status_code == 404


def test_startup_ingest_goes_to_the_startup_tenant(settings: Settings) -> None:
    settings.ingest_on_startup = SAMPLE_DOCS
    settings.startup_tenant = "acme"
    with TestClient(create_app(settings)) as client:
        assert client.get("/documents", headers=headers("acme")).json()["total_chunks"] > 0
        assert client.get("/documents", headers=headers("globex")).json()["total_chunks"] == 0


def test_retrieval_always_filters_on_the_tenant(settings: Settings) -> None:
    service = RagService(settings)
    service.ingest_directory(Path("docs"), tenant_id="acme")

    retriever = service._retriever("acme", service.options())

    assert isinstance(retriever, HybridRetriever)
    lexical = retriever.retrievers[1]
    assert isinstance(lexical, BM25Retriever)
    assert lexical.metadata_filter.as_dict() == {"tenant_id": ["acme"]}
    assert all(d.metadata["tenant_id"] == "acme" for d in retriever.invoke(QUESTION))


def test_pre_tenancy_index_is_moved_to_the_default_tenant(tmp_path: Path) -> None:
    legacy = FaissIndex(tmp_path, HashingEmbeddings(dim=64), "fake:hashing-64")
    chunks = split_documents([Document(page_content="old", metadata={"source": "a.md"})], 100, 10)
    legacy.add(chunks)

    assert migrate_single_index_layout(tmp_path) is True
    assert not (tmp_path / "manifest.json").exists()
    moved = FaissIndex(tmp_path / "default", HashingEmbeddings(dim=64), "fake:hashing-64")
    moved.load()
    assert moved.snapshot.source_counts() == {"a.md": 1}
    assert migrate_single_index_layout(tmp_path) is False
