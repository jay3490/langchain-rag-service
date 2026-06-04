from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from rag_service.api import create_app
from rag_service.config import Settings
from rag_service.filters import MetadataFilter
from rag_service.ingestion import load_directory, split_documents
from rag_service.providers import HashingEmbeddings
from rag_service.qdrant_store import QdrantIndex, point_id, qdrant_filter, to_qdrant_filter
from rag_service.retrieval import RetrievalOptions, build_retriever
from rag_service.store import IndexCompatibilityError
from tests.helpers import SAMPLE_DOCS

EMBEDDING_ID = "fake:hashing-64"


def _chunks(source: str, text: str) -> list[Document]:
    return split_documents([Document(page_content=text, metadata={"source": source})], 200, 20)


@pytest.fixture
def client() -> Iterator[QdrantClient]:
    qdrant = QdrantClient(location=":memory:")
    yield qdrant
    qdrant.close()


def _index(client: QdrantClient, tenant: str = "default", **kwargs: object) -> QdrantIndex:
    return QdrantIndex(
        client,
        f"rag_{tenant}",
        HashingEmbeddings(dim=int(kwargs.pop("dim", 64))),  # type: ignore[call-overload]
        str(kwargs.pop("embedding_id", EMBEDDING_ID)),
        tenant,
        meta_collection="rag_meta",
        sync_seconds=float(kwargs.pop("sync_seconds", 0.0)),  # type: ignore[arg-type]
    )


def test_point_ids_are_uuids_of_chunk_ids() -> None:
    assert point_id("0" * 31 + "1") == "00000000-0000-0000-0000-000000000001"


def test_filter_translation() -> None:
    assert to_qdrant_filter(MetadataFilter()) is None
    translated = to_qdrant_filter(
        MetadataFilter.from_mapping({"tenant_id": "acme", "source": ["a.md", "b.md"]})
    )
    assert translated is not None
    source, tenant = translated.must  # conditions are sorted by key
    assert isinstance(tenant, models.FieldCondition)
    assert tenant.key == "metadata.tenant_id"
    assert isinstance(source, models.Filter)
    assert len(source.should) == 2
    assert set(qdrant_filter(MetadataFilter.from_mapping({"a": 1}), 40)) == {"filter"}


def test_add_replace_and_remove(client: QdrantClient) -> None:
    index = _index(client)
    index.load()
    assert index.snapshot.is_empty

    index.add(_chunks("a.md", "old content about apples. " * 20))
    index.add(_chunks("b.md", "bananas are yellow."))
    result = index.add(_chunks("a.md", "new content about cherries."))

    assert result.chunks_removed > 0
    assert index.snapshot.source_counts() == {"a.md": 1, "b.md": 1}
    assert client.count("rag_default").count == 2
    hits = index.snapshot.vectorstore.similarity_search("cherries", k=1)
    assert hits[0].metadata["source"] == "a.md"
    assert hits[0].metadata["tenant_id"] == "default"

    assert index.add(_chunks("b.md", "bananas are yellow.")).unchanged == ["b.md"]
    assert index.remove_source("b.md") == 1
    assert client.count("rag_default").count == 1

    index.clear()
    assert index.snapshot.is_empty
    assert not client.collection_exists("rag_default")


def test_reload_from_the_collection(client: QdrantClient) -> None:
    index = _index(client)
    index.add(_chunks("a.md", "persisted text about qdrant collections. " * 10))

    reloaded = _index(client)
    reloaded.load()

    assert [c.page_content for c in reloaded.snapshot.chunks] == [
        c.page_content for c in index.snapshot.chunks
    ]
    assert reloaded.snapshot.version == index.snapshot.version


def test_embedding_model_mismatch_is_refused(client: QdrantClient) -> None:
    _index(client).add(_chunks("a.md", "text"))
    with pytest.raises(IndexCompatibilityError, match=EMBEDDING_ID):
        _index(client, embedding_id="fake:hashing-32").load()


def test_sync_picks_up_writes_from_another_replica(client: QdrantClient) -> None:
    first, second = _index(client), _index(client)
    first.load()
    second.load()

    first.add(_chunks("a.md", "written by the first replica"))

    assert second.snapshot.is_empty
    assert second.sync() is True
    assert second.snapshot.source_counts() == {"a.md": 1}
    assert second.sync() is False


def test_sync_is_rate_limited(client: QdrantClient) -> None:
    first, second = _index(client), _index(client, sync_seconds=3600)
    second.load()
    first.add(_chunks("a.md", "not seen until the interval passes"))
    assert second.sync() is False


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("search_type", ["similarity", "mmr"])
def test_retrieval_with_payload_filters(client: QdrantClient, mode: str, search_type: str) -> None:
    index = _index(client, dim=384)
    index.add(split_documents(load_directory(SAMPLE_DOCS), 600, 80))
    snapshot = index.snapshot

    unfiltered = build_retriever(
        snapshot,
        RetrievalOptions(mode=mode, search_type=search_type, k=3),  # type: ignore[arg-type]
    ).invoke("reciprocal rank fusion constant k")
    assert unfiltered[0].metadata["source"] == "rank-fusion.md"

    options = RetrievalOptions(
        mode=mode,  # type: ignore[arg-type]
        search_type=search_type,  # type: ignore[arg-type]
        k=3,
        metadata_filter=MetadataFilter.from_mapping({"source": ["mmr.md", "glossary.txt"]}),
    )
    filtered = build_retriever(snapshot, options).invoke("reciprocal rank fusion constant k")
    assert filtered
    assert {d.metadata["source"] for d in filtered} <= {"mmr.md", "glossary.txt"}


def test_api_runs_on_qdrant(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        vector_store="qdrant",
        qdrant_url=":memory:",
        index_dir=tmp_path / "index",
        ingest_root=SAMPLE_DOCS.parent,
        chunk_size=600,
        chunk_overlap=80,
    )
    with TestClient(create_app(settings)) as api:
        assert api.post("/ingest/directory", json={"path": "docs"}).status_code == 200
        body = api.post(
            "/query", json={"question": "What value is the RRF constant k commonly set to?"}
        ).json()
    assert "60" in body["answer"]
    assert body["sources"][0]["source"] == "rank-fusion.md"
