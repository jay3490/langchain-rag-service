import json
from pathlib import Path

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag_service.ingestion import split_documents
from rag_service.providers import HashingEmbeddings
from rag_service.store import FaissIndex, IndexCompatibilityError


def _chunks(source: str, text: str) -> list[Document]:
    return split_documents([Document(page_content=text, metadata={"source": source})], 200, 20)


@pytest.fixture
def index(tmp_path: Path) -> FaissIndex:
    return FaissIndex(tmp_path / "index", HashingEmbeddings(dim=64), "fake:hashing-64")


def test_new_index_is_empty(index: FaissIndex) -> None:
    index.load()
    assert index.snapshot.is_empty
    assert index.snapshot.vectorstore is None


def test_add_builds_vector_and_bm25_indexes(index: FaissIndex) -> None:
    result = index.add(_chunks("a.md", "Alpha beta gamma. " * 30))

    snapshot = index.snapshot
    assert result.sources == ["a.md"]
    assert result.chunks_added == len(snapshot.chunks) > 1
    assert snapshot.vectorstore is not None
    assert snapshot.vectorstore.index.ntotal == len(snapshot.chunks)
    assert snapshot.bm25 is not None


def test_reingesting_a_source_replaces_its_chunks(index: FaissIndex) -> None:
    index.add(_chunks("a.md", "old content about apples. " * 20))
    index.add(_chunks("b.md", "bananas are yellow."))

    result = index.add(_chunks("a.md", "new content about cherries."))

    contents = [c.page_content for c in index.snapshot.chunks]
    assert result.chunks_removed > 0
    assert not any("apples" in c for c in contents)
    assert any("cherries" in c for c in contents)
    assert index.snapshot.source_counts() == {"a.md": 1, "b.md": 1}


def test_unchanged_sources_are_not_re_embedded(tmp_path: Path) -> None:
    class CountingEmbeddings(HashingEmbeddings):
        calls = 0

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            CountingEmbeddings.calls += len(texts)
            return super().embed_documents(texts)

    index = FaissIndex(tmp_path / "idx", CountingEmbeddings(dim=64), "fake:hashing-64")
    a, b = _chunks("a.md", "alpha text"), _chunks("b.md", "beta text")
    index.add(a + b)
    calls_after_first = CountingEmbeddings.calls

    again = index.add(a + _chunks("b.md", "beta text, edited"))

    assert again.unchanged == ["a.md"]
    assert again.chunks_added == 1
    assert again.chunks_removed == 1
    assert CountingEmbeddings.calls == calls_after_first + 1
    assert index.add(a).chunks_added == 0


def test_index_persists_and_reloads(index: FaissIndex, tmp_path: Path) -> None:
    index.add(_chunks("a.md", "persisted text about faiss indexes."))

    reloaded = FaissIndex(index.directory, HashingEmbeddings(dim=64), "fake:hashing-64")
    reloaded.load()

    assert [c.page_content for c in reloaded.snapshot.chunks] == [
        c.page_content for c in index.snapshot.chunks
    ]
    hits = reloaded.snapshot.vectorstore.similarity_search("faiss indexes", k=1)
    assert hits[0].metadata["source"] == "a.md"


def test_loading_with_a_different_embedding_model_fails(index: FaissIndex) -> None:
    index.add(_chunks("a.md", "some text"))
    other = FaissIndex(index.directory, HashingEmbeddings(dim=32), "fake:hashing-32")

    with pytest.raises(IndexCompatibilityError, match="fake:hashing-64"):
        other.load()


def test_readers_keep_their_snapshot_during_writes(index: FaissIndex) -> None:
    index.add(_chunks("a.md", "first version"))
    before = index.snapshot

    index.add(_chunks("b.md", "second document"))

    assert len(before.chunks) == 1
    assert before.vectorstore is not None
    assert before.vectorstore.index.ntotal == 1
    assert len(index.snapshot.chunks) == 2


def test_remove_source_and_clear(index: FaissIndex) -> None:
    index.add(_chunks("a.md", "keep me"))
    index.add(_chunks("b.md", "remove me"))

    assert index.remove_source("b.md") == 1
    assert index.remove_source("missing.md") == 0
    assert index.snapshot.source_counts() == {"a.md": 1}

    assert index.remove_source("a.md") == 1
    assert index.snapshot.is_empty
    assert not (index.directory / "index.faiss").exists()

    index.add(_chunks("c.md", "fresh start"))
    index.clear()
    assert index.snapshot.is_empty
    assert not (index.directory / "manifest.json").exists()


def test_chunks_from_before_tenancy_are_adopted_on_load(tmp_path: Path) -> None:
    directory = tmp_path / "legacy"
    chunks = _chunks("old.md", "written by an earlier version")
    ids = [c.metadata["chunk_id"] for c in chunks]
    embeddings = HashingEmbeddings(dim=64)
    FAISS.from_documents(chunks, embeddings, ids=ids, normalize_L2=True).save_local(str(directory))
    (directory / "manifest.json").write_text(json.dumps({"embedding_id": "fake:hashing-64"}))

    index = FaissIndex(directory, embeddings, "fake:hashing-64", tenant_id="acme")
    index.load()

    assert {c.metadata["tenant_id"] for c in index.snapshot.chunks} == {"acme"}
    hit = index.snapshot.vectorstore.similarity_search("earlier version", k=1)[0]
    assert hit.metadata["tenant_id"] == "acme"

    reloaded = FaissIndex(directory, embeddings, "fake:hashing-64", tenant_id="acme")
    reloaded.load()
    assert reloaded.snapshot.chunks[0].metadata["tenant_id"] == "acme"


def test_snapshot_version_tracks_content(index: FaissIndex) -> None:
    assert index.snapshot.version == "empty"
    index.add(_chunks("a.md", "first"))
    first = index.snapshot.version
    index.add(_chunks("b.md", "second"))
    assert index.snapshot.version != first
    index.remove_source("b.md")
    assert index.snapshot.version == first
