import asyncio

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

from rag_service.filters import MetadataFilter
from rag_service.ingestion import load_directory, split_documents
from rag_service.providers import HashingEmbeddings
from rag_service.retrieval import (
    BM25Retriever,
    EmptyRetriever,
    HybridRetriever,
    RetrievalOptions,
    build_retriever,
    reciprocal_rank_fusion,
)
from rag_service.store import FaissIndex, IndexSnapshot
from tests.helpers import SAMPLE_DOCS


def _doc(chunk_id: str) -> Document:
    return Document(page_content=f"text {chunk_id}", metadata={"chunk_id": chunk_id})


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.documents


def _ids(documents: list[Document]) -> list[str]:
    return [d.metadata["chunk_id"] for d in documents]


def test_rrf_scores_follow_the_formula() -> None:
    fused = dict(
        (doc.metadata["chunk_id"], score)
        for doc, score in reciprocal_rank_fusion(
            [[_doc("a"), _doc("b")], [_doc("b"), _doc("c")]], k=60
        )
    )
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_prefers_documents_found_by_both_retrievers() -> None:
    fused = reciprocal_rank_fusion(
        [[_doc("a"), _doc("b"), _doc("c")], [_doc("d"), _doc("c"), _doc("b")]]
    )
    assert _ids([doc for doc, _ in fused])[:2] == ["b", "c"]


def test_rrf_weights_shift_the_ranking() -> None:
    lists = [[_doc("dense")], [_doc("lexical")]]
    assert _ids([d for d, _ in reciprocal_rank_fusion(lists, [2.0, 1.0])])[0] == "dense"
    assert _ids([d for d, _ in reciprocal_rank_fusion(lists, [1.0, 2.0])])[0] == "lexical"
    zero = reciprocal_rank_fusion(lists, [1.0, 0.0])
    assert zero[1][1] == 0.0


def test_rrf_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[_doc("a")]], [1.0, 1.0])


def test_hybrid_retriever_dedupes_and_truncates() -> None:
    retriever = HybridRetriever(
        retrievers=[
            StaticRetriever(documents=[_doc("a"), _doc("b"), _doc("c")]),
            StaticRetriever(documents=[_doc("c"), _doc("b"), _doc("e")]),
        ],
        weights=[1.0, 1.0],
        k=3,
    )

    results = retriever.invoke("anything")

    assert len(results) == 3
    assert len(set(_ids(results))) == 3
    assert set(_ids(results)) == {"b", "c", "a"}
    assert all("rrf_score" in d.metadata for d in results)


def test_hybrid_retriever_async_matches_sync() -> None:
    retriever = HybridRetriever(
        retrievers=[StaticRetriever(documents=[_doc("x"), _doc("y")])] * 2,
        weights=[1.0, 1.0],
        k=2,
    )
    assert _ids(asyncio.run(retriever.ainvoke("q"))) == _ids(retriever.invoke("q"))


@pytest.fixture(scope="module")
def corpus_snapshot(tmp_path_factory: pytest.TempPathFactory) -> IndexSnapshot:
    index = FaissIndex(tmp_path_factory.mktemp("index") / "idx", HashingEmbeddings(), "fake:test")
    index.add(split_documents(load_directory(SAMPLE_DOCS), 600, 80))
    return index.snapshot


def test_bm25_retriever_finds_exact_terms(corpus_snapshot: IndexSnapshot) -> None:
    results = BM25Retriever(snapshot=corpus_snapshot, k=3).invoke("Last-Event-ID header")
    assert results[0].metadata["source"] == "server-sent-events.md"
    assert results[0].metadata["bm25_score"] > 0


def test_bm25_retriever_does_not_mutate_indexed_chunks(corpus_snapshot: IndexSnapshot) -> None:
    BM25Retriever(snapshot=corpus_snapshot, k=3).invoke("nprobe")
    assert all("bm25_score" not in c.metadata for c in corpus_snapshot.chunks)


def test_bm25_retriever_returns_nothing_for_stop_words(corpus_snapshot: IndexSnapshot) -> None:
    assert BM25Retriever(snapshot=corpus_snapshot).invoke("the of and") == []


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_every_mode_returns_k_results(corpus_snapshot: IndexSnapshot, mode: str) -> None:
    retriever = build_retriever(corpus_snapshot, RetrievalOptions(mode=mode, k=3))  # type: ignore[arg-type]
    results = retriever.invoke("reciprocal rank fusion constant k")
    assert len(results) == 3
    assert results[0].metadata["source"] == "rank-fusion.md"


def test_mmr_configures_the_vector_store_retriever(corpus_snapshot: IndexSnapshot) -> None:
    options = RetrievalOptions(mode="dense", search_type="mmr", k=3, fetch_k=12, mmr_lambda=0.3)
    retriever = build_retriever(corpus_snapshot, options)

    assert isinstance(retriever, VectorStoreRetriever)
    assert retriever.search_type == "mmr"
    assert retriever.search_kwargs == {"k": 3, "fetch_k": 12, "lambda_mult": 0.3}
    assert len(retriever.invoke("chunk overlap")) == 3


def test_hybrid_candidates_are_capped_by_fetch_k(corpus_snapshot: IndexSnapshot) -> None:
    retriever = build_retriever(corpus_snapshot, RetrievalOptions(k=4, fetch_k=6))
    assert isinstance(retriever, HybridRetriever)
    dense, lexical = retriever.retrievers
    assert isinstance(dense, VectorStoreRetriever)
    assert dense.search_kwargs["k"] == 6
    assert isinstance(lexical, BM25Retriever)
    assert lexical.k == 6


def test_empty_snapshot_gives_empty_retriever() -> None:
    retriever = build_retriever(IndexSnapshot(), RetrievalOptions())
    assert isinstance(retriever, EmptyRetriever)
    assert retriever.invoke("anything") == []


def test_options_overrides_ignore_none() -> None:
    options = RetrievalOptions(k=4).with_overrides(k=None, search_type="mmr")
    assert options.k == 4
    assert options.search_type == "mmr"


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("search_type", ["similarity", "mmr"])
def test_metadata_filter_applies_to_every_mode(
    corpus_snapshot: IndexSnapshot, mode: str, search_type: str
) -> None:
    options = RetrievalOptions(
        mode=mode,  # type: ignore[arg-type]
        search_type=search_type,  # type: ignore[arg-type]
        k=3,
        metadata_filter=MetadataFilter.from_mapping({"source": ["mmr.md", "glossary.txt"]}),
    )
    results = build_retriever(corpus_snapshot, options).invoke("reciprocal rank fusion constant k")

    assert results
    assert {d.metadata["source"] for d in results} <= {"mmr.md", "glossary.txt"}


def test_chunks_are_stamped_with_the_index_tenant(corpus_snapshot: IndexSnapshot) -> None:
    assert {c.metadata["tenant_id"] for c in corpus_snapshot.chunks} == {"default"}
