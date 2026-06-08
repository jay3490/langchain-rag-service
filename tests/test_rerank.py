import asyncio
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

from rag_service.api import create_app
from rag_service.config import Settings
from rag_service.providers import ExtractiveFakeChatModel
from rag_service.rerank import (
    CrossEncoderReranker,
    LexicalReranker,
    LLMReranker,
    RerankingRetriever,
    apply_scores,
    build_reranker,
    parse_grade,
)
from rag_service.service import RagService
from tests.helpers import SAMPLE_DOCS

DOCS = [
    Document(page_content="Vector search finds nearest neighbours.", metadata={"chunk_id": "a"}),
    Document(
        page_content="BM25 parameter k1 controls term frequency saturation.",
        metadata={"chunk_id": "b"},
    ),
    Document(
        page_content="The parameter b controls length normalisation.", metadata={"chunk_id": "c"}
    ),
]
QUERY = "What does the BM25 parameter k1 control?"


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.documents


def ids(documents: list[Document]) -> list[str]:
    return [d.metadata["chunk_id"] for d in documents]


def test_apply_scores_orders_truncates_and_annotates() -> None:
    ranked = apply_scores(DOCS, [0.2, 0.9, 0.2], top_n=2)
    assert ids(ranked) == ["b", "a"]
    assert ranked[0].metadata["rerank_score"] == 0.9
    assert "rerank_score" not in DOCS[1].metadata


def test_lexical_reranker_prefers_distinctive_terms_and_phrases() -> None:
    scores = LexicalReranker().score(QUERY, DOCS)
    assert scores[1] == max(scores)
    assert scores[0] == 0.0
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_lexical_reranker_handles_empty_input() -> None:
    assert LexicalReranker().score("the of", DOCS) == [0.0, 0.0, 0.0]
    assert LexicalReranker().score(QUERY, []) == []


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("7", 0.7), ("Score: 10", 1.0), ("0", 0.0), (" 3/10", 0.3), ("relevant", None), ("42", None)],
)
def test_parse_grade(reply: str, expected: float | None) -> None:
    assert parse_grade(reply) == expected


def test_llm_reranker_uses_the_chat_model() -> None:
    reranker = LLMReranker(ExtractiveFakeChatModel())
    ranked = reranker.rerank(QUERY, DOCS, top_n=3)
    assert ids(ranked)[0] == "b"
    assert ranked[0].metadata["rerank_score"] == 0.8  # 3 of 4 query words: "controls" != "control"


def test_llm_reranker_async_matches_sync() -> None:
    reranker = LLMReranker(ExtractiveFakeChatModel())
    assert asyncio.run(reranker.ascore(QUERY, DOCS)) == reranker.score(QUERY, DOCS)


def test_llm_reranker_falls_back_to_lexical_scores_for_unparseable_replies() -> None:
    replies = iter([AIMessage(content="9"), AIMessage(content="no idea"), AIMessage(content="1")])
    reranker = LLMReranker(GenericFakeChatModel(messages=replies), max_concurrency=1)
    scores = reranker.score(QUERY, DOCS)
    assert scores[0] == 0.9
    assert scores[1] == LexicalReranker().score(QUERY, DOCS)[1]
    assert scores[2] == 0.1


def test_cross_encoder_reranker_with_a_stub_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubCrossEncoder:
        def __init__(self, name: str) -> None:
            self.name = name

        def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
            return [3.0 if "k1" in passage else -3.0 for _, passage in pairs]

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = StubCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    reranker = CrossEncoderReranker("stub-model")
    scores = reranker.score(QUERY, DOCS)
    assert scores[1] == pytest.approx(1 / (1 + 2.718281828**-3), rel=1e-6)
    assert scores[0] < 0.1
    assert reranker.score(QUERY, []) == []


def test_cross_encoder_without_the_extra_explains_how_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match=r"\[rerank\]"):
        CrossEncoderReranker("any")


@pytest.mark.parametrize(
    ("kind", "expected"), [("none", type(None)), ("lexical", LexicalReranker), ("llm", LLMReranker)]
)
def test_build_reranker(kind: str, expected: type) -> None:
    settings = Settings(_env_file=None, reranker=kind)  # type: ignore[arg-type]
    assert isinstance(build_reranker(settings, ExtractiveFakeChatModel()), expected)


def test_reranking_retriever_reorders_candidates() -> None:
    retriever = RerankingRetriever(
        base=StaticRetriever(documents=DOCS), reranker=LexicalReranker(), k=1
    )
    assert ids(retriever.invoke(QUERY)) == ["b"]
    assert ids(asyncio.run(retriever.ainvoke(QUERY))) == ["b"]


# Service and API -----------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        index_dir=tmp_path / "index",
        ingest_root=SAMPLE_DOCS.parent,
        ingest_on_startup=SAMPLE_DOCS,
        chunk_size=600,
        chunk_overlap=80,
        reranker="llm",
        rerank_candidates=8,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_service_overfetches_for_the_reranker(settings: Settings) -> None:
    service = RagService(settings)
    service.start()
    retriever = service._retriever("default", service.options(k=3))

    assert isinstance(retriever, RerankingRetriever)
    assert retriever.k == 3
    assert len(retriever.base.invoke("chunk overlap")) == 8
    assert not isinstance(
        service._retriever("default", service.options(rerank=False)), RerankingRetriever
    )


def test_query_reports_rerank_scores(client: TestClient) -> None:
    body: dict[str, Any] = client.post(
        "/query", json={"question": "What value is the RRF constant k commonly set to?"}
    ).json()
    scores = [s["rerank_score"] for s in body["sources"]]
    assert len(scores) == 4
    assert scores == sorted(scores, reverse=True)
    assert body["sources"][0]["source"] == "rank-fusion.md"

    plain = client.post(
        "/query",
        json={"question": "What value is the RRF constant k commonly set to?", "rerank": False},
    ).json()
    assert all(s["rerank_score"] is None for s in plain["sources"])
