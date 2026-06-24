import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from rag_service.api import create_app
from rag_service.chain import build_rag_chain
from rag_service.config import Settings
from rag_service.guardrails import Guardrails, check_citations, retrieval_confidence
from rag_service.prompts import NO_ANSWER
from rag_service.service import RagService
from tests.helpers import SAMPLE_DOCS
from tests.test_chain import DOCS, StaticRetriever


def test_confidence_is_best_query_word_coverage() -> None:
    docs = [
        Document(page_content="BM25 saturates term frequency."),
        Document(page_content="k1 controls BM25 saturation of term frequency."),
    ]
    assert retrieval_confidence("What does k1 control in BM25?", docs) == pytest.approx(2 / 3)
    assert retrieval_confidence("zebra", docs) == 0.0
    assert retrieval_confidence("the of", docs) == 0.0
    assert retrieval_confidence("anything", []) == 0.0


def test_confidence_prefers_rerank_scores() -> None:
    docs = [
        Document(page_content="unrelated", metadata={"rerank_score": 0.4}),
        Document(page_content="also unrelated", metadata={"rerank_score": 0.7}),
    ]
    assert retrieval_confidence("question words", docs) == 0.7


@pytest.mark.parametrize(
    ("answer", "count", "cleaned", "cited", "invalid"),
    [
        ("A [1]. B [2].", 2, "A [1]. B [2].", [1, 2], []),
        ("A [1, 3]. B [2].", 2, "A [1]. B [2].", [1, 2], [3]),
        ("A [4].", 2, "A.", [], [4]),
        ("A [0] and B [2]", 2, "A and B [2]", [2], [0]),
        ("No citations here.", 2, "No citations here.", [], []),
        ("Repeated [1] [1].", 1, "Repeated [1] [1].", [1], []),
    ],
)
def test_check_citations(
    answer: str, count: int, cleaned: str, cited: list[int], invalid: list[int]
) -> None:
    check = check_citations(answer, count)
    assert (check.answer, check.cited, check.invalid) == (cleaned, cited, invalid)


def test_review_policies() -> None:
    docs = DOCS[:1]
    strip = Guardrails(citation_policy="strip").review("Fact [1]. Other [3].", docs, 1.0)
    assert (strip.answer, strip.citations, strip.invalid_citations) == (
        "Fact [1]. Other.",
        [1],
        [3],
    )

    off = Guardrails(citation_policy="off").review("Fact [1]. Other [3].", docs, 1.0)
    assert off.answer == "Fact [1]. Other [3]."
    assert off.invalid_citations == [3]

    refuse = Guardrails(citation_policy="refuse")
    assert refuse.review("Fact [3].", docs, 1.0).refusal == "invalid_citations"
    assert refuse.review("Uncited fact.", docs, 1.0).answer == NO_ANSWER
    assert refuse.review("Fact [1].", docs, 1.0).refusal is None


def test_review_reports_why_it_declined() -> None:
    guard = Guardrails(min_confidence=0.5)
    assert guard.review("x [1]", [], 1.0).refusal == "no_documents"
    assert guard.review("x [1]", DOCS, 0.2).refusal == "low_confidence"
    assert guard.review(NO_ANSWER, DOCS, 0.9).refusal == "not_in_sources"


def test_low_confidence_skips_the_model() -> None:
    llm = GenericFakeChatModel(messages=iter([]))  # raises if it is ever called
    chain = build_rag_chain(
        llm, StaticRetriever(documents=DOCS), guardrails=Guardrails(min_confidence=0.9)
    )
    result = chain.invoke({"question": "What does k1 do in giraffes?"})
    assert result["answer"] == NO_ANSWER
    assert result["refusal"] == "low_confidence"
    assert result["confidence"] < 0.9
    assert len(result["sources"]) == 2


def test_hallucinated_citations_are_stripped_from_model_output() -> None:
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="Saturation [1][7].")]))
    result = build_rag_chain(llm, StaticRetriever(documents=DOCS)).invoke({"question": "k1?"})
    assert result["answer"] == "Saturation [1]."
    assert result["citations"] == [1]
    assert result["invalid_citations"] == [7]


# Service and API -----------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        index_dir=tmp_path / "index",
        ingest_on_startup=SAMPLE_DOCS,
        chunk_size=600,
        chunk_overlap=80,
        min_retrieval_confidence=0.5,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_api_reports_confidence_and_citations(client: TestClient) -> None:
    body = client.post(
        "/query", json={"question": "What value is the RRF constant k commonly set to?"}
    ).json()
    assert body["confidence"] >= 0.5
    assert body["refusal"] is None
    assert body["citations"]
    assert set(body["citations"]) <= {s["id"] for s in body["sources"]}


def test_api_declines_below_the_confidence_floor(client: TestClient) -> None:
    body = client.post("/query", json={"question": "giraffe zebra fusion"}).json()
    assert body["answer"] == NO_ANSWER
    assert body["refusal"] == "low_confidence"
    assert body["confidence"] < 0.5


def test_stream_done_event_carries_the_review(client: TestClient) -> None:
    with client.stream(
        "POST", "/query/stream", json={"question": "giraffe zebra fusion"}
    ) as response:
        lines = response.read().decode().splitlines()
    data = [json.loads(line.removeprefix("data:")) for line in lines if line.startswith("data:")]
    assert data[0]["confidence"] < 0.5
    assert data[-1]["refusal"] == "low_confidence"
    assert data[-1]["answer"] == NO_ANSWER


def test_service_strips_invalid_citations_before_saving_history(settings: Settings) -> None:
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="Fusion uses k = 60 [2][9].")]))
    service = RagService(settings, llm=llm)
    service.start()

    result = asyncio.run(service.answer("What value is the RRF constant k set to?", session_id="s"))

    assert result["answer"] == "Fusion uses k = 60 [2]."
    stored = service.sessions.get(service.session_key("default", "s"))
    assert stored[-1].text == "Fusion uses k = 60 [2]."
