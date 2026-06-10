import asyncio
from pathlib import Path

import pytest
import tiktoken
from langchain_core.documents import Document

from rag_service.budget import ContextBudget, TokenCounter
from rag_service.chain import build_rag_chain
from rag_service.config import Settings
from rag_service.providers import ExtractiveFakeChatModel
from rag_service.service import RagService
from tests.helpers import SAMPLE_DOCS
from tests.test_chain import StaticRetriever


def _counter() -> TokenCounter:
    counter = TokenCounter("cl100k_base")
    if not counter.exact:
        pytest.skip("tiktoken cl100k_base is not cached and cannot be downloaded")
    return counter


def _doc(name: str, words: int) -> Document:
    return Document(
        page_content=" ".join(["token"] * words), metadata={"source": name, "chunk_id": name}
    )


def test_counter_matches_tiktoken() -> None:
    counter = _counter()
    text = "Reciprocal rank fusion merges ranked lists."
    assert counter.count(text) == len(tiktoken.get_encoding("cl100k_base").encode(text))


def test_counter_truncates_to_whole_tokens() -> None:
    counter = _counter()
    text = "alpha beta gamma delta epsilon"
    shortened = counter.truncate(text, 2)
    assert counter.count(shortened) == 2
    assert text.startswith(shortened)
    assert counter.truncate(text, 100) == text
    assert counter.truncate(text, 0) == ""


def test_counter_falls_back_to_an_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(name: str) -> None:
        raise OSError("no network")

    monkeypatch.setattr(tiktoken, "get_encoding", unavailable)
    counter = TokenCounter("cl100k_base")

    assert not counter.exact
    assert counter.count("abcdefgh") == 2
    assert counter.count("abcdefghi") == 3
    assert counter.truncate("abcdefghij", 2) == "abcdefgh"


def test_budget_keeps_chunks_in_rank_order_while_they_fit() -> None:
    counter = _counter()
    small, large, tiny = _doc("small", 20), _doc("large", 400), _doc("tiny", 5)
    budget = ContextBudget(counter, budget_for(counter, small, tiny) + 1)

    kept = budget([small, large, tiny])

    assert [d.metadata["source"] for d in kept] == ["small", "tiny"]


def budget_for(counter: TokenCounter, *documents: Document) -> int:
    probe = ContextBudget(counter, 10**9)
    return sum(probe.cost(d) for d in documents)


def test_budget_truncates_a_single_oversized_chunk() -> None:
    counter = _counter()
    budget = ContextBudget(counter, 60)

    kept = budget([_doc("huge", 500), _doc("also-huge", 500)])

    assert len(kept) == 1
    assert kept[0].metadata["truncated"] is True
    assert budget.cost(kept[0]) <= 60 + 1


def test_budget_of_nothing_is_nothing() -> None:
    assert ContextBudget(_counter(), 100)([]) == []


def test_chain_reports_only_the_chunks_in_the_prompt() -> None:
    counter = _counter()
    first = Document(
        page_content="BM25 uses k1 to saturate term frequency.",
        metadata={"source": "bm25.md", "chunk_id": "c1"},
    )
    second = Document(page_content="k1 " * 300, metadata={"source": "noise.md", "chunk_id": "c2"})
    budget = ContextBudget(counter, budget_for(counter, first) + 5)
    chain = build_rag_chain(
        ExtractiveFakeChatModel(), StaticRetriever(documents=[first, second]), budget
    )

    result = asyncio.run(chain.ainvoke({"question": "What does k1 do in BM25?"}))

    assert [s["source"] for s in result["sources"]] == ["bm25.md"]
    assert result["answer"].endswith("[1]")


def test_service_applies_the_configured_budget(tmp_path: Path) -> None:
    _counter()
    settings = Settings(
        _env_file=None,
        index_dir=tmp_path / "index",
        ingest_on_startup=SAMPLE_DOCS,
        max_context_tokens=200,
        chunk_size=600,
        chunk_overlap=80,
    )
    service = RagService(settings)
    service.start()

    result = asyncio.run(service.answer("How does reciprocal rank fusion work?"))

    assert service.budget is not None
    assert 1 <= len(result["sources"]) < 4


def test_budget_can_be_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "")
    settings = Settings(_env_file=None, index_dir=tmp_path)
    assert settings.max_context_tokens is None
    assert RagService(settings).budget is None
