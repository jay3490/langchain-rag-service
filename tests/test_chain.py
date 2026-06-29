import asyncio

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever

from rag_service.chain import build_answer_chain, build_rag_chain, source_refs
from rag_service.prompts import NO_ANSWER, answer_prompt, format_context
from rag_service.providers import ExtractiveFakeChatModel

DOCS = [
    Document(
        page_content="BM25 uses k1 to saturate term frequency. It needs no training.",
        metadata={"source": "bm25.md", "chunk_id": "c1", "start_index": 0, "rrf_score": 0.03},
    ),
    Document(
        page_content="FAISS stores vectors for similarity search.",
        metadata={"source": "manual.pdf", "chunk_id": "c2", "page": 4, "bm25_score": 1.5},
    ),
]


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.documents


def test_format_context_numbers_sources_and_labels_pages() -> None:
    context = format_context(DOCS)
    assert '<source id="1" name="bm25.md">' in context
    assert '<source id="2" name="manual.pdf (page 4)">' in context


def test_answer_prompt_includes_history_and_grounding_rules() -> None:
    messages = answer_prompt().format_messages(
        context="ctx",
        question="q?",
        chat_history=[HumanMessage(content="earlier"), AIMessage(content="reply")],
    )
    assert [m.type for m in messages] == ["system", "human", "ai", "human"]
    assert NO_ANSWER in str(messages[0].content)
    assert "<question>q?</question>" in str(messages[-1].content)


def test_rag_chain_output_shape() -> None:
    chain = build_rag_chain(ExtractiveFakeChatModel(), StaticRetriever(documents=DOCS))

    result = chain.invoke({"question": "What does k1 do in BM25?"})

    assert set(result) == {
        "question",
        "standalone_question",
        "answer",
        "sources",
        "confidence",
        "citations",
        "invalid_citations",
        "refusal",
        "cached",
    }
    assert result["citations"] == [1]
    assert result["refusal"] is None
    assert result["standalone_question"] == "What does k1 do in BM25?"
    assert result["answer"].endswith("[1]")
    assert [s["id"] for s in result["sources"]] == [1, 2]
    first, second = result["sources"]
    assert first["source"] == "bm25.md"
    assert first["score"] == pytest.approx(0.03)
    assert second["label"] == "manual.pdf (page 4)"
    assert second["page"] == 4
    assert second["score"] == pytest.approx(1.5)


def test_rag_chain_without_documents_skips_the_model() -> None:
    llm = GenericFakeChatModel(messages=iter([]))  # raises if it is ever called
    result = build_rag_chain(llm, StaticRetriever(documents=[])).invoke({"question": "x"})
    assert result["answer"] == NO_ANSWER
    assert result["sources"] == []


def test_rag_chain_passes_context_to_any_chat_model() -> None:
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="Saturation [1].")]))
    result = build_rag_chain(llm, StaticRetriever(documents=DOCS)).invoke({"question": "k1?"})
    assert result["answer"] == "Saturation [1]."


def test_answer_chain_streams_tokens() -> None:
    chain = build_answer_chain(ExtractiveFakeChatModel())
    inputs = {"question": "What does k1 do?", "documents": DOCS}

    async def collect() -> list[str]:
        return [chunk async for chunk in chain.astream(inputs)]

    chunks = asyncio.run(collect())
    assert len(chunks) > 1
    assert "".join(chunks) == chain.invoke(inputs)


def test_source_refs_truncate_snippets() -> None:
    long_doc = Document(page_content="word " * 200, metadata={"source": "a.txt"})
    (ref,) = source_refs([long_doc])
    assert len(ref["snippet"]) == 240
    assert ref["score"] is None
    assert ref["page"] is None
