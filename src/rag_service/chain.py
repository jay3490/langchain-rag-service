"""LCEL pipeline: retrieve chunks for a question and generate a grounded, cited answer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from operator import itemgetter
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

from rag_service.guardrails import Guardrails, Refusal, retrieval_confidence
from rag_service.prompts import (
    NO_ANSWER,
    answer_prompt,
    format_context,
    format_conversation,
    rewrite_prompt,
    source_label,
)

SNIPPET_CHARS = 240

#: Trims a ranked list of retrieved chunks, e.g. to a token budget.
DocumentFilter = Callable[[list[Document]], list[Document]]


class SourceRef(TypedDict):
    """Metadata returned for every chunk placed in the prompt."""

    id: int
    source: str
    label: str
    chunk_id: str
    page: int | None
    start_index: int | None
    score: float | None
    rerank_score: float | None
    snippet: str


class RagResult(TypedDict):
    question: str
    standalone_question: str
    answer: str
    sources: list[SourceRef]
    confidence: float
    citations: list[int]
    invalid_citations: list[int]
    refusal: Refusal | None


def source_refs(documents: Sequence[Document]) -> list[SourceRef]:
    """Describe retrieved documents; ``id`` matches the citation numbers in the answer."""
    refs: list[SourceRef] = []
    for number, document in enumerate(documents, start=1):
        metadata = document.metadata
        score = metadata.get("rrf_score", metadata.get("bm25_score"))
        text = " ".join(document.page_content.split())
        refs.append(
            SourceRef(
                id=number,
                source=str(metadata.get("source", "unknown")),
                label=source_label(document),
                chunk_id=str(metadata.get("chunk_id", "")),
                page=metadata.get("page"),
                start_index=metadata.get("start_index"),
                score=float(score) if score is not None else None,
                rerank_score=metadata.get("rerank_score"),
                snippet=text[:SNIPPET_CHARS],
            )
        )
    return refs


def build_answer_chain(
    llm: BaseChatModel, guardrails: Guardrails | None = None
) -> Runnable[dict[str, Any], str]:
    """Generate an answer from ``question``, ``documents`` and optional ``chat_history``.

    Without documents, or when ``confidence`` is below the guardrail's floor, the model
    is not called and :data:`NO_ANSWER` is returned. The chain streams token by token
    when the model does.
    """
    guard = guardrails or Guardrails()
    generate = (
        RunnablePassthrough.assign(context=lambda x: format_context(x["documents"]))
        | answer_prompt()
        | llm
        | StrOutputParser()
    )
    return RunnableBranch(
        (
            lambda x: not x["documents"] or guard.should_decline(x.get("confidence", 1.0)),
            RunnableLambda(lambda _: NO_ANSWER),
        ),
        generate,
    ).with_config(run_name="answer")


def build_rewrite_chain(llm: BaseChatModel) -> Runnable[dict[str, Any], str]:
    """Turn ``question`` into a standalone query using ``chat_history``.

    The first turn of a conversation has no history, so the question is used as-is and
    the model is not called.
    """
    rewrite = (
        RunnablePassthrough.assign(conversation=lambda x: format_conversation(x["chat_history"]))
        | rewrite_prompt()
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda text: " ".join(text.split()))
    )
    return RunnableBranch(
        (lambda x: not x.get("chat_history"), itemgetter("question")),
        rewrite,
    ).with_config(run_name="rewrite_question")


def build_retrieval_chain(
    retriever: BaseRetriever, llm: BaseChatModel, budget: DocumentFilter | None = None
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Add ``standalone_question`` and the retrieved ``documents`` to the input.

    The retriever searches with the standalone question, so follow-ups such as
    "what value is typical for it?" retrieve the chunks the conversation is about.
    ``budget``, when given, trims the retrieved chunks before they reach the prompt;
    ``confidence`` is then computed from the chunks that remain.
    """
    retrieve: Runnable[Any, list[Document]] = itemgetter("standalone_question") | retriever
    if budget is not None:
        retrieve = retrieve | RunnableLambda(budget, name="context_budget")
    return (
        RunnablePassthrough.assign(standalone_question=build_rewrite_chain(llm))
        .assign(documents=retrieve)
        .assign(confidence=lambda x: retrieval_confidence(x["standalone_question"], x["documents"]))
        .with_config(run_name="retrieve")
    )


def to_result(state: dict[str, Any], guardrails: Guardrails) -> RagResult:
    """Check the generated answer against the guardrails and describe its sources."""
    reviewed = guardrails.review(state["answer"], state["documents"], state["confidence"])
    return RagResult(
        question=state["question"],
        standalone_question=state["standalone_question"],
        answer=reviewed.answer,
        sources=source_refs(state["documents"]),
        confidence=round(float(state["confidence"]), 6),
        citations=reviewed.citations,
        invalid_citations=reviewed.invalid_citations,
        refusal=reviewed.refusal,
    )


def build_rag_chain(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    budget: DocumentFilter | None = None,
    guardrails: Guardrails | None = None,
) -> Runnable[dict[str, Any], RagResult]:
    """Full pipeline: ``{"question", "chat_history"?}`` in, :class:`RagResult` out."""
    guard = guardrails or Guardrails()
    return (
        build_retrieval_chain(retriever, llm, budget)
        | RunnablePassthrough.assign(answer=build_answer_chain(llm, guard))
        | RunnableLambda(lambda state: to_result(state, guard), name="guardrails")
    ).with_config(run_name="rag")
