"""LCEL pipeline: retrieve chunks for a question and generate a grounded, cited answer."""

from __future__ import annotations

from collections.abc import Sequence
from operator import itemgetter
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

from rag_service.prompts import NO_ANSWER, answer_prompt, format_context, source_label

SNIPPET_CHARS = 240


class SourceRef(TypedDict):
    """Metadata returned for every chunk placed in the prompt."""

    id: int
    source: str
    label: str
    chunk_id: str
    page: int | None
    start_index: int | None
    score: float | None
    snippet: str


class RagResult(TypedDict):
    question: str
    standalone_question: str
    answer: str
    sources: list[SourceRef]


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
                snippet=text[:SNIPPET_CHARS],
            )
        )
    return refs


def build_answer_chain(llm: BaseChatModel) -> Runnable[dict[str, Any], str]:
    """Generate an answer from ``question``, ``documents`` and optional ``chat_history``.

    Without documents the model is not called and :data:`NO_ANSWER` is returned. The
    chain streams token by token when the model does.
    """
    generate = (
        RunnablePassthrough.assign(context=lambda x: format_context(x["documents"]))
        | answer_prompt()
        | llm
        | StrOutputParser()
    )
    return RunnableBranch(
        (lambda x: not x["documents"], RunnableLambda(lambda _: NO_ANSWER)),
        generate,
    ).with_config(run_name="answer")


def build_retrieval_chain(retriever: BaseRetriever) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Add ``standalone_question`` and the retrieved ``documents`` to the input."""
    return (
        RunnablePassthrough.assign(standalone_question=itemgetter("question"))
        .assign(documents=itemgetter("standalone_question") | retriever)
        .with_config(run_name="retrieve")
    )


def _to_result(state: dict[str, Any]) -> RagResult:
    return RagResult(
        question=state["question"],
        standalone_question=state["standalone_question"],
        answer=state["answer"],
        sources=source_refs(state["documents"]),
    )


def build_rag_chain(
    llm: BaseChatModel, retriever: BaseRetriever
) -> Runnable[dict[str, Any], RagResult]:
    """Full pipeline: ``{"question": ...}`` in, :class:`RagResult` out."""
    return (
        build_retrieval_chain(retriever)
        | RunnablePassthrough.assign(answer=build_answer_chain(llm))
        | RunnableLambda(_to_result)
    ).with_config(run_name="rag")
