"""Second-stage re-ranking of retrieved candidates.

The retriever over-fetches ``rerank_candidates`` chunks, a :class:`Reranker` scores each
one against the query, and the best ``k`` go into the prompt. Every reranker returns
scores in ``[0, 1]``, stored on the chunk as ``rerank_score``.

* :class:`LexicalReranker` needs nothing: weighted query-term and bigram coverage.
* :class:`LLMReranker` asks the configured chat model for a 0-10 grade per passage and
  falls back to the lexical score for a reply it cannot parse.
* :class:`CrossEncoderReranker` uses a sentence-transformers cross-encoder; it is only
  available with the optional ``rerank`` extra installed.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
    Callbacks,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from pydantic import ConfigDict, Field

from rag_service.config import Settings
from rag_service.prompts import rerank_prompt
from rag_service.text import tokenize

RERANK_KEY = "rerank_score"
_GRADE = re.compile(r"\b(10|[0-9])\b")


class Reranker(ABC):
    """Scores candidate documents for a query; higher is more relevant, range 0-1."""

    name: str

    @abstractmethod
    def score(
        self, query: str, documents: Sequence[Document], callbacks: Callbacks = None
    ) -> list[float]:
        """Score ``documents``; ``callbacks`` reach any model calls the reranker makes."""

    async def ascore(
        self, query: str, documents: Sequence[Document], callbacks: Callbacks = None
    ) -> list[float]:
        return await asyncio.to_thread(self.score, query, documents, callbacks)

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int,
        callbacks: Callbacks = None,
    ) -> list[Document]:
        return apply_scores(documents, self.score(query, documents, callbacks), top_n)

    async def arerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int,
        callbacks: Callbacks = None,
    ) -> list[Document]:
        return apply_scores(documents, await self.ascore(query, documents, callbacks), top_n)


def apply_scores(
    documents: Sequence[Document], scores: Sequence[float], top_n: int
) -> list[Document]:
    """Order by score (ties keep retrieval order) and keep ``top_n`` scored copies."""
    order = sorted(range(len(documents)), key=lambda i: (-scores[i], i))[:top_n]
    return [
        Document(
            page_content=documents[i].page_content,
            metadata={**documents[i].metadata, RERANK_KEY: round(float(scores[i]), 6)},
        )
        for i in order
    ]


def _bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return set(itertools.pairwise(tokens))


class LexicalReranker(Reranker):
    """Query-term coverage weighted by rarity among the candidates, plus phrase overlap.

    ``score = (1 - phrase_weight) * idf-weighted share of query terms present
    + phrase_weight * share of query bigrams present``. Terms that occur in every
    candidate carry little weight, so the score rewards the distinctive words.
    """

    name = "lexical"

    def __init__(self, phrase_weight: float = 0.2) -> None:
        self.phrase_weight = phrase_weight

    def score(
        self, query: str, documents: Sequence[Document], callbacks: Callbacks = None
    ) -> list[float]:
        query_tokens = tokenize(query)
        terms = set(query_tokens)
        if not terms or not documents:
            return [0.0] * len(documents)
        doc_tokens = [tokenize(d.page_content) for d in documents]
        doc_terms = [set(tokens) for tokens in doc_tokens]
        count = len(documents)
        idf = {
            term: math.log(1 + (count + 1) / (1 + sum(term in t for t in doc_terms)))
            for term in terms
        }
        total = sum(idf.values())
        query_bigrams = _bigrams(query_tokens)
        scores: list[float] = []
        for tokens, present in zip(doc_tokens, doc_terms, strict=True):
            coverage = sum(idf[term] for term in terms if term in present) / total
            phrases = (
                len(query_bigrams & _bigrams(tokens)) / len(query_bigrams)
                if query_bigrams
                else coverage
            )
            scores.append((1 - self.phrase_weight) * coverage + self.phrase_weight * phrases)
        return scores


def parse_grade(reply: str) -> float | None:
    """Read the first 0-10 integer in a model reply as a score in ``[0, 1]``."""
    match = _GRADE.search(reply)
    return int(match.group(1)) / 10 if match else None


class LLMReranker(Reranker):
    """Grades each passage with the chat model, ``max_concurrency`` calls at a time."""

    name = "llm"

    def __init__(
        self, llm: BaseChatModel, *, max_concurrency: int = 4, passage_chars: int = 2000
    ) -> None:
        # No .with_config() binding here: a binding merges its callbacks with the
        # enclosing run's, which would attach these model calls to the wrong parent.
        self.chain = rerank_prompt() | llm | StrOutputParser()
        self.max_concurrency = max_concurrency
        self.passage_chars = passage_chars
        self.fallback = LexicalReranker()

    def _config(self, callbacks: Callbacks) -> RunnableConfig:
        return {
            "run_name": "rerank",
            "max_concurrency": self.max_concurrency,
            "callbacks": callbacks,
        }

    def _inputs(self, query: str, documents: Sequence[Document]) -> list[dict[str, Any]]:
        return [
            {"query": query, "passage": d.page_content[: self.passage_chars]} for d in documents
        ]

    def _combine(
        self, query: str, documents: Sequence[Document], replies: Sequence[str]
    ) -> list[float]:
        grades = [parse_grade(reply) for reply in replies]
        if all(grade is not None for grade in grades):
            return [float(grade) for grade in grades]  # type: ignore[arg-type]
        lexical = self.fallback.score(query, documents)
        return [grade if grade is not None else lexical[i] for i, grade in enumerate(grades)]

    def score(
        self, query: str, documents: Sequence[Document], callbacks: Callbacks = None
    ) -> list[float]:
        replies = self.chain.batch(
            self._inputs(query, documents),
            config=self._config(callbacks),
        )
        return self._combine(query, documents, replies)

    async def ascore(
        self, query: str, documents: Sequence[Document], callbacks: Callbacks = None
    ) -> list[float]:
        replies = await self.chain.abatch(
            self._inputs(query, documents),
            config=self._config(callbacks),
        )
        return self._combine(query, documents, replies)


class CrossEncoderReranker(Reranker):
    """A sentence-transformers cross-encoder; logits are squashed with a sigmoid."""

    name = "cross-encoder"

    def __init__(self, model_name: str, *, batch_size: int = 16) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "RAG_RERANKER=cross-encoder needs sentence-transformers; "
                'install it with pip install "langchain-rag-service[rerank]"'
            ) from exc
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def score(
        self, query: str, documents: Sequence[Document], callbacks: Callbacks = None
    ) -> list[float]:
        if not documents:
            return []
        pairs = [(query, d.page_content) for d in documents]
        logits = self.model.predict(pairs, batch_size=self.batch_size)
        return [1 / (1 + math.exp(-float(logit))) for logit in logits]


def build_reranker(settings: Settings, llm: BaseChatModel) -> Reranker | None:
    """Create the reranker selected by ``settings.reranker`` (``None`` when off)."""
    if settings.reranker == "none":
        return None
    if settings.reranker == "lexical":
        return LexicalReranker()
    if settings.reranker == "llm":
        return LLMReranker(llm, max_concurrency=settings.rerank_max_concurrency)
    return CrossEncoderReranker(settings.cross_encoder_model)


class RerankingRetriever(BaseRetriever):
    """Retrieve candidates with ``base`` and keep the ``k`` the reranker scores highest."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base: BaseRetriever
    reranker: Reranker
    k: int = Field(default=4, ge=1)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        callbacks = run_manager.get_child()
        candidates = self.base.invoke(query, config={"callbacks": callbacks})
        return self.reranker.rerank(query, candidates, self.k, callbacks)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        callbacks = run_manager.get_child()
        candidates = await self.base.ainvoke(query, config={"callbacks": callbacks})
        return await self.reranker.arerank(query, candidates, self.k, callbacks)
