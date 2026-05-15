"""Dense, lexical and hybrid retrievers over an :class:`~rag_service.store.IndexSnapshot`."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, model_validator

from rag_service.config import RetrieverMode, SearchType, Settings
from rag_service.store import IndexSnapshot
from rag_service.text import tokenize

ID_KEY = "chunk_id"


def _with_metadata(document: Document, **extra: Any) -> Document:
    """Return a copy of ``document`` with extra metadata; indexed chunks stay untouched."""
    return Document(page_content=document.page_content, metadata={**document.metadata, **extra})


class BM25Retriever(BaseRetriever):
    """Lexical retrieval over the BM25 index held by a snapshot."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    snapshot: IndexSnapshot
    k: int = Field(default=4, ge=1)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        bm25 = self.snapshot.bm25
        tokens = tokenize(query)
        if bm25 is None or not tokens:
            return []
        scores = bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [
            _with_metadata(self.snapshot.chunks[i], bm25_score=round(float(scores[i]), 6))
            for i in ranked[: self.k]
            if scores[i] > 0
        ]


class EmptyRetriever(BaseRetriever):
    """Returned when nothing has been indexed yet."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return []


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[Document]],
    weights: Sequence[float] | None = None,
    *,
    k: int = 60,
    id_key: str = ID_KEY,
) -> list[tuple[Document, float]]:
    """Fuse ranked lists with weighted reciprocal rank fusion.

    Each document scores ``sum(weight / (k + rank))`` over the lists it appears in, with
    ``rank`` starting at 1. Documents are identified by ``metadata[id_key]``, falling back
    to their text. Ties keep the order in which documents were first seen.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights must match the number of result lists")

    scores: dict[str, float] = {}
    first_seen: dict[str, Document] = {}
    for documents, weight in zip(result_lists, weights, strict=True):
        for rank, document in enumerate(documents, start=1):
            key = str(document.metadata.get(id_key, document.page_content))
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            first_seen.setdefault(key, document)
    order = sorted(scores, key=lambda key: -scores[key])  # stable: ties keep first-seen order
    return [(first_seen[key], scores[key]) for key in order]


class HybridRetriever(BaseRetriever):
    """Runs several retrievers and merges their results with reciprocal rank fusion."""

    retrievers: list[BaseRetriever]
    weights: list[float]
    k: int = Field(default=4, ge=1)
    rrf_k: int = Field(default=60, ge=1)

    @model_validator(mode="after")
    def _check_weights(self) -> HybridRetriever:
        if len(self.weights) != len(self.retrievers):
            raise ValueError("weights must match the number of retrievers")
        return self

    def _fuse(self, result_lists: Sequence[Sequence[Document]]) -> list[Document]:
        fused = reciprocal_rank_fusion(result_lists, self.weights, k=self.rrf_k)
        return [
            _with_metadata(document, rrf_score=round(score, 6))
            for document, score in fused[: self.k]
        ]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        config = {"callbacks": run_manager.get_child()}
        return self._fuse([r.invoke(query, config=config) for r in self.retrievers])

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        config = {"callbacks": run_manager.get_child()}
        results = await asyncio.gather(*(r.ainvoke(query, config=config) for r in self.retrievers))
        return self._fuse(results)


@dataclass(frozen=True)
class RetrievalOptions:
    """How to retrieve for one request."""

    mode: RetrieverMode = "hybrid"
    search_type: SearchType = "similarity"
    k: int = 4
    fetch_k: int = 20
    mmr_lambda: float = 0.5
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0

    @classmethod
    def from_settings(cls, settings: Settings) -> RetrievalOptions:
        return cls(
            mode=settings.retriever_mode,
            search_type=settings.search_type,
            k=settings.top_k,
            fetch_k=settings.fetch_k,
            mmr_lambda=settings.mmr_lambda,
            rrf_k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            bm25_weight=settings.bm25_weight,
        )

    def with_overrides(self, **changes: Any) -> RetrievalOptions:
        """Copy with the non-``None`` values in ``changes`` applied."""
        return replace(self, **{key: value for key, value in changes.items() if value is not None})


def _dense(vectorstore: FAISS, options: RetrievalOptions, k: int) -> BaseRetriever:
    search_kwargs: dict[str, Any] = {"k": k}
    if options.search_type == "mmr":
        search_kwargs |= {"fetch_k": max(options.fetch_k, k), "lambda_mult": options.mmr_lambda}
    return vectorstore.as_retriever(search_type=options.search_type, search_kwargs=search_kwargs)


def build_retriever(snapshot: IndexSnapshot, options: RetrievalOptions) -> BaseRetriever:
    """Create the retriever described by ``options`` for ``snapshot``.

    In hybrid mode each side contributes up to ``2 * k`` candidates (capped at
    ``fetch_k``) before fusion trims the result to ``k``.
    """
    vectorstore = snapshot.vectorstore
    if snapshot.is_empty or vectorstore is None:
        return EmptyRetriever()
    if options.mode == "dense":
        return _dense(vectorstore, options, options.k)
    if options.mode == "bm25":
        return BM25Retriever(snapshot=snapshot, k=options.k)
    depth = max(options.k, min(options.fetch_k, 2 * options.k))
    return HybridRetriever(
        retrievers=[_dense(vectorstore, options, depth), BM25Retriever(snapshot=snapshot, k=depth)],
        weights=[options.dense_weight, options.bm25_weight],
        k=options.k,
        rrf_k=options.rrf_k,
    )
