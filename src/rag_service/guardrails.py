"""Answer guardrails: a retrieval-confidence floor and citation validation.

* **Confidence.** Before the model is called, the retrieved chunks get a confidence in
  ``[0, 1]``: the highest ``rerank_score`` when a reranker ran, otherwise the largest
  share of the question's distinct words that any single chunk contains. Below
  ``min_confidence`` the service declines with :data:`~rag_service.prompts.NO_ANSWER`
  without calling the model.
* **Citations.** Every ``[n]`` (or ``[n, m]``) in the answer must name one of the
  numbered sources that were in the prompt. With the ``strip`` policy invalid markers
  are removed; with ``refuse`` an answer citing a missing source, or citing nothing,
  is replaced by the decline message. ``off`` only reports them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.documents import Document

from rag_service.prompts import NO_ANSWER
from rag_service.text import tokenize

CitationPolicy = Literal["off", "strip", "refuse"]
Refusal = Literal["no_documents", "low_confidence", "invalid_citations", "not_in_sources"]

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def retrieval_confidence(question: str, documents: Sequence[Document]) -> float:
    """Confidence that ``documents`` can answer ``question`` (see the module docstring)."""
    if not documents:
        return 0.0
    reranked = [d.metadata["rerank_score"] for d in documents if "rerank_score" in d.metadata]
    if reranked:
        return float(max(reranked))
    wanted = set(tokenize(question))
    if not wanted:
        return 0.0
    return max(len(wanted & set(tokenize(d.page_content))) / len(wanted) for d in documents)


@dataclass(frozen=True)
class CitationCheck:
    answer: str
    cited: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)


def check_citations(answer: str, source_count: int) -> CitationCheck:
    """Find citation numbers; ``answer`` comes back with invalid numbers removed."""
    cited: list[int] = []
    invalid: list[int] = []

    def keep_valid(match: re.Match[str]) -> str:
        numbers = [int(n) for n in re.split(r"\s*,\s*", match.group(1))]
        valid = [n for n in numbers if 1 <= n <= source_count]
        for n in numbers:
            target = cited if n in valid else invalid
            if n not in target:
                target.append(n)
        return f"[{', '.join(map(str, valid))}]" if valid else ""

    cleaned = _CITATION.sub(keep_valid, answer)
    if invalid:
        cleaned = _MULTI_SPACE.sub(" ", _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)).strip()
    return CitationCheck(answer=cleaned, cited=sorted(cited), invalid=sorted(invalid))


@dataclass(frozen=True)
class GuardedAnswer:
    answer: str
    citations: list[int]
    invalid_citations: list[int]
    refusal: Refusal | None


@dataclass(frozen=True)
class Guardrails:
    min_confidence: float = 0.0
    citation_policy: CitationPolicy = "strip"

    def should_decline(self, confidence: float) -> bool:
        return confidence < self.min_confidence

    def review(
        self, answer: str, documents: Sequence[Document], confidence: float
    ) -> GuardedAnswer:
        """Validate a generated answer against the sources that were in the prompt."""
        if not documents:
            return GuardedAnswer(NO_ANSWER, [], [], "no_documents")
        if self.should_decline(confidence):
            return GuardedAnswer(NO_ANSWER, [], [], "low_confidence")
        if answer.strip() == NO_ANSWER:
            return GuardedAnswer(NO_ANSWER, [], [], "not_in_sources")
        check = check_citations(answer, len(documents))
        if self.citation_policy == "refuse" and (check.invalid or not check.cited):
            return GuardedAnswer(NO_ANSWER, [], check.invalid, "invalid_citations")
        final = check.answer if self.citation_policy == "strip" else answer
        return GuardedAnswer(final, check.cited, check.invalid, None)
