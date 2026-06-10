"""Token counting with tiktoken and a context budget for the answer prompt.

Retrieved chunks are added to the prompt in rank order while they fit within
``max_context_tokens``; a chunk that does not fit is skipped and smaller, lower-ranked
chunks may still be added. If not even the best chunk fits, it is cut to the budget.
The chunks returned are exactly the ones the model sees, so citation numbers and the
reported sources stay aligned.

tiktoken downloads its encoding files on first use. Without network access and without
a cached copy (``TIKTOKEN_CACHE_DIR``), counting falls back to an estimate of one token
per four characters and logs a warning.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

import tiktoken
from langchain_core.documents import Document

from rag_service.prompts import format_context

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4


class Encoding(Protocol):
    def encode(self, text: str, *, disallowed_special: tuple[()] = ...) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


class TokenCounter:
    """Counts tokens with a tiktoken encoding, or estimates them if it is unavailable."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self._encoding: Encoding | None
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            logger.warning(
                "tiktoken encoding %s unavailable (%s); estimating tokens from characters",
                encoding_name,
                exc,
            )
            self._encoding = None

    @property
    def exact(self) -> bool:
        return self._encoding is not None

    def count(self, text: str) -> int:
        if self._encoding is None:
            return math.ceil(len(text) / CHARS_PER_TOKEN)
        return len(self._encoding.encode(text, disallowed_special=()))

    def truncate(self, text: str, max_tokens: int) -> str:
        """Keep the first ``max_tokens`` tokens of ``text``."""
        if max_tokens <= 0:
            return ""
        if self._encoding is None:
            return text[: max_tokens * CHARS_PER_TOKEN]
        tokens = self._encoding.encode(text, disallowed_special=())
        return text if len(tokens) <= max_tokens else self._encoding.decode(tokens[:max_tokens])


@lru_cache(maxsize=4)
def get_token_counter(encoding_name: str = "cl100k_base") -> TokenCounter:
    return TokenCounter(encoding_name)


class ContextBudget:
    """Callable that trims a ranked list of chunks to ``max_tokens`` of rendered context."""

    def __init__(self, counter: TokenCounter, max_tokens: int) -> None:
        self.counter = counter
        self.max_tokens = max_tokens

    def cost(self, document: Document) -> int:
        """Tokens of the chunk as rendered into the prompt, ``<source>`` tags included."""
        return self.counter.count(format_context([document])) + 1  # +1 for the separator

    def __call__(self, documents: Sequence[Document]) -> list[Document]:
        kept: list[Document] = []
        used = 0
        for document in documents:
            cost = self.cost(document)
            if used + cost <= self.max_tokens:
                kept.append(document)
                used += cost
        if documents and not kept:
            kept.append(self._shrink(documents[0]))
        return kept

    def _shrink(self, document: Document) -> Document:
        overhead = self.cost(Document(page_content="", metadata=document.metadata))
        text = self.counter.truncate(document.page_content, self.max_tokens - overhead)
        return Document(page_content=text, metadata={**document.metadata, "truncated": True})
