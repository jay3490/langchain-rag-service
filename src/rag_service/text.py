"""Tokenisation shared by the BM25 index and the hashing embeddings."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# fmt: off
STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from",
    "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "so",
    "such", "than", "that", "the", "their", "them", "then", "there", "these", "this", "those",
    "to", "was", "what", "when", "where", "which", "while", "who", "why", "will", "with",
    "you", "your",
})
# fmt: on


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and return its alphanumeric tokens without stop words.

    Underscored identifiers such as ``top_k`` stay a single token.
    """
    return [token for token in _TOKEN.findall(text.lower()) if token not in STOP_WORDS]


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences on terminal punctuation, collapsing whitespace."""
    flat = " ".join(text.split())
    return [sentence for sentence in _SENTENCE_END.split(flat) if sentence]
