"""Embedding and chat model providers.

``fake`` needs no network or credentials and is fully deterministic, which makes it
suitable for tests, CI and offline demos. ``openai`` talks to any OpenAI-compatible
endpoint, including a local Ollama server.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterator
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import OpenAIEmbeddings
from pydantic import Field

from rag_service.config import Settings
from rag_service.prompts import NO_ANSWER
from rag_service.text import split_sentences, tokenize

_SOURCE_BLOCK = re.compile(r'<source id="(\d+)"[^>]*>(.*?)</source>', re.DOTALL)
_QUESTION_BLOCK = re.compile(r"<question>(.*?)</question>", re.DOTALL)
_CONVERSATION_BLOCK = re.compile(r"<conversation>(.*?)</conversation>", re.DOTALL)
_QUERY_BLOCK = re.compile(r"<query>(.*?)</query>", re.DOTALL)
_PASSAGE_BLOCK = re.compile(r"<passage>(.*?)</passage>", re.DOTALL)
_STREAM_PIECE = re.compile(r"\S+\s*")


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-features embeddings using the hashing trick.

    Each text is mapped to word tokens plus character trigrams of those words; every
    feature is hashed to a signed bucket of a fixed-size vector, which is then
    L2-normalised. Texts that share vocabulary end up close together, so retrieval
    behaves sensibly without a model. It is a stand-in, not a semantic model: it has no
    notion of synonyms.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, (1.0 if (value >> 63) & 1 else -1.0)

    def _features(self, text: str) -> Counter[str]:
        features: Counter[str] = Counter()
        for token in tokenize(text):
            features[f"w:{token}"] += 2
            padded = f"#{token}#"
            for start in range(len(padded) - 2):
                features[f"c:{padded[start : start + 3]}"] += 1
        return features

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature, count in self._features(text).items():
            index, sign = self._bucket(feature)
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ExtractiveFakeChatModel(BaseChatModel):
    """A deterministic chat model that answers by extraction instead of generation.

    It reads the last human message and understands the tagged prompts defined in
    :mod:`rag_service.prompts`:

    * a prompt with ``<source id="N">`` blocks and a ``<question>`` is answered with the
      source sentence that shares the most tokens with the question, cited as ``[N]``;
      with no overlapping sentence it declines with :data:`NO_ANSWER`;
    * a prompt with a ``<conversation>`` block and a ``<question>`` (query rewriting)
      is answered with the previous user question joined to the follow-up question;
    * a prompt with a ``<query>`` and a ``<passage>`` (re-ranking) is answered with a
      0-10 grade: the share of query words that occur in the passage.
    """

    max_sentences: int = Field(default=2, ge=1)

    @property
    def _llm_type(self) -> str:
        return "extractive-fake"

    def _respond(self, messages: list[BaseMessage]) -> str:
        human = [message for message in messages if isinstance(message, HumanMessage)]
        prompt = str(human[-1].content) if human else ""
        query, passage = _QUERY_BLOCK.search(prompt), _PASSAGE_BLOCK.search(prompt)
        if query is not None and passage is not None:
            return self._grade(query.group(1), passage.group(1))

        question_match = _QUESTION_BLOCK.search(prompt)
        question = question_match.group(1).strip() if question_match else prompt.strip()

        conversation = _CONVERSATION_BLOCK.search(prompt)
        if conversation is not None:
            return self._rewrite(conversation.group(1), question)

        sources = _SOURCE_BLOCK.findall(prompt)
        return self._extract(sources, question)

    @staticmethod
    def _grade(query: str, passage: str) -> str:
        wanted = set(tokenize(query))
        if not wanted:
            return "0"
        found = len(wanted & set(tokenize(passage)))
        return str(round(10 * found / len(wanted)))

    @staticmethod
    def _rewrite(conversation: str, question: str) -> str:
        user_turns = [
            line.removeprefix("user:").strip()
            for line in conversation.splitlines()
            if line.startswith("user:")
        ]
        if not user_turns:
            return question
        return f"{user_turns[-1]} {question}"

    def _extract(self, sources: list[tuple[str, str]], question: str) -> str:
        wanted = set(tokenize(question))
        scored: list[tuple[int, int, int, str, str]] = []
        for position, (source_id, body) in enumerate(sources):
            for order, sentence in enumerate(split_sentences(body)):
                overlap = len(wanted & set(tokenize(sentence)))
                if overlap:
                    scored.append((-overlap, position, order, source_id, sentence))
        if not scored:
            return NO_ANSWER
        scored.sort()
        best = sorted(scored[: self.max_sentences], key=lambda item: (item[1], item[2]))
        return " ".join(f"{sentence} [{source_id}]" for *_, source_id, sentence in best)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=self._respond(messages))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for piece in _STREAM_PIECE.findall(self._respond(messages)):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager is not None:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


def _uses_openai_tokenizer(base_url: str | None) -> bool:
    """Only the official API accepts the token-id inputs OpenAIEmbeddings sends by default."""
    return base_url is None or "api.openai.com" in base_url


def build_embeddings(settings: Settings) -> Embeddings:
    """Create the embedding model selected by ``settings.provider``."""
    if settings.provider == "fake":
        return HashingEmbeddings(dim=settings.fake_embedding_dim)
    kwargs: dict[str, Any] = {
        "model": settings.embedding_model,
        "check_embedding_ctx_length": _uses_openai_tokenizer(settings.openai_base_url),
    }
    kwargs.update(_openai_connection(settings))
    return OpenAIEmbeddings(**kwargs)


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Create the chat model selected by ``settings.provider``."""
    if settings.provider == "fake":
        return ExtractiveFakeChatModel()
    return init_chat_model(
        settings.chat_model,
        model_provider="openai",
        temperature=settings.temperature,
        **_openai_connection(settings),
    )


def _openai_connection(settings: Settings) -> dict[str, Any]:
    connection: dict[str, Any] = {}
    if settings.openai_base_url:
        connection["base_url"] = settings.openai_base_url
    if settings.openai_api_key is not None:
        connection["api_key"] = settings.openai_api_key
    elif settings.openai_base_url and not _uses_openai_tokenizer(settings.openai_base_url):
        # Local OpenAI-compatible servers such as Ollama ignore the key but the client
        # refuses to start without one.
        connection["api_key"] = "unused"
    return connection
