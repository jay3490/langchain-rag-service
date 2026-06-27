"""Key-value caches (Redis or in-process) and an embedding cache built on them.

:func:`build_cache` connects to ``RAG_REDIS_URL`` when it is set and reachable, so
replicas share entries; otherwise it falls back to a bounded in-process LRU cache and
logs why. Cache failures never fail a request: a Redis error is logged and treated as
a miss.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from array import array
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import redis
from langchain_core.embeddings import Embeddings

from rag_service.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Lookup counters for one use of a cache (for example ``embedding``)."""

    name: str
    hits: int = 0
    misses: int = 0

    def record(self, hits: int, misses: int) -> None:
        self.hits += hits
        self.misses += misses

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class Cache(ABC):
    """Bytes in, bytes out, with an optional time-to-live in seconds."""

    backend: str

    @abstractmethod
    def get_many(self, keys: Sequence[str]) -> list[bytes | None]: ...

    @abstractmethod
    def set_many(self, items: dict[str, bytes], ttl: float | None = None) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    def get(self, key: str) -> bytes | None:
        return self.get_many([key])[0]

    def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        self.set_many({key: value}, ttl)

    @abstractmethod
    def close(self) -> None:
        """Release the cache's resources."""


class MemoryCache(Cache):
    """Thread-safe LRU cache with per-entry expiry, local to this process."""

    backend = "memory"

    def __init__(self, max_entries: int = 10_000, clock: Any = time.monotonic) -> None:
        self.max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, tuple[bytes, float | None]] = OrderedDict()
        self._lock = threading.Lock()

    def get_many(self, keys: Sequence[str]) -> list[bytes | None]:
        now = self._clock()
        values: list[bytes | None] = []
        with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is None:
                    values.append(None)
                    continue
                value, expires = entry
                if expires is not None and expires <= now:
                    del self._entries[key]
                    values.append(None)
                    continue
                self._entries.move_to_end(key)
                values.append(value)
        return values

    def set_many(self, items: dict[str, bytes], ttl: float | None = None) -> None:
        expires = self._clock() + ttl if ttl else None
        with self._lock:
            for key, value in items.items():
                self._entries[key] = (value, expires)
                self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def close(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class RedisCache(Cache):
    """Entries in Redis, shared by every replica that points at the same server."""

    backend = "redis"

    def __init__(self, client: redis.Redis) -> None:
        self.client = client

    def get_many(self, keys: Sequence[str]) -> list[bytes | None]:
        if not keys:
            return []
        try:
            return list(self.client.mget(list(keys)))  # type: ignore[arg-type]
        except redis.RedisError as exc:
            logger.warning("redis read failed, treating as a miss: %s", exc)
            return [None] * len(keys)

    def set_many(self, items: dict[str, bytes], ttl: float | None = None) -> None:
        if not items:
            return
        try:
            with self.client.pipeline(transaction=False) as pipe:
                for key, value in items.items():
                    pipe.set(key, value, px=int(ttl * 1000) if ttl else None)
                pipe.execute()
        except redis.RedisError as exc:
            logger.warning("redis write failed, entry not cached: %s", exc)

    def delete(self, key: str) -> None:
        try:
            self.client.delete(key)
        except redis.RedisError as exc:
            logger.warning("redis delete failed: %s", exc)

    def close(self) -> None:
        self.client.close()


def build_cache(settings: Settings) -> Cache:
    """Redis when configured and reachable, otherwise an in-process cache."""
    if settings.redis_url:
        client = redis.Redis.from_url(
            settings.redis_url, socket_timeout=1.0, socket_connect_timeout=1.0
        )
        try:
            client.ping()
        except redis.RedisError as exc:
            logger.warning(
                "redis at %s unreachable (%s); using an in-process cache instead",
                settings.redis_url,
                exc,
            )
            client.close()
        else:
            return RedisCache(client)
    return MemoryCache(settings.cache_max_entries)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class CachedEmbeddings(Embeddings):
    """Caches vectors per embedding model and text, so unchanged text is embedded once.

    Document and query vectors are cached under separate keys, because some models
    embed the two differently. Vectors are stored as packed float64 values, which
    round-trips them exactly.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        cache: Cache,
        model_id: str,
        *,
        ttl: float | None = None,
        prefix: str = "rag:",
    ) -> None:
        self.embeddings = embeddings
        self.cache = cache
        self.ttl = ttl
        self.namespace = f"{prefix}emb:{digest(model_id)[:16]}:"
        self.stats = CacheStats("embedding")

    def _key(self, kind: str, text: str) -> str:
        return f"{self.namespace}{kind}:{digest(text)}"

    @staticmethod
    def _encode(vector: Sequence[float]) -> bytes:
        return array("d", vector).tobytes()

    @staticmethod
    def _decode(raw: bytes) -> list[float]:
        return array("d", raw).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cached = self.cache.get_many([self._key("doc", text) for text in texts])
        vectors: list[list[float] | None] = [
            self._decode(raw) if raw is not None else None for raw in cached
        ]
        missing: dict[str, list[int]] = {}
        for i, vector in enumerate(vectors):
            if vector is None:
                missing.setdefault(texts[i], []).append(i)
        misses = sum(len(positions) for positions in missing.values())
        self.stats.record(hits=len(texts) - misses, misses=misses)
        if missing:
            fresh = self.embeddings.embed_documents(list(missing))
            for positions, vector in zip(missing.values(), fresh, strict=True):
                for i in positions:
                    vectors[i] = vector
            self.cache.set_many(
                {
                    self._key("doc", text): self._encode(vector)
                    for text, vector in zip(missing, fresh, strict=True)
                },
                self.ttl,
            )
        return [vector for vector in vectors if vector is not None]

    def embed_query(self, text: str) -> list[float]:
        key = self._key("query", text)
        raw = self.cache.get(key)
        if raw is not None:
            self.stats.record(hits=1, misses=0)
            return self._decode(raw)
        self.stats.record(hits=0, misses=1)
        vector = self.embeddings.embed_query(text)
        self.cache.set(key, self._encode(vector), self.ttl)
        return vector
