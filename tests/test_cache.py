from collections.abc import Iterator
from pathlib import Path

import fakeredis
import pytest
import redis

from rag_service.cache import (
    Cache,
    CachedEmbeddings,
    MemoryCache,
    RedisCache,
    build_cache,
)
from rag_service.config import Settings
from rag_service.providers import HashingEmbeddings
from rag_service.service import RagService


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def fake_redis() -> Iterator[redis.Redis]:
    client = fakeredis.FakeRedis()
    yield client
    client.close()


@pytest.fixture(params=["memory", "redis"])
def cache(request: pytest.FixtureRequest, fake_redis: redis.Redis) -> Cache:
    return MemoryCache() if request.param == "memory" else RedisCache(fake_redis)


def test_round_trip_and_misses(cache: Cache) -> None:
    assert cache.get("missing") is None
    cache.set("a", b"1")
    cache.set_many({"b": b"2", "c": b"3"})
    assert cache.get_many(["a", "missing", "c"]) == [b"1", None, b"3"]
    cache.delete("a")
    assert cache.get("a") is None
    assert cache.get_many([]) == []


def test_memory_cache_expires_entries() -> None:
    clock = Clock()
    cache = MemoryCache(clock=clock)
    cache.set("k", b"v", ttl=10)
    clock.now += 9
    assert cache.get("k") == b"v"
    clock.now += 1
    assert cache.get("k") is None
    assert len(cache) == 0


def test_memory_cache_evicts_least_recently_used() -> None:
    cache = MemoryCache(max_entries=2)
    cache.set("a", b"1")
    cache.set("b", b"2")
    cache.get("a")
    cache.set("c", b"3")
    assert cache.get_many(["a", "b", "c"]) == [b"1", None, b"3"]


def test_redis_cache_sets_a_ttl(fake_redis: redis.Redis) -> None:
    RedisCache(fake_redis).set("k", b"v", ttl=30)
    assert 0 < fake_redis.pttl("k") <= 30_000


class BrokenRedis(fakeredis.FakeRedis):
    def execute_command(self, *args: object, **kwargs: object) -> object:
        raise redis.ConnectionError("connection refused")


def test_redis_errors_are_misses_not_failures() -> None:
    cache = RedisCache(BrokenRedis())
    assert cache.get_many(["a", "b"]) == [None, None]
    cache.set("a", b"1")
    cache.delete("a")


def test_build_cache_falls_back_to_memory_when_redis_is_unreachable(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, redis_url="redis://127.0.0.1:1/0")
    assert isinstance(build_cache(settings), MemoryCache)
    assert isinstance(build_cache(Settings(_env_file=None)), MemoryCache)


def test_build_cache_uses_redis_when_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    server = fakeredis.FakeServer()
    monkeypatch.setattr(
        redis.Redis,
        "from_url",
        classmethod(lambda cls, url, **kwargs: fakeredis.FakeRedis(server=server)),
    )
    cache = build_cache(Settings(_env_file=None, redis_url="redis://cache:6379/0"))
    assert isinstance(cache, RedisCache)


class CountingEmbeddings(HashingEmbeddings):
    def __init__(self) -> None:
        super().__init__(dim=32)
        self.documents: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents.extend(texts)
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return super().embed_query(text)


def test_cached_embeddings_embed_each_text_once(cache: Cache) -> None:
    inner = CountingEmbeddings()
    embeddings = CachedEmbeddings(inner, cache, "fake:hashing-32")

    first = embeddings.embed_documents(["alpha", "beta", "alpha"])
    second = embeddings.embed_documents(["beta", "gamma"])

    assert inner.documents == ["alpha", "beta", "gamma"]
    assert first[0] == first[2] == HashingEmbeddings(dim=32).embed_query("alpha")
    assert second[0] == first[1]
    assert embeddings.stats.hits == 1
    assert embeddings.stats.misses == 4
    assert embeddings.embed_documents([]) == []


def test_cached_query_vectors_are_separate_from_document_vectors(cache: Cache) -> None:
    inner = CountingEmbeddings()
    embeddings = CachedEmbeddings(inner, cache, "fake:hashing-32")

    embeddings.embed_documents(["alpha"])
    embeddings.embed_query("alpha")
    embeddings.embed_query("alpha")

    assert inner.queries == ["alpha"]
    assert embeddings.stats.hit_ratio == pytest.approx(1 / 3)


def test_models_do_not_share_cache_entries(cache: Cache) -> None:
    inner = CountingEmbeddings()
    CachedEmbeddings(inner, cache, "model-a").embed_query("text")
    CachedEmbeddings(inner, cache, "model-b").embed_query("text")
    assert inner.queries == ["text", "text"]


def test_service_embeds_unchanged_text_once_across_tenants(tmp_path: Path) -> None:
    inner = CountingEmbeddings()
    settings = Settings(_env_file=None, index_dir=tmp_path, fake_embedding_dim=32)
    service = RagService(settings, embeddings=inner)
    files = [("notes.md", b"# Notes\n\nThe deploy window is Tuesday.")]

    service.ingest_files(files, tenant_id="acme")
    embedded = len(inner.documents)
    service.ingest_files(files, tenant_id="globex")

    assert embedded > 0
    assert len(inner.documents) == embedded
    assert isinstance(service.embeddings, CachedEmbeddings)


def test_caching_can_be_disabled(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, index_dir=tmp_path, cache_enabled=False)
    service = RagService(settings)
    assert service.cache is None
    assert isinstance(service.embeddings, HashingEmbeddings)
