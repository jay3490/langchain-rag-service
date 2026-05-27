from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_service.config import Settings


def test_defaults_need_no_credentials() -> None:
    settings = Settings(_env_file=None)
    assert settings.provider == "fake"
    assert settings.openai_api_key is None
    assert settings.retriever_mode == "hybrid"


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PROVIDER", "openai")
    monkeypatch.setenv("RAG_OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_INDEX_DIR", "/tmp/rag-index")

    settings = Settings(_env_file=None)

    assert settings.provider == "openai"
    assert settings.openai_base_url == "http://localhost:11434/v1"
    assert settings.top_k == 7
    assert settings.index_dir == Path("/tmp/rag-index")


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_fetch_k_must_cover_top_k() -> None:
    with pytest.raises(ValidationError, match="fetch_k"):
        Settings(_env_file=None, top_k=10, fetch_k=5)


def test_fusion_needs_a_positive_weight() -> None:
    with pytest.raises(ValidationError, match="weight"):
        Settings(_env_file=None, dense_weight=0, bm25_weight=0)


def test_blank_values_count_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_OPENAI_API_KEY", "")
    monkeypatch.setenv("RAG_OPENAI_BASE_URL", " ")
    monkeypatch.setenv("RAG_INGEST_ON_STARTUP", "")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is None
    assert settings.openai_base_url is None
    assert settings.ingest_on_startup is None
