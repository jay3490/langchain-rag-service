"""Runtime configuration loaded from environment variables (prefix ``RAG_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["fake", "openai"]
SearchType = Literal["similarity", "mmr"]
RetrieverMode = Literal["dense", "bm25", "hybrid"]


class Settings(BaseSettings):
    """All tunables for the service.

    Every field can be overridden with an environment variable named
    ``RAG_<FIELD_NAME>`` (case-insensitive) or through a ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model provider -------------------------------------------------------
    provider: Provider = "fake"
    openai_base_url: str | None = Field(
        default=None,
        description="OpenAI-compatible endpoint, e.g. http://localhost:11434/v1 for Ollama.",
    )
    openai_api_key: SecretStr | None = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    fake_embedding_dim: int = Field(default=384, ge=16)

    # Storage and ingestion -----------------------------------------------
    index_dir: Path = Path("var/index")
    ingest_root: Path = Path("data")
    ingest_on_startup: Path | None = None
    chunk_size: int = Field(default=800, ge=50)
    chunk_overlap: int = Field(default=120, ge=0)

    # Retrieval -----------------------------------------------------------
    retriever_mode: RetrieverMode = "hybrid"
    search_type: SearchType = "similarity"
    top_k: int = Field(default=4, ge=1, le=50)
    fetch_k: int = Field(default=20, ge=1, le=200)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=1.0, ge=0.0)
    bm25_weight: float = Field(default=1.0, ge=0.0)

    # Conversation --------------------------------------------------------
    max_history_messages: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.fetch_k < self.top_k:
            raise ValueError("fetch_k must be at least top_k")
        if self.dense_weight == 0 and self.bm25_weight == 0:
            raise ValueError("at least one of dense_weight and bm25_weight must be positive")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
