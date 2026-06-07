"""Runtime configuration loaded from environment variables (prefix ``RAG_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_service.auth import TENANT_PATTERN, validate_key_digests

Provider = Literal["fake", "openai"]
SearchType = Literal["similarity", "mmr"]
RetrieverMode = Literal["dense", "bm25", "hybrid"]
VectorStoreKind = Literal["faiss", "qdrant"]


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

    # HTTP server ----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    # Tenancy -------------------------------------------------------------
    api_keys: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "JSON object mapping the SHA-256 hex digest of each API key to its tenant id. "
            "Empty disables authentication; every request then uses default_tenant."
        ),
    )
    default_tenant: str = Field(default="default", pattern=TENANT_PATTERN)

    # Storage and ingestion -----------------------------------------------
    vector_store: VectorStoreKind = "faiss"
    index_dir: Path = Field(
        default=Path("var/index"),
        description="FAISS indexes, one subdirectory per tenant; also local Qdrant data.",
    )
    qdrant_url: str | None = Field(
        default=None,
        description=(
            "Qdrant server URL, or ':memory:' for an in-process store. "
            "Unset means an on-disk local store under index_dir/qdrant."
        ),
    )
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_prefix: str = Field(default="rag_", pattern=r"^[A-Za-z0-9_-]*$")
    qdrant_sync_seconds: float = Field(
        default=5.0,
        ge=0.0,
        description="How often a replica checks Qdrant for writes made by other replicas.",
    )
    ingest_root: Path = Field(
        default=Path("data"),
        description="Directory ingestion through the API is restricted to this tree.",
    )
    ingest_on_startup: Path | None = None
    startup_tenant: str | None = Field(
        default=None,
        pattern=TENANT_PATTERN,
        description="Tenant that receives the startup ingest; defaults to default_tenant.",
    )
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    chunk_size: int = Field(default=800, ge=50)
    chunk_overlap: int = Field(default=120, ge=0)

    # Background ingestion ------------------------------------------------
    ingest_worker: bool = Field(
        default=True, description="Run the background ingestion worker in this process."
    )
    jobs_database_url: str | None = Field(
        default=None,
        description="SQLAlchemy URL of the job table; unset means SQLite in index_dir.",
    )
    job_max_attempts: int = Field(default=3, ge=1, le=20)
    job_retry_backoff_seconds: float = Field(default=2.0, ge=0.0)
    job_poll_seconds: float = Field(default=1.0, gt=0.0)
    job_idempotency_seconds: float = Field(
        default=24 * 3600.0,
        ge=0.0,
        description="How long a succeeded job answers resubmissions of the same upload.",
    )

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

    @field_validator(
        "openai_base_url",
        "openai_api_key",
        "ingest_on_startup",
        "qdrant_url",
        "qdrant_api_key",
        "jobs_database_url",
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """Treat empty strings (e.g. ``VAR=`` in compose files) as not configured."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("api_keys")
    @classmethod
    def _check_api_keys(cls, value: dict[str, str]) -> dict[str, str]:
        return validate_key_digests(value)

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
