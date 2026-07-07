"""HTTP API for ingestion, question answering and session management."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    Security,
    UploadFile,
    status,
)
from fastapi.responses import PlainTextResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from rag_service import __version__
from rag_service.auth import ApiKeyAuthenticator
from rag_service.config import RetrieverMode, SearchType, Settings, VectorStoreKind, get_settings
from rag_service.filters import TENANT_KEY, FilterValue, MetadataFilter
from rag_service.guardrails import Refusal
from rag_service.ingestion import UnsupportedFileError, checked_source_name
from rag_service.jobs import IngestWorker, Job, JobStatus, JobStore, create_job_engine, sqlite_url
from rag_service.logging_setup import RequestContextMiddleware
from rag_service.metrics import CONTENT_TYPE as METRICS_CONTENT_TYPE
from rag_service.retrieval import RetrievalOptions
from rag_service.service import EmptyIndexError, PathOutsideRootError, RagService
from rag_service.store import IngestResult

logger = logging.getLogger(__name__)

SESSION_ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,128}$"


# Schemas -------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)
    k: int | None = Field(default=None, ge=1, le=50)
    retriever_mode: RetrieverMode | None = None
    search_type: SearchType | None = None
    rerank: bool | None = Field(
        default=None, description="Set false to skip the configured reranker for this query."
    )
    filters: dict[str, FilterValue] | None = Field(
        default=None,
        description=(
            "Metadata conditions every retrieved chunk must meet, e.g. "
            '{"source": ["bm25.md", "mmr.md"]}. A list means any of its values.'
        ),
    )

    def metadata_filter(self) -> MetadataFilter:
        try:
            return MetadataFilter.from_mapping(self.filters)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


class Source(BaseModel):
    id: int = Field(description="Citation number used in the answer, e.g. [1].")
    source: str
    label: str
    chunk_id: str
    page: int | None = None
    start_index: int | None = None
    score: float | None = Field(
        default=None, description="RRF score (hybrid) or BM25 score; absent for dense search."
    )
    rerank_score: float | None = Field(
        default=None, description="Reranker relevance in [0, 1] when re-ranking ran."
    )
    snippet: str


class QueryResponse(BaseModel):
    question: str
    standalone_question: str
    answer: str
    sources: list[Source]
    session_id: str | None = None
    confidence: float = Field(
        description="Retrieval confidence in [0, 1]: best rerank score or query-word coverage."
    )
    citations: list[int] = Field(description="Source ids cited in the answer.")
    invalid_citations: list[int] = Field(
        description="Cited numbers that matched no source; removed under the strip policy."
    )
    refusal: Refusal | None = Field(
        default=None, description="Why the service declined to answer, if it did."
    )
    cached: bool = Field(default=False, description="Served from the answer cache.")


class DirectoryIngestRequest(BaseModel):
    path: str = Field(default="docs", description="Directory relative to the ingest root.")


class IngestResponse(BaseModel):
    sources: list[str]
    chunks_added: int
    chunks_removed: int
    unchanged: list[str]
    total_chunks: int


class DocumentsResponse(BaseModel):
    sources: dict[str, int] = Field(description="Chunk count per source.")
    total_chunks: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str = __version__
    provider: str
    vector_store: VectorStoreKind
    retriever_mode: RetrieverMode
    auth_enabled: bool
    tenants_loaded: int = Field(description="Tenants whose index this replica has loaded.")
    chunks: int = Field(description="Chunks across the loaded tenant indexes.")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    attempts: int
    max_attempts: int
    files: list[str]
    content_hash: str = Field(description="SHA-256 over the uploaded names and contents.")
    error: str | None = None
    result: dict[str, Any] | None = Field(
        default=None, description="Ingest summary once the job has succeeded."
    )
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, job: Job) -> JobResponse:
        return cls(
            id=job.id,
            status=job.status,
            attempts=job.attempts,
            max_attempts=job.max_attempts,
            files=job.files,
            content_hash=job.content_hash,
            error=job.error,
            result=job.result,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


class JobListResponse(BaseModel):
    jobs: list[JobResponse]


class SessionResponse(BaseModel):
    session_id: str
    messages: list[Message]


# Application ---------------------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_token = HTTPBearer(auto_error=False)


def get_service(request: Request) -> RagService:
    return request.app.state.service


ServiceDep = Annotated[RagService, Depends(get_service)]


def get_tenant(
    request: Request,
    api_key: Annotated[str | None, Security(api_key_header)],
    bearer: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_token)],
) -> str:
    """Resolve the caller's tenant from ``X-API-Key`` or ``Authorization: Bearer``."""
    authenticator: ApiKeyAuthenticator = request.app.state.authenticator
    tenant = authenticator.tenant_for(api_key or (bearer.credentials if bearer else None))
    if tenant is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "a valid API key is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return tenant


TenantDep = Annotated[str, Depends(get_tenant)]


def require_documents(service: ServiceDep, tenant: TenantDep) -> None:
    """Fail with 409 before a stream starts if the tenant has nothing ingested."""
    if service.snapshot_for(tenant).is_empty:
        raise HTTPException(status.HTTP_409_CONFLICT, "the index is empty; ingest documents first")


def request_options(service: ServiceDep, body: QueryRequest) -> RetrievalOptions:
    """Per-request retrieval options; validated before a stream starts."""
    metadata_filter = body.metadata_filter()
    if TENANT_KEY in set(metadata_filter.keys()):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"{TENANT_KEY!r} cannot be filtered on"
        )
    return service.options(
        k=body.k,
        mode=body.retriever_mode,
        search_type=body.search_type,
        rerank=body.rerank,
        metadata_filter=metadata_filter,
    )


OptionsDep = Annotated[RetrievalOptions, Depends(request_options)]


def get_jobs(request: Request) -> JobStore:
    return request.app.state.jobs


JobsDep = Annotated[JobStore, Depends(get_jobs)]


def read_uploads(files: list[UploadFile], limit: int) -> list[tuple[str, bytes]]:
    """Read each upload, refusing any larger than ``limit`` bytes (413)."""
    payload: list[tuple[str, bytes]] = []
    for upload in files:
        data = upload.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"{upload.filename} exceeds the {limit}-byte upload limit",
            )
        payload.append((upload.filename or "", data))
    return payload


def build_job_store(settings: Settings) -> JobStore:
    url = settings.jobs_database_url or sqlite_url(settings.index_dir / "jobs.sqlite3")
    return JobStore(
        create_job_engine(url),
        max_attempts=settings.job_max_attempts,
        retry_backoff_seconds=settings.job_retry_backoff_seconds,
        idempotency_seconds=settings.job_idempotency_seconds,
    )


def _ingest_response(service: RagService, tenant: str, result: IngestResult) -> IngestResponse:
    return IngestResponse(
        sources=result.sources,
        chunks_added=result.chunks_added,
        chunks_removed=result.chunks_removed,
        unchanged=result.unchanged,
        total_chunks=len(service.snapshot_for(tenant).chunks),
    )


def create_app(settings: Settings | None = None, service: RagService | None = None) -> FastAPI:
    """Build the FastAPI application; pass ``service`` to inject custom models in tests."""
    rag = service or RagService(settings or get_settings())
    authenticator = ApiKeyAuthenticator(rag.settings.api_keys, rag.default_tenant)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not authenticator.enabled:
            logger.warning(
                "RAG_API_KEYS is empty: authentication is off and every request uses tenant %r",
                rag.default_tenant,
            )
        rag.start()
        jobs = build_job_store(rag.settings)
        app.state.jobs = jobs
        rag.metrics.watch_gauge(
            "rag_ingest_jobs_queued", "Ingest jobs waiting to run.", jobs.queued_count
        )
        worker = IngestWorker(
            jobs,
            lambda tenant, files: rag.ingest_files(files, tenant_id=tenant),
            poll_seconds=rag.settings.job_poll_seconds,
        )
        app.state.worker = worker
        if rag.settings.ingest_worker:
            worker.start()
        try:
            yield
        finally:
            await worker.stop()
            jobs.close()
            rag.close()

    app = FastAPI(
        title="langchain-rag-service",
        version=__version__,
        description="Retrieval-augmented question answering over your own documents.",
        lifespan=lifespan,
    )
    app.state.service = rag
    app.state.authenticator = authenticator
    app.add_middleware(RequestContextMiddleware, observers=[rag.metrics.observe_request])

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(service: ServiceDep) -> HealthResponse:
        loaded = service.indexes.loaded()
        return HealthResponse(
            provider=service.settings.provider,
            vector_store=service.settings.vector_store,
            retriever_mode=service.default_options.mode,
            auth_enabled=authenticator.enabled,
            tenants_loaded=len(loaded),
            chunks=sum(len(index.snapshot.chunks) for index in loaded.values()),
        )

    @app.get("/metrics", tags=["meta"], response_class=PlainTextResponse)
    def metrics(service: ServiceDep) -> PlainTextResponse:
        """Prometheus metrics in the text exposition format."""
        return PlainTextResponse(service.metrics.render(), media_type=METRICS_CONTENT_TYPE)

    @app.post("/ingest/files", response_model=IngestResponse, tags=["ingest"])
    def ingest_files(
        service: ServiceDep, tenant: TenantDep, files: Annotated[list[UploadFile], File()]
    ) -> IngestResponse:
        payload = read_uploads(files, service.settings.max_upload_bytes)
        try:
            result = service.ingest_files(payload, tenant_id=tenant)
        except UnsupportedFileError as exc:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
        return _ingest_response(service, tenant, result)

    @app.post("/ingest/directory", response_model=IngestResponse, tags=["ingest"])
    def ingest_directory(
        service: ServiceDep, tenant: TenantDep, body: DirectoryIngestRequest
    ) -> IngestResponse:
        try:
            result = service.ingest_directory(Path(body.path), tenant_id=tenant)
        except PathOutsideRootError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"no such directory: {body.path}"
            ) from exc
        return _ingest_response(service, tenant, result)

    @app.post(
        "/jobs/ingest",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["jobs"],
        responses={200: {"model": JobResponse, "description": "Same upload already queued"}},
    )
    def submit_ingest_job(
        service: ServiceDep,
        tenant: TenantDep,
        jobs: JobsDep,
        request: Request,
        response: Response,
        files: Annotated[list[UploadFile], File()],
    ) -> JobResponse:
        """Queue files for background ingestion and return the job at once.

        Re-submitting an upload whose job is still queued, running or recently
        succeeded returns that job with status 200 instead of creating another.
        """
        payload = read_uploads(files, service.settings.max_upload_bytes)
        try:
            payload = [(checked_source_name(name), data) for name, data in payload]
        except UnsupportedFileError as exc:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
        job, created = jobs.submit(tenant, payload)
        if created:
            request.app.state.worker.notify()
        else:
            response.status_code = status.HTTP_200_OK
        response.headers["Location"] = f"/jobs/{job.id}"
        return JobResponse.of(job)

    @app.get("/jobs/{job_id}", response_model=JobResponse, tags=["jobs"])
    def get_job(tenant: TenantDep, jobs: JobsDep, job_id: str) -> JobResponse:
        job = jobs.get(job_id, tenant)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown job: {job_id}")
        return JobResponse.of(job)

    @app.get("/jobs", response_model=JobListResponse, tags=["jobs"])
    def list_jobs(
        tenant: TenantDep, jobs: JobsDep, limit: Annotated[int, Query(ge=1, le=100)] = 20
    ) -> JobListResponse:
        return JobListResponse(jobs=[JobResponse.of(job) for job in jobs.list(tenant, limit)])

    @app.get("/documents", response_model=DocumentsResponse, tags=["ingest"])
    def list_documents(service: ServiceDep, tenant: TenantDep) -> DocumentsResponse:
        snapshot = service.snapshot_for(tenant)
        return DocumentsResponse(
            sources=snapshot.source_counts(), total_chunks=len(snapshot.chunks)
        )

    @app.delete("/documents/{source:path}", status_code=status.HTTP_204_NO_CONTENT, tags=["ingest"])
    def delete_document(service: ServiceDep, tenant: TenantDep, source: str) -> None:
        if service.index_for(tenant).remove_source(source) == 0:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown source: {source}")

    @app.post("/query", response_model=QueryResponse, tags=["query"])
    async def query(
        service: ServiceDep, tenant: TenantDep, body: QueryRequest, options: OptionsDep
    ) -> QueryResponse:
        try:
            result = await service.answer(
                body.question, tenant_id=tenant, session_id=body.session_id, options=options
            )
        except EmptyIndexError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return QueryResponse.model_validate({**result, "session_id": body.session_id})

    @app.post(
        "/query/stream",
        response_class=EventSourceResponse,
        dependencies=[Depends(require_documents)],
        tags=["query"],
    )
    async def query_stream(
        service: ServiceDep, tenant: TenantDep, body: QueryRequest, options: OptionsDep
    ) -> AsyncIterator[ServerSentEvent]:
        """Stream the answer as Server-Sent Events.

        Events, in order: ``sources`` (retrieved chunks and the standalone question),
        one ``token`` per generated text chunk, then ``done`` with the full answer. A
        failure after the stream has started is reported as an ``error`` event.
        """
        try:
            async for item in service.stream_answer(
                body.question, tenant_id=tenant, session_id=body.session_id, options=options
            ):
                yield ServerSentEvent(event=item.event, data=item.data)
        except Exception:
            logger.exception("streaming answer failed")
            yield ServerSentEvent(event="error", data={"detail": "answer generation failed"})

    @app.get("/sessions/{session_id}", response_model=SessionResponse, tags=["sessions"])
    def get_session(service: ServiceDep, tenant: TenantDep, session_id: str) -> SessionResponse:
        roles = {"human": "user", "ai": "assistant"}
        history = service.sessions.get(service.session_key(tenant, session_id))
        messages = [Message(role=roles[m.type], content=m.text) for m in history]
        return SessionResponse(session_id=session_id, messages=messages)

    @app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["sessions"])
    def delete_session(service: ServiceDep, tenant: TenantDep, session_id: str) -> None:
        if not service.sessions.clear(service.session_key(tenant, session_id)):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown session: {session_id}")

    return app
