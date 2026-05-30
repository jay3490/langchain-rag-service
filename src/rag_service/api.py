"""HTTP API for ingestion, question answering and session management."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from rag_service import __version__
from rag_service.config import RetrieverMode, SearchType, Settings, get_settings
from rag_service.filters import TENANT_KEY, FilterValue, MetadataFilter
from rag_service.ingestion import UnsupportedFileError
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
    snippet: str


class QueryResponse(BaseModel):
    question: str
    standalone_question: str
    answer: str
    sources: list[Source]
    session_id: str | None = None


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
    retriever_mode: RetrieverMode
    chunks: int


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionResponse(BaseModel):
    session_id: str
    messages: list[Message]


# Application ---------------------------------------------------------------------


def get_service(request: Request) -> RagService:
    return request.app.state.service


ServiceDep = Annotated[RagService, Depends(get_service)]


def require_documents(service: ServiceDep) -> None:
    """Fail with 409 before a stream starts if nothing has been ingested."""
    if service.index.snapshot.is_empty:
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
        metadata_filter=metadata_filter,
    )


OptionsDep = Annotated[RetrievalOptions, Depends(request_options)]


def _ingest_response(service: RagService, result: IngestResult) -> IngestResponse:
    return IngestResponse(
        sources=result.sources,
        chunks_added=result.chunks_added,
        chunks_removed=result.chunks_removed,
        unchanged=result.unchanged,
        total_chunks=len(service.index.snapshot.chunks),
    )


def create_app(settings: Settings | None = None, service: RagService | None = None) -> FastAPI:
    """Build the FastAPI application; pass ``service`` to inject custom models in tests."""
    rag = service or RagService(settings or get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        rag.start()
        yield

    app = FastAPI(
        title="langchain-rag-service",
        version=__version__,
        description="Retrieval-augmented question answering over your own documents.",
        lifespan=lifespan,
    )
    app.state.service = rag

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(service: ServiceDep) -> HealthResponse:
        return HealthResponse(
            provider=service.settings.provider,
            retriever_mode=service.default_options.mode,
            chunks=len(service.index.snapshot.chunks),
        )

    @app.post("/ingest/files", response_model=IngestResponse, tags=["ingest"])
    def ingest_files(
        service: ServiceDep, files: Annotated[list[UploadFile], File()]
    ) -> IngestResponse:
        limit = service.settings.max_upload_bytes
        payload: list[tuple[str, bytes]] = []
        for upload in files:
            data = upload.file.read(limit + 1)
            if len(data) > limit:
                raise HTTPException(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    f"{upload.filename} exceeds the {limit}-byte upload limit",
                )
            payload.append((upload.filename or "", data))
        try:
            result = service.ingest_files(payload)
        except UnsupportedFileError as exc:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
        return _ingest_response(service, result)

    @app.post("/ingest/directory", response_model=IngestResponse, tags=["ingest"])
    def ingest_directory(service: ServiceDep, body: DirectoryIngestRequest) -> IngestResponse:
        try:
            result = service.ingest_directory(Path(body.path))
        except PathOutsideRootError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"no such directory: {body.path}"
            ) from exc
        return _ingest_response(service, result)

    @app.get("/documents", response_model=DocumentsResponse, tags=["ingest"])
    def list_documents(service: ServiceDep) -> DocumentsResponse:
        snapshot = service.index.snapshot
        return DocumentsResponse(
            sources=snapshot.source_counts(), total_chunks=len(snapshot.chunks)
        )

    @app.delete("/documents/{source:path}", status_code=status.HTTP_204_NO_CONTENT, tags=["ingest"])
    def delete_document(service: ServiceDep, source: str) -> None:
        if service.index.remove_source(source) == 0:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown source: {source}")

    @app.post("/query", response_model=QueryResponse, tags=["query"])
    async def query(service: ServiceDep, body: QueryRequest, options: OptionsDep) -> QueryResponse:
        try:
            result = await service.answer(
                body.question, session_id=body.session_id, options=options
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
        service: ServiceDep, body: QueryRequest, options: OptionsDep
    ) -> AsyncIterator[ServerSentEvent]:
        """Stream the answer as Server-Sent Events.

        Events, in order: ``sources`` (retrieved chunks and the standalone question),
        one ``token`` per generated text chunk, then ``done`` with the full answer. A
        failure after the stream has started is reported as an ``error`` event.
        """
        try:
            async for item in service.stream_answer(
                body.question, session_id=body.session_id, options=options
            ):
                yield ServerSentEvent(event=item.event, data=item.data)
        except Exception:
            logger.exception("streaming answer failed")
            yield ServerSentEvent(event="error", data={"detail": "answer generation failed"})

    @app.get("/sessions/{session_id}", response_model=SessionResponse, tags=["sessions"])
    def get_session(service: ServiceDep, session_id: str) -> SessionResponse:
        roles = {"human": "user", "ai": "assistant"}
        messages = [
            Message(role=roles[m.type], content=m.text) for m in service.sessions.get(session_id)
        ]
        return SessionResponse(session_id=session_id, messages=messages)

    @app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["sessions"])
    def delete_session(service: ServiceDep, session_id: str) -> None:
        if not service.sessions.clear(session_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown session: {session_id}")

    return app
