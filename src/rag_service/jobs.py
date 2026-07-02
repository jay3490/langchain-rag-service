"""Background ingestion: a job table in SQL (SQLite by default) and an asyncio worker.

Uploads submitted through ``POST /jobs/ingest`` are written to the job table together
with their bytes and acknowledged straight away. :class:`IngestWorker` claims queued
jobs one at a time, runs the ingestion in a thread, and records the outcome.

* **Retries.** A failure that may be transient (for example an embedding API timeout)
  puts the job back in the queue with exponential backoff until ``max_attempts`` is
  reached. Files that cannot be parsed fail at once.
* **Idempotency.** A job is identified by its tenant and a hash of its files' names and
  contents. Submitting the same upload again while an earlier job is queued, running,
  or succeeded within the idempotency window returns that job instead of a new one.
  The ``active_key`` column carries a unique constraint, so this also holds across
  processes sharing one database.
* **Claiming** is a conditional ``UPDATE ... WHERE status = 'queued'``; only one worker
  can win a job, whichever database is used.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from pypdf.errors import PdfReadError
from sqlalchemy import (
    JSON,
    DateTime,
    Engine,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    TypeDecorator,
    create_engine,
    event,
    func,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    selectinload,
    sessionmaker,
)

from rag_service.ingestion import UnsupportedFileError
from rag_service.logging_setup import request_id_var
from rag_service.store import IngestResult

logger = logging.getLogger(__name__)

#: Errors that will not go away by trying again.
PERMANENT_ERRORS: tuple[type[BaseException], ...] = (UnsupportedFileError, PdfReadError)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Store naive UTC (SQLite has no time zones) and always return aware datetimes."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        return None if value is None else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    pass


class JobRecord(Base):
    __tablename__ = "ingest_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(63))
    content_hash: Mapped[str] = mapped_column(String(64))
    active_key: Mapped[str | None] = mapped_column(String(160), unique=True)
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())
    available_at: Mapped[datetime] = mapped_column(UTCDateTime())

    files: Mapped[list[JobFileRecord]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="JobFileRecord.position"
    )

    __table_args__ = (
        Index("ix_ingest_jobs_claim", "status", "available_at"),
        Index("ix_ingest_jobs_tenant", "tenant_id", "created_at"),
    )


class JobFileRecord(Base):
    __tablename__ = "ingest_job_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("ingest_jobs.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    filename: Mapped[str] = mapped_column(String(255))
    data: Mapped[bytes] = mapped_column(LargeBinary)

    job: Mapped[JobRecord] = relationship(back_populates="files")


@dataclass(frozen=True)
class Job:
    """A read-only view of a job row, safe to use outside a database session."""

    id: str
    tenant_id: str
    status: JobStatus
    attempts: int
    max_attempts: int
    files: list[str]
    content_hash: str
    error: str | None
    result: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, record: JobRecord) -> Job:
        return cls(
            id=record.id,
            tenant_id=record.tenant_id,
            status=JobStatus(record.status),
            attempts=record.attempts,
            max_attempts=record.max_attempts,
            files=[f.filename for f in record.files],
            content_hash=record.content_hash,
            error=record.error,
            result=record.result,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


@dataclass(frozen=True)
class ClaimedJob:
    job: Job
    payload: list[tuple[str, bytes]]


def content_hash(files: Sequence[tuple[str, bytes]]) -> str:
    """Hash of the upload: file names and contents, independent of their order."""
    parts = sorted(f"{name}\x00{hashlib.sha256(data).hexdigest()}" for name, data in files)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def create_job_engine(url: str) -> Engine:
    """Create the engine; SQLite gets WAL mode and a busy timeout for concurrent access."""
    if not url.startswith("sqlite"):
        return create_engine(url, pool_pre_ping=True)
    database = url.split("///", 1)[-1]
    if database and database != ":memory:":
        Path(database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(connection: Any, _record: Any) -> None:
        cursor = connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


class JobStore:
    """Synchronous access to the job table; the worker calls it from a thread."""

    def __init__(
        self,
        engine: Engine,
        *,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 2.0,
        idempotency_seconds: float = 86400.0,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.engine = engine
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.idempotency_window = timedelta(seconds=idempotency_seconds)
        self.clock = clock
        self._sessions = sessionmaker(engine, expire_on_commit=False)
        Base.metadata.create_all(engine)

    def close(self) -> None:
        self.engine.dispose()

    # Submission ------------------------------------------------------------------

    def _existing(self, session: Session, key: str) -> JobRecord | None:
        return session.scalar(
            select(JobRecord)
            .where(JobRecord.active_key == key)
            .options(selectinload(JobRecord.files))
        )

    def submit(self, tenant_id: str, files: Sequence[tuple[str, bytes]]) -> tuple[Job, bool]:
        """Queue ``files`` for ``tenant_id``; returns the job and whether it is new."""
        digest = content_hash(files)
        key = f"{tenant_id}:{digest}"
        now = self.clock()
        try:
            with self._sessions.begin() as session:
                return self._submit(session, tenant_id, digest, key, files, now)
        except IntegrityError:
            # Another process queued the same upload between our check and insert.
            with self._sessions() as session:
                winner = self._existing(session, key)
                if winner is None:
                    raise
                return Job.from_record(winner), False

    def _submit(
        self,
        session: Session,
        tenant_id: str,
        digest: str,
        key: str,
        files: Sequence[tuple[str, bytes]],
        now: datetime,
    ) -> tuple[Job, bool]:
        existing = self._existing(session, key)
        if existing is not None:
            expired = (
                existing.status == JobStatus.SUCCEEDED
                and now - existing.updated_at > self.idempotency_window
            )
            if not expired:
                return Job.from_record(existing), False
            existing.active_key = None
            session.flush()
        record = JobRecord(
            id=uuid.uuid4().hex,
            tenant_id=tenant_id,
            content_hash=digest,
            active_key=key,
            status=JobStatus.QUEUED,
            attempts=0,
            max_attempts=self.max_attempts,
            created_at=now,
            updated_at=now,
            available_at=now,
            files=[
                JobFileRecord(position=i, filename=name, data=data)
                for i, (name, data) in enumerate(files)
            ],
        )
        session.add(record)
        session.flush()
        return Job.from_record(record), True

    # Queries ---------------------------------------------------------------------

    def get(self, job_id: str, tenant_id: str) -> Job | None:
        """The job, if it exists and belongs to ``tenant_id``."""
        with self._sessions() as session:
            record = session.scalar(
                select(JobRecord)
                .where(JobRecord.id == job_id, JobRecord.tenant_id == tenant_id)
                .options(selectinload(JobRecord.files))
            )
            return Job.from_record(record) if record else None

    def list(self, tenant_id: str, limit: int = 20) -> list[Job]:
        with self._sessions() as session:
            records = session.scalars(
                select(JobRecord)
                .where(JobRecord.tenant_id == tenant_id)
                .order_by(JobRecord.created_at.desc(), JobRecord.id)
                .limit(limit)
                .options(selectinload(JobRecord.files))
            )
            return [Job.from_record(r) for r in records]

    def queued_count(self) -> int:
        with self._sessions() as session:
            count = session.scalar(
                select(func.count())
                .select_from(JobRecord)
                .where(JobRecord.status == JobStatus.QUEUED)
            )
            return int(count or 0)

    # Worker side -----------------------------------------------------------------

    def recover(self) -> int:
        """Requeue jobs left ``running`` by a process that stopped mid-job."""
        with self._sessions.begin() as session:
            result = session.execute(
                update(JobRecord)
                .where(JobRecord.status == JobStatus.RUNNING)
                .values(status=JobStatus.QUEUED, updated_at=self.clock())
            )
            return int(getattr(result, "rowcount", 0) or 0)

    def claim(self) -> ClaimedJob | None:
        """Take the oldest job that is due, or return ``None`` if there is none."""
        now = self.clock()
        while True:
            with self._sessions.begin() as session:
                job_id = session.scalar(
                    select(JobRecord.id)
                    .where(JobRecord.status == JobStatus.QUEUED, JobRecord.available_at <= now)
                    .order_by(JobRecord.available_at, JobRecord.created_at)
                    .limit(1)
                )
                if job_id is None:
                    return None
                won = session.execute(
                    update(JobRecord)
                    .where(JobRecord.id == job_id, JobRecord.status == JobStatus.QUEUED)
                    .values(
                        status=JobStatus.RUNNING,
                        attempts=JobRecord.attempts + 1,
                        updated_at=now,
                    )
                )
                if not getattr(won, "rowcount", 0):
                    continue  # another worker took it; look again
                record = session.get_one(JobRecord, job_id, populate_existing=True)
                payload = [(f.filename, f.data) for f in record.files]
                return ClaimedJob(Job.from_record(record), payload)

    def complete(self, job_id: str, result: IngestResult) -> None:
        with self._sessions.begin() as session:
            record = session.get_one(JobRecord, job_id)
            record.status = JobStatus.SUCCEEDED
            record.error = None
            record.result = {
                "sources": result.sources,
                "chunks_added": result.chunks_added,
                "chunks_removed": result.chunks_removed,
                "unchanged": result.unchanged,
            }
            record.updated_at = self.clock()

    def fail(self, job_id: str, error: str, *, retryable: bool) -> JobStatus:
        """Record a failed attempt; returns the job's new status."""
        now = self.clock()
        with self._sessions.begin() as session:
            record = session.get_one(JobRecord, job_id)
            record.error = error
            record.updated_at = now
            if retryable and record.attempts < record.max_attempts:
                delay = self.retry_backoff_seconds * 2 ** (record.attempts - 1)
                record.status = JobStatus.QUEUED
                record.available_at = now + timedelta(seconds=delay)
            else:
                record.status = JobStatus.FAILED
                # A failed upload may be submitted again as a new job.
                record.active_key = None
            return JobStatus(record.status)


IngestFunction = Callable[[str, list[tuple[str, bytes]]], IngestResult]


class IngestWorker:
    """Runs queued jobs in the background of the API process."""

    def __init__(
        self, store: JobStore, ingest: IngestFunction, *, poll_seconds: float = 1.0
    ) -> None:
        self.store = store
        self.ingest = ingest
        self.poll_seconds = poll_seconds
        self._wake = asyncio.Event()
        self._stopping = False
        self._task: asyncio.Task[None] | None = None

    def notify(self) -> None:
        """Wake the worker after a submission instead of waiting for the next poll."""
        self._wake.set()

    async def run_once(self) -> Job | None:
        """Process one due job; returns it as it was claimed, or ``None`` if idle."""
        claimed = await asyncio.to_thread(self.store.claim)
        if claimed is None:
            return None
        job = claimed.job
        token = request_id_var.set(f"job-{job.id}")
        try:
            return await self._process(claimed)
        finally:
            request_id_var.reset(token)

    async def _process(self, claimed: ClaimedJob) -> Job:
        job = claimed.job
        logger.info(
            "ingest job %s started (tenant %s, attempt %d of %d)",
            job.id,
            job.tenant_id,
            job.attempts,
            job.max_attempts,
        )
        try:
            result = await asyncio.to_thread(self.ingest, job.tenant_id, claimed.payload)
        except Exception as exc:
            retryable = not isinstance(exc, PERMANENT_ERRORS)
            status = await asyncio.to_thread(
                self.store.fail, job.id, f"{type(exc).__name__}: {exc}", retryable=retryable
            )
            logger.warning("ingest job %s attempt %d failed: %s", job.id, job.attempts, exc)
            if status is JobStatus.FAILED:
                logger.error("ingest job %s failed permanently", job.id)
        else:
            await asyncio.to_thread(self.store.complete, job.id, result)
            logger.info("ingest job %s succeeded: %d chunks added", job.id, result.chunks_added)
        return job

    async def run(self) -> None:
        recovered = await asyncio.to_thread(self.store.recover)
        if recovered:
            logger.info("requeued %d interrupted ingest jobs", recovered)
        while not self._stopping:
            try:
                worked = await self.run_once()
            except Exception:
                logger.exception("ingest worker iteration failed")
                worked = None
            if worked is None and not self._stopping:
                self._wake.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), timeout=self.poll_seconds)

    def start(self) -> None:
        if self._task is None:
            self._stopping = False
            self._task = asyncio.create_task(self.run(), name="ingest-worker")

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=30)
            except TimeoutError:
                self._task.cancel()
            self._task = None
