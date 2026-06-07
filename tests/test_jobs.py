import asyncio
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_service.api import create_app
from rag_service.auth import hash_api_key
from rag_service.config import Settings
from rag_service.ingestion import UnsupportedFileError
from rag_service.jobs import (
    IngestWorker,
    JobStatus,
    JobStore,
    content_hash,
    create_job_engine,
    sqlite_url,
)
from rag_service.store import IngestResult
from tests.helpers import SAMPLE_DOCS

FILES = [("notes.md", b"# Notes\n\nThe deploy window is Tuesday.")]


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path: Path, clock: Clock) -> Iterator[JobStore]:
    job_store = JobStore(
        create_job_engine(sqlite_url(tmp_path / "jobs.sqlite3")),
        max_attempts=3,
        retry_backoff_seconds=10,
        idempotency_seconds=3600,
        clock=clock,
    )
    yield job_store
    job_store.close()


def ok_result(files: list[tuple[str, bytes]]) -> IngestResult:
    return IngestResult(sources=[name for name, _ in files], chunks_added=1, chunks_removed=0)


def test_content_hash_ignores_file_order() -> None:
    a, b = ("a.md", b"alpha"), ("b.md", b"beta")
    assert content_hash([a, b]) == content_hash([b, a])
    assert content_hash([a]) != content_hash([("a.md", b"alpha!")])
    assert content_hash([a]) != content_hash([("renamed.md", b"alpha")])


def test_submit_is_idempotent_per_tenant(store: JobStore) -> None:
    job, created = store.submit("acme", FILES)
    again, created_again = store.submit("acme", FILES)
    other, created_other = store.submit("globex", FILES)

    assert created and not created_again and created_other
    assert again.id == job.id
    assert other.id != job.id
    assert job.status is JobStatus.QUEUED
    assert job.files == ["notes.md"]
    assert store.queued_count() == 2


def test_jobs_are_only_visible_to_their_tenant(store: JobStore) -> None:
    job, _ = store.submit("acme", FILES)
    assert store.get(job.id, "acme") is not None
    assert store.get(job.id, "globex") is None
    assert [j.id for j in store.list("acme")] == [job.id]
    assert store.list("globex") == []


def test_claim_runs_each_job_once(store: JobStore) -> None:
    job, _ = store.submit("acme", FILES)

    claimed = store.claim()
    assert claimed is not None
    assert claimed.job.id == job.id
    assert claimed.job.status is JobStatus.RUNNING
    assert claimed.job.attempts == 1
    assert claimed.payload == FILES
    assert store.claim() is None


def test_complete_records_the_result(store: JobStore) -> None:
    job, _ = store.submit("acme", FILES)
    store.claim()
    store.complete(job.id, ok_result(FILES))

    done = store.get(job.id, "acme")
    assert done is not None
    assert done.status is JobStatus.SUCCEEDED
    assert done.result == {
        "sources": ["notes.md"],
        "chunks_added": 1,
        "chunks_removed": 0,
        "unchanged": [],
    }
    assert store.submit("acme", FILES)[0].id == job.id


def test_succeeded_jobs_stop_deduplicating_after_the_window(store: JobStore, clock: Clock) -> None:
    job, _ = store.submit("acme", FILES)
    store.claim()
    store.complete(job.id, ok_result(FILES))

    clock.advance(3601)
    fresh, created = store.submit("acme", FILES)

    assert created
    assert fresh.id != job.id


def test_retries_back_off_then_fail(store: JobStore, clock: Clock) -> None:
    job, _ = store.submit("acme", FILES)

    store.claim()
    assert store.fail(job.id, "timeout", retryable=True) is JobStatus.QUEUED
    assert store.claim() is None  # backing off for 10 s
    clock.advance(10)
    assert store.claim() is not None

    assert store.fail(job.id, "timeout", retryable=True) is JobStatus.QUEUED
    clock.advance(19)
    assert store.claim() is None  # second delay is 20 s
    clock.advance(1)
    assert store.claim() is not None

    assert store.fail(job.id, "timeout", retryable=True) is JobStatus.FAILED
    failed = store.get(job.id, "acme")
    assert failed is not None
    assert failed.attempts == 3
    assert failed.error == "timeout"

    retry, created = store.submit("acme", FILES)
    assert created
    assert retry.id != job.id


def test_permanent_failures_are_not_retried(store: JobStore) -> None:
    job, _ = store.submit("acme", FILES)
    store.claim()
    assert store.fail(job.id, "bad pdf", retryable=False) is JobStatus.FAILED


def test_recover_requeues_interrupted_jobs(store: JobStore) -> None:
    store.submit("acme", FILES)
    store.claim()
    assert store.recover() == 1
    assert store.claim() is not None


def test_worker_retries_transient_errors(store: JobStore, clock: Clock) -> None:
    calls: list[str] = []

    def flaky(tenant: str, files: list[tuple[str, bytes]]) -> IngestResult:
        calls.append(tenant)
        if len(calls) == 1:
            raise ConnectionError("embedding endpoint unreachable")
        return ok_result(files)

    worker = IngestWorker(store, flaky)
    job, _ = store.submit("acme", FILES)

    assert asyncio.run(worker.run_once()) is not None
    queued = store.get(job.id, "acme")
    assert queued is not None
    assert queued.status is JobStatus.QUEUED
    assert queued.error == "ConnectionError: embedding endpoint unreachable"

    clock.advance(10)
    asyncio.run(worker.run_once())
    done = store.get(job.id, "acme")
    assert done is not None
    assert done.status is JobStatus.SUCCEEDED
    assert done.error is None
    assert calls == ["acme", "acme"]
    assert asyncio.run(worker.run_once()) is None


def test_worker_fails_unparseable_files_immediately(store: JobStore) -> None:
    def broken(tenant: str, files: list[tuple[str, bytes]]) -> IngestResult:
        raise UnsupportedFileError("cannot parse")

    job, _ = store.submit("acme", FILES)
    asyncio.run(IngestWorker(store, broken).run_once())

    failed = store.get(job.id, "acme")
    assert failed is not None
    assert failed.status is JobStatus.FAILED
    assert failed.attempts == 1


# API -------------------------------------------------------------------------------


KEY = "key-for-acme"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        index_dir=tmp_path / "index",
        ingest_root=SAMPLE_DOCS.parent,
        chunk_size=600,
        chunk_overlap=80,
        api_keys={hash_api_key(KEY): "acme"},
        job_poll_seconds=0.05,
    )


def wait_for(client: TestClient, job_id: str, timeout: float = 10.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}", headers={"X-API-Key": KEY}).json()
        if body["status"] in {"succeeded", "failed"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def test_api_ingests_in_the_background(settings: Settings) -> None:
    auth = {"X-API-Key": KEY}
    upload = [("files", ("notes.md", FILES[0][1], "text/markdown"))]
    with TestClient(create_app(settings)) as client:
        response = client.post("/jobs/ingest", files=upload, headers=auth)
        assert response.status_code == 202
        job = response.json()
        assert response.headers["location"] == f"/jobs/{job['id']}"

        finished = wait_for(client, job["id"])
        assert finished["status"] == "succeeded"
        assert finished["result"]["sources"] == ["notes.md"]  # type: ignore[index]

        duplicate = client.post("/jobs/ingest", files=upload, headers=auth)
        assert duplicate.status_code == 200
        assert duplicate.json()["id"] == job["id"]

        answer = client.post(
            "/query", json={"question": "When is the deploy window?"}, headers=auth
        ).json()
        assert "Tuesday" in answer["answer"]
        assert [j["id"] for j in client.get("/jobs", headers=auth).json()["jobs"]] == [job["id"]]


def test_api_rejects_unsupported_files_before_queueing(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/jobs/ingest",
            files=[("files", ("image.png", b"\x89PNG", "image/png"))],
            headers={"X-API-Key": KEY},
        )
        assert response.status_code == 415
        assert client.get("/jobs", headers={"X-API-Key": KEY}).json()["jobs"] == []


def test_api_unknown_job_is_404(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        assert client.get("/jobs/missing", headers={"X-API-Key": KEY}).status_code == 404
        assert client.get("/jobs/missing").status_code == 401


def test_jobs_survive_a_restart_without_a_worker(settings: Settings) -> None:
    settings.ingest_worker = False
    upload = [("files", ("notes.md", FILES[0][1], "text/markdown"))]
    with TestClient(create_app(settings)) as client:
        job = client.post("/jobs/ingest", files=upload, headers={"X-API-Key": KEY}).json()
        assert client.get(f"/jobs/{job['id']}", headers={"X-API-Key": KEY}).json()["status"] == (
            "queued"
        )

    settings.ingest_worker = True
    with TestClient(create_app(settings)) as client:
        assert wait_for(client, job["id"])["status"] == "succeeded"
