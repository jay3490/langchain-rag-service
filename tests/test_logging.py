import asyncio
import json
import logging
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_service.api import create_app
from rag_service.config import Settings
from rag_service.jobs import IngestWorker, JobStore, create_job_engine, sqlite_url
from rag_service.logging_setup import (
    JsonFormatter,
    RequestIdFilter,
    configure_logging,
    request_id_var,
)
from rag_service.store import IngestResult


def _record(message: str, **extra: object) -> logging.LogRecord:
    record = logging.getLogger("test").makeRecord(
        "test", logging.INFO, __file__, 1, message, (), None, extra=extra
    )
    RequestIdFilter().filter(record)
    return record


def test_json_formatter_includes_request_id_and_extras() -> None:
    token = request_id_var.set("req-1")
    try:
        line = JsonFormatter().format(_record("hello %s", tenant="acme"))
    finally:
        request_id_var.reset(token)

    entry = json.loads(line)
    assert entry["message"] == "hello %s"
    assert entry["level"] == "INFO"
    assert entry["logger"] == "test"
    assert entry["request_id"] == "req-1"
    assert entry["tenant"] == "acme"
    assert entry["ts"].endswith("+00:00")


def test_json_formatter_renders_exceptions() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.getLogger("test").makeRecord(
            "test", logging.ERROR, __file__, 1, "failed", (), sys.exc_info()
        )
    entry = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in entry["exception"]
    assert "request_id" not in entry


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    access = logging.getLogger("uvicorn.access")
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    access.disabled = False


def test_configure_logging_installs_one_formatted_handler(restore_logging: None) -> None:
    configure_logging("DEBUG", "json")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)
    assert root.level == logging.DEBUG
    assert logging.getLogger("uvicorn.access").disabled

    configure_logging("WARNING", "text")
    assert not isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(_env_file=None, index_dir=tmp_path / "index")
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_responses_carry_a_generated_request_id(client: TestClient) -> None:
    first = client.get("/health").headers["x-request-id"]
    second = client.get("/health").headers["x-request-id"]
    assert re.fullmatch(r"[0-9a-f]{32}", first)
    assert first != second


@pytest.mark.parametrize(
    ("supplied", "echoed"),
    [("trace-abc.123", True), ("has spaces", False), ("x" * 200, False)],
)
def test_well_formed_request_ids_are_kept(client: TestClient, supplied: str, echoed: bool) -> None:
    returned = client.get("/health", headers={"X-Request-ID": supplied}).headers["x-request-id"]
    assert (returned == supplied) is echoed


def test_access_log_has_route_status_and_request_id(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="rag_service.access")
    client.get("/jobs/abc", headers={"X-Request-ID": "req-42"})

    (record,) = [r for r in caplog.records if r.name == "rag_service.access"]
    assert record.route == "/jobs/{job_id}"  # type: ignore[attr-defined]
    assert record.status == 404  # type: ignore[attr-defined]
    assert record.duration_ms >= 0  # type: ignore[attr-defined]
    RequestIdFilter().filter(record)
    entry = json.loads(JsonFormatter().format(record))
    assert entry["path"] == "/jobs/abc"


def test_application_logs_inside_a_request_carry_its_id(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    caplog.handler.addFilter(RequestIdFilter())
    client.post("/query", json={"question": "anything"}, headers={"X-Request-ID": "req-7"})

    ids = {getattr(r, "request_id", None) for r in caplog.records if r.name.startswith("rag")}
    assert "req-7" in ids


def test_worker_logs_carry_the_job_id(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="rag_service.jobs")
    caplog.handler.addFilter(RequestIdFilter())
    store = JobStore(create_job_engine(sqlite_url(tmp_path / "jobs.sqlite3")))
    job, _ = store.submit("acme", [("a.md", b"text")])

    def ingest(tenant: str, files: list[tuple[str, bytes]]) -> IngestResult:
        logging.getLogger("rag_service.jobs").info("ingesting")
        return IngestResult(sources=["a.md"], chunks_added=1, chunks_removed=0)

    asyncio.run(IngestWorker(store, ingest).run_once())
    store.close()

    job_records = [r for r in caplog.records if r.name == "rag_service.jobs"]
    assert job_records
    assert {getattr(r, "request_id", None) for r in job_records} == {f"job-{job.id}"}
    assert request_id_var.get() is None
