"""Structured logging and per-request ids.

:class:`RequestContextMiddleware` gives every HTTP request an id (the caller's
``X-Request-ID`` when it is well-formed, otherwise a new one), echoes it in the
response header, and logs one access line per request. The id lives in a context
variable, so every log record emitted while handling the request carries it, including
records from LangChain callbacks and worker threads started with ``asyncio.to_thread``.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Literal

LogFormat = Literal["json", "text"]
REQUEST_ID_HEADER = "x-request-id"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

#: Attributes every LogRecord has; anything else was passed with ``extra=``.
_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()
    | {"message", "asctime", "request_id", "taskName"}
)

access_logger = logging.getLogger("rag_service.access")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class RequestIdFilter(logging.Filter):
    """Copy the current request id onto each record as ``request_id``."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message, request id, extras."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            entry["request_id"] = request_id
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s]: %(message)s"


def configure_logging(level: str = "INFO", fmt: LogFormat = "json") -> None:
    """Send all logs (uvicorn's included) to stderr in the chosen format."""
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FORMAT))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    # The middleware below writes access lines with the request id.
    logging.getLogger("uvicorn.access").disabled = True


def new_request_id() -> str:
    return uuid.uuid4().hex


def route_template(scope: Scope) -> str:
    """The matched route's path template (``/jobs/{job_id}``), or ``unmatched``."""
    route = scope.get("route")
    return str(getattr(route, "path", "unmatched"))


RequestObserver = Callable[[str, str, int, float], None]


class RequestContextMiddleware:
    """Assign request ids, add ``X-Request-ID`` to responses and log each request.

    ``observers`` are called with ``(method, route, status, seconds)`` once a response
    has been sent; the metrics module registers one.
    """

    def __init__(self, app: ASGIApp, observers: list[RequestObserver] | None = None) -> None:
        self.app = app
        self.observers = observers if observers is not None else []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = next(
            (
                value.decode("latin-1")
                for name, value in scope.get("headers", [])
                if name.decode("latin-1").lower() == REQUEST_ID_HEADER
            ),
            None,
        )
        request_id = supplied if supplied and _VALID_REQUEST_ID.match(supplied) else None
        request_id = request_id or new_request_id()
        token = request_id_var.set(request_id)
        status_code = 500
        started = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            elapsed = time.perf_counter() - started
            route = route_template(scope)
            access_logger.info(
                "%s %s %d",
                scope.get("method", ""),
                scope.get("path", ""),
                status_code,
                extra={
                    "method": scope.get("method", ""),
                    "path": scope.get("path", ""),
                    "route": route,
                    "status": status_code,
                    "duration_ms": round(elapsed * 1000, 2),
                },
            )
            for observer in self.observers:
                observer(str(scope.get("method", "")), route, status_code, elapsed)
            request_id_var.reset(token)
