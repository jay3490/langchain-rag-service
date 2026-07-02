"""``rag-service`` entry point: run the API with uvicorn."""

from __future__ import annotations

import uvicorn

from rag_service.api import create_app
from rag_service.config import get_settings
from rag_service.logging_setup import configure_logging


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_config=None)


if __name__ == "__main__":
    main()
