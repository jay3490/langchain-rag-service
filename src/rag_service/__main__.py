"""``rag-service`` entry point: run the API with uvicorn."""

from __future__ import annotations

import logging

import uvicorn

from rag_service.api import create_app
from rag_service.config import get_settings


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
