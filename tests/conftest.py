from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = REPO_ROOT / "data" / "docs"


@pytest.fixture
def sample_docs_dir() -> Path:
    return SAMPLE_DOCS
