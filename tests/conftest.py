from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import SAMPLE_DOCS


@pytest.fixture
def sample_docs_dir() -> Path:
    return SAMPLE_DOCS
