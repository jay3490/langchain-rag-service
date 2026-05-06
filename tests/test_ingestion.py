from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_service.ingestion import (
    UnsupportedFileError,
    build_splitter,
    load_bytes,
    load_directory,
    safe_source_name,
    split_documents,
)
from tests.helpers import build_pdf


def test_splitter_uses_configured_size_and_overlap() -> None:
    splitter = build_splitter(120, 30, markdown=False)
    assert splitter._chunk_size == 120
    assert splitter._chunk_overlap == 30


def test_markdown_splitter_prefers_heading_boundaries() -> None:
    splitter = build_splitter(200, 0, markdown=True)
    text = "# Title\n\n" + "alpha " * 25 + "\n\n## Second section\n\n" + "beta " * 25
    chunks = splitter.split_text(text)
    assert any(chunk.startswith("## Second section") for chunk in chunks)


def test_chunks_respect_size_and_overlap() -> None:
    words = " ".join(f"word{i}" for i in range(300))
    chunks = split_documents([Document(page_content=words, metadata={"source": "a.txt"})], 200, 50)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 200 for chunk in chunks)
    first_tail = chunks[0].page_content.split()[-1]
    assert first_tail in chunks[1].page_content.split()


def test_chunk_metadata_is_complete_and_stable() -> None:
    document = Document(page_content="one two three " * 100, metadata={"source": "notes.txt"})

    first = split_documents([document], 100, 10)
    second = split_documents([document], 100, 10)

    assert [c.metadata["chunk_index"] for c in first] == list(range(len(first)))
    assert all(c.metadata["source"] == "notes.txt" for c in first)
    assert all("start_index" in c.metadata for c in first)
    assert [c.metadata["chunk_id"] for c in first] == [c.metadata["chunk_id"] for c in second]
    assert len({c.metadata["chunk_id"] for c in first}) == len(first)


def test_load_text_and_markdown() -> None:
    docs = load_bytes("guide.md", b"# Heading\n\nBody text.")
    assert len(docs) == 1
    assert docs[0].metadata == {"source": "guide.md"}
    assert load_bytes("empty.txt", b"   \n") == []


def test_load_pdf_yields_one_document_per_page() -> None:
    docs = load_bytes("manual.pdf", build_pdf(["First page text", "Second page text"]))

    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "First page text" in docs[0].page_content
    assert "Second page text" in docs[1].page_content


def test_unsupported_file_type_is_rejected() -> None:
    with pytest.raises(UnsupportedFileError, match="unsupported"):
        load_bytes("image.png", b"\x89PNG")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("report.md", "report.md"), ("../../etc/passwd.txt", "passwd.txt"), ("a\\b\\c.txt", "c.txt")],
)
def test_safe_source_name_strips_directories(raw: str, expected: str) -> None:
    assert safe_source_name(raw) == expected


def test_safe_source_name_rejects_empty_names() -> None:
    with pytest.raises(UnsupportedFileError):
        safe_source_name("../")


def test_load_directory_uses_relative_posix_sources(sample_docs_dir: Path) -> None:
    docs = load_directory(sample_docs_dir)
    sources = [d.metadata["source"] for d in docs]

    assert sources == sorted(sources)
    assert "bm25.md" in sources
    assert "glossary.txt" in sources
    assert all("/" not in s and "\\" not in s for s in sources)
