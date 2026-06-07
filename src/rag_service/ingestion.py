"""Loading files into LangChain documents and splitting them into chunks."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable
from pathlib import Path, PurePath

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from pypdf import PdfReader

TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown"})
PDF_SUFFIXES = frozenset({".pdf"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})


class UnsupportedFileError(ValueError):
    """Raised when a file type cannot be ingested."""


def safe_source_name(filename: str) -> str:
    """Reduce a client-supplied filename to a bare, non-empty file name."""
    name = PurePath(filename.replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise UnsupportedFileError(f"invalid file name: {filename!r}")
    return name


def checked_source_name(filename: str) -> str:
    """:func:`safe_source_name`, also refusing file types that cannot be ingested."""
    name = safe_source_name(filename)
    if PurePath(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise UnsupportedFileError(f"{name}: unsupported file type (expected one of {supported})")
    return name


def load_bytes(source: str, data: bytes) -> list[Document]:
    """Turn the raw bytes of one file into documents.

    Text and Markdown produce a single document. PDFs produce one document per page
    that has extractable text, with the 1-based page number in the metadata.
    """
    suffix = PurePath(source).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = data.decode("utf-8-sig", errors="replace")
        if not text.strip():
            return []
        return [Document(page_content=text, metadata={"source": source})]
    if suffix in PDF_SUFFIXES:
        reader = PdfReader(io.BytesIO(data))
        pages: list[Document] = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    Document(page_content=text, metadata={"source": source, "page": number})
                )
        return pages
    supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
    raise UnsupportedFileError(f"{source}: unsupported file type (expected one of {supported})")


def load_directory(directory: Path) -> list[Document]:
    """Load every supported file below ``directory``, in a stable order.

    The ``source`` metadata is the path relative to ``directory`` in POSIX form, so the
    same corpus produces the same source names on every platform.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            source = path.relative_to(directory).as_posix()
            documents.extend(load_bytes(source, path.read_bytes()))
    return documents


def build_splitter(
    chunk_size: int, chunk_overlap: int, *, markdown: bool
) -> RecursiveCharacterTextSplitter:
    """Create a recursive splitter; Markdown files split on headings first."""
    if markdown:
        return RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )


def chunk_id(source: str, index: int, content: str) -> str:
    """Stable identifier for a chunk, used as the vector-store id and fusion key."""
    digest = hashlib.sha256(f"{source}\x00{index}\x00{content}".encode()).hexdigest()
    return digest[:32]


def split_documents(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split documents into chunks that carry ``source``, ``chunk_index`` and ``chunk_id``.

    ``chunk_index`` counts chunks per source, across pages for PDFs.
    """
    markdown_splitter = build_splitter(chunk_size, chunk_overlap, markdown=True)
    text_splitter = build_splitter(chunk_size, chunk_overlap, markdown=False)
    counters: dict[str, int] = {}
    chunks: list[Document] = []
    for document in documents:
        source = str(document.metadata["source"])
        is_markdown = PurePath(source).suffix.lower() in MARKDOWN_SUFFIXES
        splitter = markdown_splitter if is_markdown else text_splitter
        for piece in splitter.split_documents([document]):
            index = counters.get(source, 0)
            counters[source] = index + 1
            piece.metadata["chunk_index"] = index
            piece.metadata["chunk_id"] = chunk_id(source, index, piece.page_content)
            chunks.append(piece)
    return chunks
