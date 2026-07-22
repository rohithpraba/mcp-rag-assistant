"""Tests for page-aware PDF chunking and citations."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from mcp_rag_assistant.rag.chunking.pdf_pages import (
    chunk_pdf_document,
)
from mcp_rag_assistant.rag.ingestion.pdf_file import (
    load_local_pdf,
)
from mcp_rag_assistant.rag.retrieval.service import (
    format_chunk_citation,
)
from mcp_rag_assistant.rag.storage.chroma_store import (
    RetrievedChunk,
)


def write_pdf(
    path: Path,
) -> None:
    """Create a predictable two-page PDF."""
    document = pymupdf.open()

    try:
        document.insert_page(
            -1,
            text="one two three four",
        )

        document.insert_page(
            -1,
            text="five six seven eight",
        )

        document.save(path)
    finally:
        document.close()


def as_retrieved_chunk(
    chunk,
) -> RetrievedChunk:
    """Convert one TextChunk into a retrieval result."""
    return RetrievedChunk(
        rank=1,
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        distance=0.2,
        similarity=0.8,
        metadata=chunk.metadata(),
    )


def test_pdf_chunks_preserve_page_ranges(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.pdf"
    write_pdf(path)

    document = load_local_pdf(path)

    chunks = chunk_pdf_document(
        document,
        chunk_size_words=6,
        overlap_words=0,
    )

    assert len(chunks) == 2

    assert (
        chunks[0].page_start,
        chunks[0].page_end,
    ) == (1, 2)

    assert (
        chunks[1].page_start,
        chunks[1].page_end,
    ) == (2, 2)

    assert chunks[0].source_page_count == 2
    assert chunks[1].source_page_count == 2


def test_pdf_citations_include_page_ranges(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.pdf"
    write_pdf(path)

    chunks = chunk_pdf_document(
        load_local_pdf(path),
        chunk_size_words=6,
        overlap_words=0,
    )

    first_citation = format_chunk_citation(
        as_retrieved_chunk(chunks[0])
    )

    second_citation = format_chunk_citation(
        as_retrieved_chunk(chunks[1])
    )

    assert first_citation == (
        "[notes.pdf, pages 1-2, chunk 1/2]"
    )

    assert second_citation == (
        "[notes.pdf, page 2, chunk 2/2]"
    )
