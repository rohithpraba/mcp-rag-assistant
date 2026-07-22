"""Apply word-window chunking while preserving PDF page ranges."""

from __future__ import annotations

from dataclasses import replace

from ..ingestion.pdf_file import PdfSourceDocument
from .models import TextChunk
from .word_window import chunk_document


def chunk_pdf_document(
    document: PdfSourceDocument,
    *,
    chunk_size_words: int = 120,
    overlap_words: int = 20,
) -> list[TextChunk]:
    """Chunk an extracted PDF and attach one-based page ranges."""
    base_chunks = chunk_document(
        document.source,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )

    page_aware_chunks: list[TextChunk] = []

    for chunk in base_chunks:
        overlapping_pages = [
            span.page_number
            for span in document.page_spans
            if (
                chunk.start_char < span.end_char
                and chunk.end_char > span.start_char
            )
        ]

        if not overlapping_pages:
            raise RuntimeError(
                "PDF chunk could not be mapped to a page"
            )

        page_aware_chunks.append(
            replace(
                chunk,
                page_start=min(overlapping_pages),
                page_end=max(overlapping_pages),
                source_page_count=document.page_count,
            )
        )

    return page_aware_chunks
