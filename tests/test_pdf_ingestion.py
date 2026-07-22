"""Tests for page-aware PDF ingestion."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from mcp_rag_assistant.rag.ingestion.pdf_file import (
    load_local_pdf,
)


def write_pdf(
    path: Path,
    pages: list[str],
) -> None:
    """Create a small text PDF for testing."""
    path.unlink(missing_ok=True)

    document = pymupdf.open()

    try:
        for text in pages:
            if text:
                document.insert_page(
                    -1,
                    text=text,
                    fontsize=11,
                )
            else:
                document.new_page()

        document.save(path)
    finally:
        document.close()


def test_pdf_ingestion_preserves_page_spans(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.pdf"

    write_pdf(
        path,
        [
            "Page one explains retrieval.",
            "Page two explains citations.",
        ],
    )

    document = load_local_pdf(
        path,
        workspace_id="demo",
    )

    assert document.page_count == 2
    assert document.text_page_count == 2
    assert document.source.media_type == "application/pdf"
    assert document.source.source_name == "notes.pdf"

    assert [
        span.page_number
        for span in document.page_spans
    ] == [1, 2]

    first_span = document.page_spans[0]
    second_span = document.page_spans[1]

    assert "Page one explains retrieval" in (
        document.source.text[
            first_span.start_char:first_span.end_char
        ]
    )

    assert "Page two explains citations" in (
        document.source.text[
            second_span.start_char:second_span.end_char
        ]
    )


def test_pdf_source_id_is_stable_and_hash_changes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.pdf"

    write_pdf(
        path,
        ["First source version."],
    )

    first = load_local_pdf(
        path,
        workspace_id="demo",
    )

    write_pdf(
        path,
        ["Second source version."],
    )

    second = load_local_pdf(
        path,
        workspace_id="demo",
    )

    assert (
        first.source.source_id
        == second.source.source_id
    )

    assert (
        first.source.content_hash
        != second.source.content_hash
    )


def test_image_only_or_blank_pdf_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "blank.pdf"

    write_pdf(
        path,
        [""],
    )

    with pytest.raises(
        ValueError,
        match="no extractable text",
    ):
        load_local_pdf(path)


def test_non_pdf_extension_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text(
        "Not a PDF",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\.pdf extension",
    ):
        load_local_pdf(path)
