"""Tests for local text and Markdown ingestion."""

from pathlib import Path

import pytest

from mcp_rag_assistant.rag.ingestion.local_file import (
    load_local_text_file,
)


def test_load_markdown_file_returns_text_and_metadata(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "notes.md"
    source_path.write_bytes(
        b"# Retrieval\r\n\r\nSemantic search retrieves by meaning.\r\n"
    )

    document = load_local_text_file(
        source_path,
        workspace_id="demo",
    )

    assert document.workspace_id == "demo"
    assert document.origin_type == "local_file"
    assert document.media_type == "text/markdown"
    assert document.source_name == "notes.md"
    assert document.source_uri == source_path.resolve().as_uri()
    assert document.text == (
        "# Retrieval\n\nSemantic search retrieves by meaning."
    )
    assert document.byte_size > 0
    assert document.source_id.startswith("src_")
    assert len(document.content_hash) == 64


def test_source_id_stays_stable_when_file_content_changes(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "notes.txt"
    source_path.write_text(
        "First version",
        encoding="utf-8",
        newline="\n",
    )

    first_document = load_local_text_file(
        source_path,
        workspace_id="demo",
    )

    source_path.write_text(
        "Second version",
        encoding="utf-8",
        newline="\n",
    )

    second_document = load_local_text_file(
        source_path,
        workspace_id="demo",
    )

    assert first_document.source_id == second_document.source_id
    assert first_document.content_hash != second_document.content_hash


def test_equivalent_line_endings_have_same_content_hash(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "notes.txt"

    source_path.write_bytes(b"Line one\nLine two\n")
    unix_document = load_local_text_file(source_path)

    source_path.write_bytes(b"Line one\r\nLine two\r\n")
    windows_document = load_local_text_file(source_path)

    assert unix_document.content_hash == windows_document.content_hash
    assert unix_document.text == windows_document.text


def test_unsupported_file_extension_is_rejected(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "notes.csv"
    source_path.write_text("column,value", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported file type"):
        load_local_text_file(source_path)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    source_path = tmp_path / "empty.txt"
    source_path.write_text(
        "   \n\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="no non-whitespace text"):
        load_local_text_file(source_path)


def test_file_size_limit_is_enforced(tmp_path: Path) -> None:
    source_path = tmp_path / "large.txt"
    source_path.write_text("12345", encoding="utf-8")

    with pytest.raises(ValueError, match="size limit"):
        load_local_text_file(source_path, max_bytes=4)


def test_invalid_workspace_id_is_rejected(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "notes.txt"
    source_path.write_text("Valid content", encoding="utf-8")

    with pytest.raises(ValueError, match="workspace_id"):
        load_local_text_file(
            source_path,
            workspace_id="Invalid Workspace!",
        )
