"""Tests for deterministic overlapping word-window chunking."""

from __future__ import annotations

import pytest

from mcp_rag_assistant.rag.chunking.word_window import (
    CHUNKING_STRATEGY,
    chunk_document,
)
from mcp_rag_assistant.rag.ingestion.local_file import (
    calculate_content_hash,
    normalize_text,
)
from mcp_rag_assistant.rag.ingestion.models import (
    SourceDocument,
)


def make_source(
    text: str,
    *,
    source_id: str = "src_test",
) -> SourceDocument:
    """Create a predictable source document for chunking tests."""
    normalized_text = normalize_text(text)

    return SourceDocument(
        workspace_id="demo",
        source_id=source_id,
        origin_type="local_file",
        media_type="text/plain",
        source_name="notes.txt",
        source_uri="file:///test/notes.txt",
        content_hash=calculate_content_hash(normalized_text),
        byte_size=len(normalized_text.encode("utf-8")),
        ingested_at_utc="2026-07-21T00:00:00Z",
        text=normalized_text,
    )


def test_short_document_produces_one_chunk_with_provenance() -> None:
    document = make_source("alpha beta gamma")

    chunks = chunk_document(
        document,
        chunk_size_words=5,
        overlap_words=1,
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.text == "alpha beta gamma"
    assert chunk.workspace_id == document.workspace_id
    assert chunk.source_id == document.source_id
    assert chunk.source_uri == document.source_uri
    assert chunk.content_hash == document.content_hash
    assert chunk.chunking_strategy == CHUNKING_STRATEGY
    assert chunk.chunk_index == 0
    assert chunk.chunk_count == 1
    assert chunk.start_word == 0
    assert chunk.end_word == 3
    assert chunk.word_count == 3
    assert chunk.chunk_id.startswith("chk_")


def test_adjacent_chunks_repeat_the_configured_overlap() -> None:
    document = make_source(
        "one two three four five six seven eight nine ten"
    )

    chunks = chunk_document(
        document,
        chunk_size_words=4,
        overlap_words=1,
    )

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]

    assert [chunk.start_word for chunk in chunks] == [
        0,
        3,
        6,
    ]

    assert [chunk.end_word for chunk in chunks] == [
        4,
        7,
        10,
    ]


def test_chunk_ids_are_deterministic() -> None:
    document = make_source(
        "one two three four five six seven"
    )

    first_run = chunk_document(
        document,
        chunk_size_words=4,
        overlap_words=1,
    )

    second_run = chunk_document(
        document,
        chunk_size_words=4,
        overlap_words=1,
    )

    assert [chunk.chunk_id for chunk in first_run] == [
        chunk.chunk_id for chunk in second_run
    ]


def test_content_change_produces_new_chunk_ids() -> None:
    first_document = make_source(
        "one two three four five",
        source_id="src_same",
    )

    second_document = make_source(
        "one two changed four five",
        source_id="src_same",
    )

    first_chunks = chunk_document(
        first_document,
        chunk_size_words=4,
        overlap_words=1,
    )

    second_chunks = chunk_document(
        second_document,
        chunk_size_words=4,
        overlap_words=1,
    )

    assert first_document.source_id == second_document.source_id
    assert first_document.content_hash != second_document.content_hash
    assert first_chunks[0].chunk_id != second_chunks[0].chunk_id


def test_chunking_configuration_affects_chunk_identity() -> None:
    document = make_source(
        "one two three four five six seven eight"
    )

    smaller_chunks = chunk_document(
        document,
        chunk_size_words=4,
        overlap_words=1,
    )

    larger_chunks = chunk_document(
        document,
        chunk_size_words=5,
        overlap_words=1,
    )

    assert smaller_chunks[0].chunk_id != larger_chunks[0].chunk_id


@pytest.mark.parametrize(
    ("chunk_size_words", "overlap_words", "message"),
    [
        (0, 0, "greater than zero"),
        (4, -1, "zero or greater"),
        (4, 4, "smaller than chunk_size_words"),
    ],
)
def test_invalid_chunking_configuration_is_rejected(
    chunk_size_words: int,
    overlap_words: int,
    message: str,
) -> None:
    document = make_source("one two three four five")

    with pytest.raises(ValueError, match=message):
        chunk_document(
            document,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )
