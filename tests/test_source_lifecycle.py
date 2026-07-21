"""Tests for source refresh and deletion in Chroma."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mcp_rag_assistant.rag.chunking.models import (
    TextChunk,
)
from mcp_rag_assistant.rag.chunking.word_window import (
    chunk_document,
)
from mcp_rag_assistant.rag.ingestion.local_file import (
    calculate_content_hash,
)
from mcp_rag_assistant.rag.ingestion.models import (
    SourceDocument,
)
from mcp_rag_assistant.rag.storage.chroma_store import (
    ChromaVectorStore,
)


def make_chunks(
    source_id: str,
    text: str,
) -> list[TextChunk]:
    """Create predictable four-word chunks for one source."""
    document = SourceDocument(
        workspace_id="demo",
        source_id=source_id,
        origin_type="local_file",
        media_type="text/plain",
        source_name=f"{source_id}.txt",
        source_uri=f"file:///test/{source_id}.txt",
        content_hash=calculate_content_hash(text),
        byte_size=len(text.encode("utf-8")),
        ingested_at_utc="2026-07-21T00:00:00Z",
        text=text,
    )

    return chunk_document(
        document,
        chunk_size_words=4,
        overlap_words=0,
    )


def make_embeddings(
    count: int,
) -> np.ndarray:
    """Create deterministic normalized three-dimensional vectors."""
    basis_vectors = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )

    return np.asarray(
        [
            basis_vectors[index % 3]
            for index in range(count)
        ],
        dtype=np.float32,
    )


def make_store(
    path: Path,
) -> ChromaVectorStore:
    """Create a test workspace using three-dimensional embeddings."""
    return ChromaVectorStore(
        path,
        workspace_id="demo",
        embedding_model="test-model",
        embedding_dimension=3,
    )


def test_refresh_replaces_all_previous_source_chunks(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "chroma")

    old_chunks = make_chunks(
        "src_main",
        (
            "one two three four five six "
            "seven eight nine ten eleven twelve"
        ),
    )

    first_result = store.replace_source_chunks(
        old_chunks,
        make_embeddings(len(old_chunks)),
    )

    old_ids = {
        chunk.chunk_id
        for chunk in old_chunks
    }

    new_chunks = make_chunks(
        "src_main",
        "one two changed four five",
    )

    second_result = store.replace_source_chunks(
        new_chunks,
        make_embeddings(len(new_chunks)),
    )

    new_ids = {
        chunk.chunk_id
        for chunk in new_chunks
    }

    assert first_result.previous_chunk_count == 0
    assert first_result.current_chunk_count == 3

    assert second_result.previous_chunk_count == 3
    assert second_result.upserted_chunk_count == 2
    assert second_result.deleted_stale_chunk_count == 3
    assert second_result.current_chunk_count == 2
    assert second_result.record_set_changed is True

    assert old_ids.isdisjoint(new_ids)
    assert set(
        store.get_source_chunk_ids("src_main")
    ) == new_ids
    assert store.count() == 2


def test_replacing_unchanged_source_is_idempotent(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "chroma")

    chunks = make_chunks(
        "src_main",
        "one two three four five six",
    )

    embeddings = make_embeddings(len(chunks))

    store.replace_source_chunks(
        chunks,
        embeddings,
    )

    repeated_result = store.replace_source_chunks(
        chunks,
        embeddings,
    )

    assert repeated_result.previous_chunk_count == 2
    assert repeated_result.current_chunk_count == 2
    assert repeated_result.deleted_stale_chunk_count == 0
    assert repeated_result.record_set_changed is False
    assert store.count() == 2


def test_delete_source_preserves_other_sources(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "chroma")

    source_a = make_chunks(
        "src_a",
        "one two three four five six seven eight",
    )

    source_b = make_chunks(
        "src_b",
        "alpha beta gamma delta",
    )

    store.replace_source_chunks(
        source_a,
        make_embeddings(len(source_a)),
    )

    store.replace_source_chunks(
        source_b,
        make_embeddings(len(source_b)),
    )

    assert store.count() == 3

    deleted_count = store.delete_source("src_a")

    assert deleted_count == 2
    assert store.get_source_chunk_ids("src_a") == []
    assert len(
        store.get_source_chunk_ids("src_b")
    ) == 1
    assert store.count() == 1
    assert store.delete_source("src_a") == 0


def test_mixed_sources_cannot_be_replaced_together(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "chroma")

    source_a = make_chunks(
        "src_a",
        "one two three four five six seven eight",
    )

    source_b = make_chunks(
        "src_b",
        "alpha beta gamma delta epsilon zeta eta theta",
    )

    mixed_chunks = [
        source_a[0],
        source_b[1],
    ]

    with pytest.raises(
        ValueError,
        match="same source_id",
    ):
        store.replace_source_chunks(
            mixed_chunks,
            make_embeddings(2),
        )
