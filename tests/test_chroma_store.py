"""Tests for persistent Chroma chunk storage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

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
    workspace_id: str = "demo",
):
    """Create two predictable chunks."""
    text = (
        "alpha beta gamma delta "
        "epsilon zeta eta theta"
    )

    document = SourceDocument(
        workspace_id=workspace_id,
        source_id="src_test",
        origin_type="local_file",
        media_type="text/plain",
        source_name="notes.txt",
        source_uri="file:///test/notes.txt",
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


def make_store(
    path: Path,
    *,
    model_name: str = "test-model",
) -> ChromaVectorStore:
    """Create a predictable three-dimensional test store."""
    return ChromaVectorStore(
        path,
        workspace_id="demo",
        embedding_model=model_name,
        embedding_dimension=3,
    )


def test_upsert_and_search_returns_nearest_chunk(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "chroma")
    chunks = make_chunks()

    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    inserted = store.upsert_chunks(
        chunks,
        embeddings,
    )

    results = store.search(
        np.asarray(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        top_k=2,
    )

    assert inserted == 2
    assert store.count() == 2
    assert len(results) == 2

    assert results[0].chunk_id == chunks[0].chunk_id
    assert results[0].text == chunks[0].text
    assert results[0].distance == pytest.approx(
        0.0,
        abs=1e-6,
    )
    assert results[0].similarity == pytest.approx(
        1.0,
        abs=1e-6,
    )
    assert (
        results[0].metadata["source_id"]
        == "src_test"
    )


def test_repeated_upsert_keeps_record_count_stable(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "chroma")
    chunks = make_chunks()

    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    store.upsert_chunks(chunks, embeddings)
    store.upsert_chunks(chunks, embeddings)

    assert store.count() == 2


def test_wrong_embedding_dimension_is_rejected(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "chroma")
    chunks = make_chunks()

    wrong_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="expected",
    ):
        store.upsert_chunks(
            chunks,
            wrong_embeddings,
        )


def test_workspace_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "chroma")
    chunks = make_chunks()

    mismatched_chunk = replace(
        chunks[0],
        workspace_id="other",
    )

    with pytest.raises(
        ValueError,
        match="workspace",
    ):
        store.upsert_chunks(
            [mismatched_chunk],
            np.asarray(
                [[1.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
        )


def test_existing_collection_rejects_different_model(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chroma"

    make_store(
        database_path,
        model_name="model-a",
    )

    with pytest.raises(
        ValueError,
        match="embedding_model",
    ):
        make_store(
            database_path,
            model_name="model-b",
        )
