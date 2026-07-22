"""Tests for application-level semantic retrieval."""

from __future__ import annotations

import numpy as np
import pytest

from mcp_rag_assistant.rag.retrieval.service import (
    format_chunk_citation,
    retrieve_chunks,
)
from mcp_rag_assistant.rag.storage.chroma_store import (
    RetrievedChunk,
)


def make_result(
    *,
    with_position: bool = True,
) -> RetrievedChunk:
    """Create one predictable retrieved chunk."""
    metadata: dict[str, object] = {
        "source_name": "notes.md",
        "source_id": "src_notes",
        "source_uri": "file:///test/notes.md",
    }

    if with_position:
        metadata.update(
            {
                "chunk_index": 0,
                "chunk_count": 3,
            }
        )

    return RetrievedChunk(
        rank=1,
        chunk_id="chk_notes_0",
        text="Semantic retrieval returns relevant chunks.",
        distance=0.2,
        similarity=0.8,
        metadata=metadata,
    )


class FakeEmbedder:
    """Record the query and return a deterministic vector."""

    def __init__(self) -> None:
        self.received_query: str | None = None

    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:
        self.received_query = text

        return np.asarray(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )


class FakeStore:
    """Record search arguments and return a fixed result."""

    def __init__(self) -> None:
        self.received_embedding: np.ndarray | None = None
        self.received_top_k: int | None = None
        self.received_source_id: str | None = None

    def search(
        self,
        query_embedding: object,
        *,
        top_k: int = 3,
        source_id: str | None = None,
    ) -> list[RetrievedChunk]:
        self.received_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        self.received_top_k = top_k
        self.received_source_id = source_id

        return [make_result()]


def test_retrieve_chunks_embeds_cleaned_query_and_searches_store() -> None:
    embedder = FakeEmbedder()
    store = FakeStore()

    response = retrieve_chunks(
        "  How does retrieval work?  ",
        embedder=embedder,
        store=store,
        top_k=2,
        source_id="  src_notes  ",
    )

    assert response.query == "How does retrieval work?"
    assert response.requested_top_k == 2
    assert response.source_id == "src_notes"
    assert len(response.results) == 1

    assert (
        embedder.received_query
        == "How does retrieval work?"
    )

    assert store.received_top_k == 2
    assert store.received_source_id == "src_notes"

    np.testing.assert_array_equal(
        store.received_embedding,
        np.asarray(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    )


def test_empty_query_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        retrieve_chunks(
            "   ",
            embedder=FakeEmbedder(),
            store=FakeStore(),
        )


def test_nonpositive_top_k_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        retrieve_chunks(
            "valid query",
            embedder=FakeEmbedder(),
            store=FakeStore(),
            top_k=0,
        )


def test_empty_source_filter_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="source_id",
    ):
        retrieve_chunks(
            "valid query",
            embedder=FakeEmbedder(),
            store=FakeStore(),
            source_id="   ",
        )


def test_citation_includes_source_and_chunk_position() -> None:
    citation = format_chunk_citation(
        make_result(with_position=True)
    )

    assert citation == "[notes.md, chunk 1/3]"


def test_citation_falls_back_to_source_name() -> None:
    citation = format_chunk_citation(
        make_result(with_position=False)
    )

    assert citation == "[notes.md]"
