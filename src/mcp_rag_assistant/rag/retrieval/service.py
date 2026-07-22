"""Application-level semantic retrieval over indexed chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..storage.chroma_store import RetrievedChunk


class QueryEmbedder(Protocol):
    """Interface required from a query embedding provider."""

    def embed_query(
        self,
        text: str,
    ) -> NDArray[np.float32]:
        """Generate one embedding from a natural-language query."""
        ...


class ChunkSearcher(Protocol):
    """Interface required from a vector-search implementation."""

    def search(
        self,
        query_embedding: ArrayLike,
        *,
        top_k: int = 3,
        source_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return ranked chunks for one query embedding."""
        ...


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    """Complete response from one semantic-retrieval operation."""

    query: str
    requested_top_k: int
    source_id: str | None
    results: tuple[RetrievedChunk, ...]


def retrieve_chunks(
    query: str,
    *,
    embedder: QueryEmbedder,
    store: ChunkSearcher,
    top_k: int = 3,
    source_id: str | None = None,
) -> RetrievalResponse:
    """Embed a query and retrieve the nearest indexed chunks.

    Args:
        query:
            Natural-language information need.
        embedder:
            Component responsible for query embedding generation.
        store:
            Component responsible for vector similarity search.
        top_k:
            Maximum number of ranked chunks to request.
        source_id:
            Optional source restriction. When provided, only chunks
            belonging to this logical source are considered.

    Returns:
        A retrieval response containing the cleaned query and results.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("query must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    cleaned_source_id: str | None = None

    if source_id is not None:
        if not isinstance(source_id, str):
            raise TypeError("source_id must be a string")

        cleaned_source_id = source_id.strip()

        if not cleaned_source_id:
            raise ValueError(
                "source_id must not be empty when provided"
            )

    query_embedding = embedder.embed_query(
        cleaned_query
    )

    results = store.search(
        query_embedding,
        top_k=top_k,
        source_id=cleaned_source_id,
    )

    return RetrievalResponse(
        query=cleaned_query,
        requested_top_k=top_k,
        source_id=cleaned_source_id,
        results=tuple(results),
    )


def format_chunk_citation(
    result: RetrievedChunk,
) -> str:
    """Create a readable citation label from chunk metadata."""
    source_name_value = result.metadata.get(
        "source_name"
    )

    if (
        isinstance(source_name_value, str)
        and source_name_value.strip()
    ):
        source_name = source_name_value.strip()
    else:
        source_name = "unknown source"

    chunk_index = result.metadata.get(
        "chunk_index"
    )

    chunk_count = result.metadata.get(
        "chunk_count"
    )

    if (
        isinstance(chunk_index, int)
        and isinstance(chunk_count, int)
        and chunk_index >= 0
        and chunk_count > 0
        and chunk_index < chunk_count
    ):
        return (
            f"[{source_name}, "
            f"chunk {chunk_index + 1}/{chunk_count}]"
        )

    return f"[{source_name}]"
