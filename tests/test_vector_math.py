"""Tests for the manual vector-search operations."""

import numpy as np
import pytest

from mcp_rag_assistant.rag.vector_math import (
    cosine_similarity_scores,
    top_k_indices,
)


def test_cosine_similarity_scores_known_directions() -> None:
    query = np.array([1.0, 0.0])

    documents = np.array(
        [
            [1.0, 0.0],   # Same direction
            [0.0, 1.0],   # Perpendicular
            [-1.0, 0.0],  # Opposite direction
        ]
    )

    scores = cosine_similarity_scores(query, documents)

    assert scores == pytest.approx([1.0, 0.0, -1.0], abs=1e-6)


def test_cosine_similarity_is_not_affected_by_vector_scale() -> None:
    query = np.array([2.0, 0.0])

    documents = np.array(
        [
            [10.0, 0.0],
            [0.0, 5.0],
        ]
    )

    scores = cosine_similarity_scores(query, documents)

    assert scores == pytest.approx([1.0, 0.0], abs=1e-6)


def test_top_k_indices_orders_highest_scores_first() -> None:
    scores = np.array([0.25, 0.91, 0.52, 0.10])

    indices = top_k_indices(scores, k=2)

    assert indices.tolist() == [1, 2]


def test_zero_query_vector_is_rejected() -> None:
    query = np.array([0.0, 0.0])
    documents = np.array([[1.0, 0.0]])

    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity_scores(query, documents)


def test_incompatible_embedding_dimensions_are_rejected() -> None:
    query = np.array([1.0, 0.0])
    documents = np.array([[1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="same dimensions"):
        cosine_similarity_scores(query, documents)
