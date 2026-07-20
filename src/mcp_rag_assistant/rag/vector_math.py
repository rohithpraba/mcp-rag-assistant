"""Low-level vector operations used by the retrieval pipeline."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cosine_similarity_scores(
    query_vector: ArrayLike,
    document_matrix: ArrayLike,
) -> NDArray[np.float32]:
    """Calculate cosine similarity between one query and many documents.

    Args:
        query_vector:
            One-dimensional query embedding with shape ``(dimensions,)``.
        document_matrix:
            Two-dimensional matrix of document embeddings with shape
            ``(number_of_documents, dimensions)``.

    Returns:
        One similarity score per document.

    Raises:
        ValueError:
            If the arrays have invalid shapes, incompatible dimensions,
            or contain zero-length vectors.
    """
    query = np.asarray(query_vector, dtype=np.float32)
    documents = np.asarray(document_matrix, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError("query_vector must be one-dimensional")

    if documents.ndim != 2:
        raise ValueError("document_matrix must be two-dimensional")

    if documents.shape[0] == 0:
        raise ValueError("document_matrix must contain at least one document")

    if documents.shape[1] != query.shape[0]:
        raise ValueError(
            "query and document embeddings must have the same dimensions"
        )

    query_norm = np.linalg.norm(query)
    document_norms = np.linalg.norm(documents, axis=1)

    if np.isclose(query_norm, 0.0):
        raise ValueError("query_vector must not be a zero vector")

    if np.any(np.isclose(document_norms, 0.0)):
        raise ValueError("document_matrix must not contain zero vectors")

    dot_products = documents @ query
    similarities = dot_products / (document_norms * query_norm)

    return similarities.astype(np.float32, copy=False)


def top_k_indices(
    scores: ArrayLike,
    k: int,
) -> NDArray[np.intp]:
    """Return indices of the highest scores in descending order."""
    values = np.asarray(scores, dtype=np.float32)

    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")

    if values.size == 0:
        raise ValueError("scores must not be empty")

    if k <= 0:
        raise ValueError("k must be greater than zero")

    result_count = min(k, values.size)

    # Negating the values lets argsort order the largest scores first.
    return np.argsort(-values, kind="stable")[:result_count]
