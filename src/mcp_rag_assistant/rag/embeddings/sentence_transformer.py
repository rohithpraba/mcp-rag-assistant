"""Sentence Transformer adapter for document and query embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


class SentenceTransformerEmbedder:
    """Generate normalized document and query embeddings.

    The optional ``model`` argument exists primarily for unit testing.
    Production code normally loads the model from ``model_name``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        model: Any | None = None,
    ) -> None:
        cleaned_model_name = model_name.strip()

        if not cleaned_model_name:
            raise ValueError("model_name must not be empty")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self._model_name = cleaned_model_name
        self._batch_size = batch_size

        self._model = (
            model
            if model is not None
            else SentenceTransformer(
                cleaned_model_name,
                device=device,
            )
        )

        dimension = self._model.get_sentence_embedding_dimension()

        if dimension is None or int(dimension) <= 0:
            raise ValueError(
                "embedding model must report a positive dimension"
            )

        self._dimension = int(dimension)

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the number of values in each embedding."""
        return self._dimension

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        """Generate one normalized embedding per document passage."""
        if isinstance(texts, str):
            raise TypeError(
                "texts must be a sequence of strings, not one string"
            )

        items = list(texts)

        if not items:
            raise ValueError("texts must contain at least one document")

        if any(not isinstance(text, str) for text in items):
            raise TypeError("every document must be a string")

        cleaned_items = [text.strip() for text in items]

        if any(not text for text in cleaned_items):
            raise ValueError("documents must not be empty")

        raw_embeddings = self._model.encode_document(
            cleaned_items,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        matrix = np.asarray(
            raw_embeddings,
            dtype=np.float32,
        )

        expected_shape = (
            len(cleaned_items),
            self.dimension,
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                "document embeddings have shape "
                f"{matrix.shape}; expected {expected_shape}"
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                "document embeddings contain non-finite values"
            )

        return matrix

    def embed_query(
        self,
        text: str,
    ) -> NDArray[np.float32]:
        """Generate one normalized query embedding."""
        if not isinstance(text, str):
            raise TypeError("query text must be a string")

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError("query text must not be empty")

        raw_embedding = self._model.encode_query(
            cleaned_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        vector = np.asarray(
            raw_embedding,
            dtype=np.float32,
        )

        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]

        expected_shape = (self.dimension,)

        if vector.shape != expected_shape:
            raise ValueError(
                "query embedding has shape "
                f"{vector.shape}; expected {expected_shape}"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "query embedding contains non-finite values"
            )

        return vector
