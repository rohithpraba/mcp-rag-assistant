"""Tests for the Sentence Transformer embedding adapter."""

from __future__ import annotations

import numpy as np
import pytest

from mcp_rag_assistant.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbedder,
)


class FakeSentenceTransformer:
    """Small deterministic substitute for the real model."""

    def get_embedding_dimension(self) -> int:
        return 3

    def encode_document(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> np.ndarray:
        assert kwargs["normalize_embeddings"] is True

        return np.asarray(
            [
                [1.0, 0.0, 0.0]
                if "alpha" in text
                else [0.0, 1.0, 0.0]
                for text in texts
            ],
            dtype=np.float32,
        )

    def encode_query(
        self,
        text: str,
        **kwargs: object,
    ) -> np.ndarray:
        assert kwargs["normalize_embeddings"] is True

        if "alpha" in text:
            return np.asarray(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            )

        return np.asarray(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        )


class WrongDimensionModel(FakeSentenceTransformer):
    """Return an invalid query shape for validation testing."""

    def encode_query(
        self,
        text: str,
        **kwargs: object,
    ) -> np.ndarray:
        return np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        )


def test_document_and_query_embeddings_have_expected_shapes() -> None:
    embedder = SentenceTransformerEmbedder(
        model_name="fake-model",
        model=FakeSentenceTransformer(),
    )

    document_vectors = embedder.embed_documents(
        [
            "alpha document",
            "beta document",
        ]
    )

    query_vector = embedder.embed_query(
        "alpha question"
    )

    assert embedder.model_name == "fake-model"
    assert embedder.dimension == 3
    assert document_vectors.shape == (2, 3)
    assert query_vector.shape == (3,)
    assert document_vectors.dtype == np.float32
    assert query_vector.dtype == np.float32
    assert query_vector.tolist() == [
        1.0,
        0.0,
        0.0,
    ]


def test_empty_document_list_is_rejected() -> None:
    embedder = SentenceTransformerEmbedder(
        model_name="fake-model",
        model=FakeSentenceTransformer(),
    )

    with pytest.raises(
        ValueError,
        match="at least one document",
    ):
        embedder.embed_documents([])


def test_empty_query_is_rejected() -> None:
    embedder = SentenceTransformerEmbedder(
        model_name="fake-model",
        model=FakeSentenceTransformer(),
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        embedder.embed_query("   ")


def test_incorrect_embedding_dimension_is_rejected() -> None:
    embedder = SentenceTransformerEmbedder(
        model_name="fake-model",
        model=WrongDimensionModel(),
    )

    with pytest.raises(
        ValueError,
        match="expected",
    ):
        embedder.embed_query("alpha question")
