"""Persistent Chroma storage for document chunks and embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from numpy.typing import ArrayLike

from ..chunking.models import TextChunk
from ..ingestion.local_file import validate_workspace_id


COLLECTION_SCHEMA_VERSION = 1
DISTANCE_SPACE = "cosine"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One ranked vector-search result."""

    rank: int
    chunk_id: str
    text: str
    distance: float
    similarity: float
    metadata: dict[str, object]


def workspace_collection_name(workspace_id: str) -> str:
    """Create a valid Chroma collection name for one workspace."""
    workspace = validate_workspace_id(workspace_id)
    return f"kb_{workspace}"


class ChromaVectorStore:
    """Store and retrieve chunk embeddings in one workspace collection."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        workspace_id: str,
        embedding_model: str,
        embedding_dimension: int,
    ) -> None:
        self.workspace_id = validate_workspace_id(workspace_id)

        cleaned_model_name = embedding_model.strip()

        if not cleaned_model_name:
            raise ValueError("embedding_model must not be empty")

        if embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be greater than zero"
            )

        self.embedding_model = cleaned_model_name
        self.embedding_dimension = int(embedding_dimension)

        self.database_path = Path(database_path)
        self.database_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.collection_name = workspace_collection_name(
            self.workspace_id
        )

        self._client = chromadb.PersistentClient(
            path=str(self.database_path),
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )

        self._collection = (
            self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=None,
                configuration={
                    "hnsw": {
                        "space": DISTANCE_SPACE,
                    }
                },
                metadata=self._expected_collection_metadata(),
            )
        )

        self._validate_existing_collection()

    def _expected_collection_metadata(
        self,
    ) -> dict[str, str | int]:
        """Return compatibility metadata for this workspace."""
        return {
            "workspace_id": self.workspace_id,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "distance_space": DISTANCE_SPACE,
        }

    def _validate_existing_collection(self) -> None:
        """Reject collections created with incompatible settings."""
        actual_metadata = self._collection.metadata or {}
        expected_metadata = (
            self._expected_collection_metadata()
        )

        for key, expected_value in expected_metadata.items():
            actual_value = actual_metadata.get(key)

            if actual_value != expected_value:
                raise ValueError(
                    "collection metadata mismatch for "
                    f"'{key}': expected {expected_value!r}, "
                    f"found {actual_value!r}"
                )

        try:
            actual_space = self._collection.configuration[
                "hnsw"
            ]["space"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "collection does not expose an HNSW distance space"
            ) from error

        if actual_space != DISTANCE_SPACE:
            raise ValueError(
                "collection distance space mismatch: "
                f"expected {DISTANCE_SPACE!r}, "
                f"found {actual_space!r}"
            )

    def count(self) -> int:
        """Return the number of stored chunk records."""
        return int(self._collection.count())

    def upsert_chunks(
        self,
        chunks: Sequence[TextChunk],
        embeddings: ArrayLike,
    ) -> int:
        """Insert or update chunks using deterministic chunk IDs."""
        chunk_list = list(chunks)

        if not chunk_list:
            raise ValueError(
                "chunks must contain at least one item"
            )

        chunk_ids = [
            chunk.chunk_id
            for chunk in chunk_list
        ]

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError(
                "chunk IDs must be unique within one upsert"
            )

        for chunk in chunk_list:
            if chunk.workspace_id != self.workspace_id:
                raise ValueError(
                    "chunk workspace does not match "
                    "the vector-store workspace"
                )

        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        expected_shape = (
            len(chunk_list),
            self.embedding_dimension,
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                "embedding matrix has shape "
                f"{matrix.shape}; expected {expected_shape}"
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                "embedding matrix contains non-finite values"
            )

        self._collection.upsert(
            ids=chunk_ids,
            embeddings=matrix.tolist(),
            documents=[
                chunk.text
                for chunk in chunk_list
            ],
            metadatas=[
                chunk.metadata()
                for chunk in chunk_list
            ],
        )

        return len(chunk_list)

    def search(
        self,
        query_embedding: ArrayLike,
        *,
        top_k: int = 3,
        source_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the nearest chunks for one query embedding."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        record_count = self.count()

        if record_count == 0:
            return []

        vector = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        expected_shape = (
            self.embedding_dimension,
        )

        if vector.shape != expected_shape:
            raise ValueError(
                "query embedding has shape "
                f"{vector.shape}; expected {expected_shape}"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "query embedding contains non-finite values"
            )

        if np.isclose(
            np.linalg.norm(vector),
            0.0,
        ):
            raise ValueError(
                "query embedding must not be a zero vector"
            )

        query_arguments: dict[str, object] = {
            "query_embeddings": [
                vector.tolist(),
            ],
            "n_results": min(
                top_k,
                record_count,
            ),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if source_id is not None:
            query_arguments["where"] = {
                "source_id": source_id,
            }

        result = self._collection.query(
            **query_arguments
        )

        ids = result["ids"][0]
        documents = (
            result["documents"][0]
            if result["documents"]
            else []
        )
        metadatas = (
            result["metadatas"][0]
            if result["metadatas"]
            else []
        )
        distances = (
            result["distances"][0]
            if result["distances"]
            else []
        )

        retrieved: list[RetrievedChunk] = []

        for rank, (
            chunk_id,
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                ids,
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):
            numeric_distance = float(distance)

            retrieved.append(
                RetrievedChunk(
                    rank=rank,
                    chunk_id=chunk_id,
                    text=document or "",
                    distance=numeric_distance,
                    similarity=1.0 - numeric_distance,
                    metadata=dict(metadata or {}),
                )
            )

        return retrieved
