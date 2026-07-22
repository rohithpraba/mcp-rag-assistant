"""Search an existing persistent knowledge workspace."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .embeddings.sentence_transformer import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
)
from .retrieval.service import (
    RetrievalResponse,
    format_chunk_citation,
    retrieve_chunks,
)
from .storage.chroma_store import ChromaVectorStore


DEFAULT_DATABASE_PATH = Path("indexes/chroma")


@dataclass(frozen=True, slots=True)
class WorkspaceSearchResult:
    """Summary and ranked results for one workspace search."""

    workspace_id: str
    stored_chunk_count: int
    embedding_model: str
    embedding_dimension: int
    retrieval: RetrievalResponse


def search_workspace(
    query: str,
    *,
    workspace_id: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    top_k: int = 3,
    source_id: str | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
) -> WorkspaceSearchResult:
    """Search an existing Chroma-backed knowledge workspace."""
    embedder = SentenceTransformerEmbedder(
        model_name=model_name,
        device=device,
    )

    store = ChromaVectorStore(
        database_path=database_path,
        workspace_id=workspace_id,
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.dimension,
    )

    retrieval = retrieve_chunks(
        query,
        embedder=embedder,
        store=store,
        top_k=top_k,
        source_id=source_id,
    )

    return WorkspaceSearchResult(
        workspace_id=workspace_id,
        stored_chunk_count=store.count(),
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.dimension,
        retrieval=retrieval,
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Search an existing persistent knowledge workspace "
            "using semantic vector retrieval."
        )
    )

    parser.add_argument(
        "query",
        nargs="+",
        help="Natural-language query.",
    )

    parser.add_argument(
        "--workspace",
        required=True,
        help="Knowledge workspace identifier.",
    )

    parser.add_argument(
        "--database-path",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=(
            "Local Chroma database directory. "
            "Default: indexes/chroma."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of results. Default: 3.",
    )

    parser.add_argument(
        "--source-id",
        default=None,
        help=(
            "Optional logical source ID used to restrict "
            "the retrieval search."
        ),
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence Transformer model identifier.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding device. Default: cpu.",
    )

    return parser.parse_args()


def main() -> None:
    """Run persistent workspace search from the command line."""
    arguments = parse_arguments()
    query = " ".join(arguments.query)

    try:
        result = search_workspace(
            query,
            workspace_id=arguments.workspace,
            database_path=arguments.database_path,
            top_k=arguments.top_k,
            source_id=arguments.source_id,
            model_name=arguments.model_name,
            device=arguments.device,
        )
    except (
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        raise SystemExit(
            f"search failed: {error}"
        ) from error

    summary = {
        "workspace_id": result.workspace_id,
        "stored_chunk_count": (
            result.stored_chunk_count
        ),
        "embedding_model": (
            result.embedding_model
        ),
        "embedding_dimension": (
            result.embedding_dimension
        ),
        "query": result.retrieval.query,
        "requested_top_k": (
            result.retrieval.requested_top_k
        ),
        "source_filter": (
            result.retrieval.source_id
        ),
        "result_count": len(
            result.retrieval.results
        ),
    }

    print("Search summary:")
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    if not result.retrieval.results:
        print(
            "\nNo indexed chunks were available "
            "for this search."
        )
        return

    for retrieved in result.retrieval.results:
        metadata = retrieved.metadata

        print(
            f"\n{retrieved.rank}. "
            f"similarity={retrieved.similarity:.4f} "
            f"distance={retrieved.distance:.4f}"
        )

        print(
            "   Citation:",
            format_chunk_citation(retrieved),
        )

        print(
            "   Chunk ID:",
            retrieved.chunk_id,
        )

        print(
            "   Source ID:",
            metadata.get(
                "source_id",
                "unknown",
            ),
        )

        print(
            "   Source URI:",
            metadata.get(
                "source_uri",
                "unknown",
            ),
        )

        print("   Text:")
        print(retrieved.text)


if __name__ == "__main__":
    main()
