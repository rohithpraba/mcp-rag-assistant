"""Index or refresh one local text source in a Chroma workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking.word_window import chunk_document
from .embeddings.sentence_transformer import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
)
from .ingestion.local_file import load_local_text_file
from .storage.chroma_store import (
    ChromaVectorStore,
    SourceRefreshResult,
)


DEFAULT_DATABASE_PATH = Path("indexes/chroma")


def index_local_file(
    path: str | Path,
    *,
    workspace_id: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    chunk_size_words: int = 120,
    overlap_words: int = 20,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
) -> SourceRefreshResult:
    """Ingest, chunk, embed, and persist one local source."""
    document = load_local_text_file(
        path=path,
        workspace_id=workspace_id,
    )

    chunks = chunk_document(
        document,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )

    embedder = SentenceTransformerEmbedder(
        model_name=model_name,
        device=device,
    )

    embeddings = embedder.embed_documents(
        [
            chunk.text
            for chunk in chunks
        ]
    )

    store = ChromaVectorStore(
        database_path=database_path,
        workspace_id=workspace_id,
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.dimension,
    )

    return store.replace_source_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Index or refresh one local text or Markdown source "
            "inside a persistent Chroma workspace."
        )
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to a UTF-8 .txt, .md, or .markdown file.",
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
        "--chunk-size-words",
        type=int,
        default=120,
        help="Maximum words per chunk. Default: 120.",
    )

    parser.add_argument(
        "--overlap-words",
        type=int,
        default=20,
        help="Repeated words between chunks. Default: 20.",
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
    """Run local-file indexing from the command line."""
    arguments = parse_arguments()

    try:
        result = index_local_file(
            path=arguments.path,
            workspace_id=arguments.workspace,
            database_path=arguments.database_path,
            chunk_size_words=arguments.chunk_size_words,
            overlap_words=arguments.overlap_words,
            model_name=arguments.model_name,
            device=arguments.device,
        )
    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        raise SystemExit(
            f"indexing failed: {error}"
        ) from error

    print(
        json.dumps(
            {
                "source_id": result.source_id,
                "previous_chunk_count": (
                    result.previous_chunk_count
                ),
                "upserted_chunk_count": (
                    result.upserted_chunk_count
                ),
                "deleted_stale_chunk_count": (
                    result.deleted_stale_chunk_count
                ),
                "current_chunk_count": (
                    result.current_chunk_count
                ),
                "record_set_changed": (
                    result.record_set_changed
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
