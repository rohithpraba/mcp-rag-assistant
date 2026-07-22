"""Index or refresh one local PDF in a Chroma workspace."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .chunking.pdf_pages import chunk_pdf_document
from .embeddings.sentence_transformer import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
)
from .ingestion.pdf_file import load_local_pdf
from .storage.chroma_store import (
    ChromaVectorStore,
    SourceRefreshResult,
)


DEFAULT_DATABASE_PATH = Path("indexes/chroma")


@dataclass(frozen=True, slots=True)
class PdfIndexResult:
    """Source and refresh information after PDF indexing."""

    page_count: int
    text_page_count: int
    refresh: SourceRefreshResult


def index_pdf_file(
    path: str | Path,
    *,
    workspace_id: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    chunk_size_words: int = 120,
    overlap_words: int = 20,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
) -> PdfIndexResult:
    """Extract, chunk, embed, and persist one PDF source."""
    document = load_local_pdf(
        path,
        workspace_id=workspace_id,
    )

    chunks = chunk_pdf_document(
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

    refresh = store.replace_source_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    return PdfIndexResult(
        page_count=document.page_count,
        text_page_count=document.text_page_count,
        refresh=refresh,
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Index or refresh a text-based local PDF "
            "inside a persistent Chroma workspace."
        )
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to a local text-based PDF.",
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
        help="Chroma database path. Default: indexes/chroma.",
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
    """Run PDF indexing from the command line."""
    arguments = parse_arguments()

    try:
        result = index_pdf_file(
            arguments.path,
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
            f"PDF indexing failed: {error}"
        ) from error

    refresh = result.refresh

    print(
        json.dumps(
            {
                "source_id": refresh.source_id,
                "page_count": result.page_count,
                "text_page_count": result.text_page_count,
                "previous_chunk_count": (
                    refresh.previous_chunk_count
                ),
                "upserted_chunk_count": (
                    refresh.upserted_chunk_count
                ),
                "deleted_stale_chunk_count": (
                    refresh.deleted_stale_chunk_count
                ),
                "current_chunk_count": (
                    refresh.current_chunk_count
                ),
                "record_set_changed": (
                    refresh.record_set_changed
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
