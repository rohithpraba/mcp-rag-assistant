"""Index or refresh one public HTML or PDF URL."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .chunking.pdf_pages import chunk_pdf_document
from .chunking.word_window import chunk_document
from .embeddings.sentence_transformer import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
)
from .ingestion.models import SourceDocument
from .ingestion.pdf_file import PdfSourceDocument
from .ingestion.public_url import (
    DEFAULT_MAX_DOWNLOAD_BYTES,
    DEFAULT_MAX_REDIRECTS,
    DEFAULT_TIMEOUT_SECONDS,
    UrlFetchError,
    load_public_url,
)
from .storage.chroma_store import (
    ChromaVectorStore,
    SourceRefreshResult,
)


DEFAULT_DATABASE_PATH = Path("indexes/chroma")


@dataclass(frozen=True, slots=True)
class UrlIndexResult:
    """Source metadata and refresh result after URL indexing."""

    source: SourceDocument
    page_count: int | None
    text_page_count: int | None
    refresh: SourceRefreshResult


def index_url(
    url: str,
    *,
    workspace_id: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    chunk_size_words: int = 160,
    overlap_words: int = 30,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
) -> UrlIndexResult:
    """Fetch, extract, chunk, embed, and index one public URL."""
    loaded = load_public_url(
        url,
        workspace_id=workspace_id,
        max_bytes=max_download_bytes,
        timeout_seconds=timeout_seconds,
        max_redirects=max_redirects,
    )

    if isinstance(loaded, PdfSourceDocument):
        source = loaded.source

        chunks = chunk_pdf_document(
            loaded,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )

        page_count: int | None = loaded.page_count
        text_page_count: int | None = (
            loaded.text_page_count
        )
    else:
        source = loaded

        chunks = chunk_document(
            source,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )

        page_count = None
        text_page_count = None

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

    return UrlIndexResult(
        source=source,
        page_count=page_count,
        text_page_count=text_page_count,
        refresh=refresh,
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Index or refresh one public static HTML page "
            "or direct PDF URL."
        )
    )

    parser.add_argument(
        "url",
        help="Public HTTP or HTTPS URL.",
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
        default=160,
        help="Maximum words per chunk. Default: 160.",
    )

    parser.add_argument(
        "--overlap-words",
        type=int,
        default=30,
        help="Repeated words between chunks. Default: 30.",
    )

    parser.add_argument(
        "--max-download-bytes",
        type=int,
        default=DEFAULT_MAX_DOWNLOAD_BYTES,
        help="Maximum downloaded response size.",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Network timeout. Default: 20.",
    )

    parser.add_argument(
        "--max-redirects",
        type=int,
        default=DEFAULT_MAX_REDIRECTS,
        help="Maximum validated redirects. Default: 3.",
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
    """Run URL indexing from the command line."""
    arguments = parse_arguments()

    try:
        result = index_url(
            arguments.url,
            workspace_id=arguments.workspace,
            database_path=arguments.database_path,
            chunk_size_words=arguments.chunk_size_words,
            overlap_words=arguments.overlap_words,
            model_name=arguments.model_name,
            device=arguments.device,
            max_download_bytes=(
                arguments.max_download_bytes
            ),
            timeout_seconds=arguments.timeout_seconds,
            max_redirects=arguments.max_redirects,
        )
    except (
        TypeError,
        ValueError,
        RuntimeError,
        UrlFetchError,
    ) as error:
        raise SystemExit(
            f"URL indexing failed: {error}"
        ) from error

    source = result.source
    refresh = result.refresh

    print(
        json.dumps(
            {
                "source_id": source.source_id,
                "source_name": source.source_name,
                "media_type": source.media_type,
                "requested_uri": source.requested_uri,
                "final_uri": source.source_uri,
                "redirect_count": source.redirect_count,
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
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
