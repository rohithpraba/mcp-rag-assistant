"""Deterministic overlapping word-window document chunking."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..ingestion.local_file import load_local_text_file
from ..ingestion.models import SourceDocument
from .models import TextChunk


CHUNKING_STRATEGY = "word_window_v1"

_WORD_PATTERN = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class _ChunkSpan:
    """Internal word and character boundaries for one chunk."""

    start_word: int
    end_word: int
    start_char: int
    end_char: int


def count_words(text: str) -> int:
    """Count non-whitespace word units in text."""
    return sum(1 for _ in _WORD_PATTERN.finditer(text))


def _validate_chunking_parameters(
    chunk_size_words: int,
    overlap_words: int,
) -> None:
    """Validate the word-window configuration."""
    if chunk_size_words <= 0:
        raise ValueError(
            "chunk_size_words must be greater than zero"
        )

    if overlap_words < 0:
        raise ValueError(
            "overlap_words must be zero or greater"
        )

    if overlap_words >= chunk_size_words:
        raise ValueError(
            "overlap_words must be smaller than chunk_size_words"
        )


def _build_chunk_spans(
    text: str,
    chunk_size_words: int,
    overlap_words: int,
) -> list[_ChunkSpan]:
    """Calculate deterministic overlapping word and character spans."""
    word_matches = list(_WORD_PATTERN.finditer(text))

    if not word_matches:
        raise ValueError(
            "document text must contain at least one word"
        )

    step_size = chunk_size_words - overlap_words
    spans: list[_ChunkSpan] = []
    start_word = 0

    while start_word < len(word_matches):
        end_word = min(
            start_word + chunk_size_words,
            len(word_matches),
        )

        start_char = word_matches[start_word].start()
        end_char = word_matches[end_word - 1].end()

        spans.append(
            _ChunkSpan(
                start_word=start_word,
                end_word=end_word,
                start_char=start_char,
                end_char=end_char,
            )
        )

        if end_word == len(word_matches):
            break

        start_word += step_size

    return spans


def _create_chunk_id(
    document: SourceDocument,
    chunk_size_words: int,
    overlap_words: int,
    chunk_index: int,
    span: _ChunkSpan,
) -> str:
    """Create a deterministic identifier for one chunk version."""
    identity_parts = (
        document.source_id,
        document.content_hash,
        CHUNKING_STRATEGY,
        str(chunk_size_words),
        str(overlap_words),
        str(chunk_index),
        str(span.start_word),
        str(span.end_word),
    )

    identity_value = "\0".join(identity_parts)

    digest = hashlib.sha256(
        identity_value.encode("utf-8")
    ).hexdigest()

    return f"chk_{digest[:32]}"


def chunk_document(
    document: SourceDocument,
    *,
    chunk_size_words: int = 120,
    overlap_words: int = 20,
) -> list[TextChunk]:
    """Divide one source document into overlapping word windows.

    Args:
        document:
            Validated and normalized source document.
        chunk_size_words:
            Maximum number of word units in each chunk.
        overlap_words:
            Number of word units repeated between adjacent chunks.

    Returns:
        Ordered chunks preserving source provenance and chunk positions.
    """
    _validate_chunking_parameters(
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )

    spans = _build_chunk_spans(
        text=document.text,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )

    chunk_count = len(spans)
    chunks: list[TextChunk] = []

    for chunk_index, span in enumerate(spans):
        chunk_text = document.text[
            span.start_char : span.end_char
        ]

        chunks.append(
            TextChunk(
                workspace_id=document.workspace_id,
                source_id=document.source_id,
                chunk_id=_create_chunk_id(
                    document=document,
                    chunk_size_words=chunk_size_words,
                    overlap_words=overlap_words,
                    chunk_index=chunk_index,
                    span=span,
                ),
                origin_type=document.origin_type,
                media_type=document.media_type,
                source_name=document.source_name,
                source_uri=document.source_uri,
                content_hash=document.content_hash,
                source_byte_size=document.byte_size,
                ingested_at_utc=document.ingested_at_utc,
                chunking_strategy=CHUNKING_STRATEGY,
                chunk_size_words=chunk_size_words,
                overlap_words=overlap_words,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                start_word=span.start_word,
                end_word=span.end_word,
                start_char=span.start_char,
                end_char=span.end_char,
                word_count=span.end_word - span.start_word,
                text=chunk_text,
            )
        )

    return chunks


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Ingest a local text source and divide it into "
            "overlapping word-window chunks."
        )
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to a UTF-8 .txt, .md, or .markdown file.",
    )

    parser.add_argument(
        "--workspace",
        default="default",
        help="Knowledge workspace identifier. Default: default.",
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
        help="Repeated words between adjacent chunks. Default: 20.",
    )

    parser.add_argument(
        "--preview-chars",
        type=int,
        default=180,
        help="Maximum preview characters per chunk. Default: 180.",
    )

    return parser.parse_args()


def main() -> None:
    """Run local ingestion followed by word-window chunking."""
    arguments = parse_arguments()

    if arguments.preview_chars <= 0:
        raise SystemExit(
            "--preview-chars must be greater than zero"
        )

    try:
        document = load_local_text_file(
            path=arguments.path,
            workspace_id=arguments.workspace,
        )

        chunks = chunk_document(
            document,
            chunk_size_words=arguments.chunk_size_words,
            overlap_words=arguments.overlap_words,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"chunking failed: {error}") from error

    summary = {
        "workspace_id": document.workspace_id,
        "source_id": document.source_id,
        "source_name": document.source_name,
        "source_word_count": count_words(document.text),
        "chunking_strategy": CHUNKING_STRATEGY,
        "chunk_size_words": arguments.chunk_size_words,
        "overlap_words": arguments.overlap_words,
        "chunk_count": len(chunks),
    }

    print("Chunking summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    for chunk in chunks:
        preview = chunk.text[: arguments.preview_chars]

        if len(chunk.text) > arguments.preview_chars:
            preview += "..."

        print(
            f"\nChunk {chunk.chunk_index + 1}/"
            f"{chunk.chunk_count}"
        )
        print(
            json.dumps(
                chunk.metadata(),
                indent=2,
                ensure_ascii=False,
            )
        )
        print("Text preview:")
        print(preview)


if __name__ == "__main__":
    main()
