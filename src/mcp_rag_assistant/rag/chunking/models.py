"""Data models produced by the document-chunking pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from ..ingestion.models import OriginType


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One retrievable portion of an ingested source document.

    Word and character end positions are exclusive. For example,
    start_word=0 and end_word=10 describes the first ten words.

    PDF page numbers are one-based because they are shown to users.
    """

    workspace_id: str
    source_id: str
    chunk_id: str
    origin_type: OriginType
    media_type: str
    source_name: str
    source_uri: str
    content_hash: str
    source_byte_size: int
    ingested_at_utc: str

    chunking_strategy: str
    chunk_size_words: int
    overlap_words: int

    chunk_index: int
    chunk_count: int
    start_word: int
    end_word: int
    start_char: int
    end_char: int
    word_count: int

    text: str

    page_start: int | None = None
    page_end: int | None = None
    source_page_count: int | None = None

    def metadata(self) -> dict[str, str | int]:
        """Return scalar metadata suitable for vector storage."""
        metadata: dict[str, str | int] = {
            "workspace_id": self.workspace_id,
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "origin_type": self.origin_type,
            "media_type": self.media_type,
            "source_name": self.source_name,
            "source_uri": self.source_uri,
            "content_hash": self.content_hash,
            "source_byte_size": self.source_byte_size,
            "ingested_at_utc": self.ingested_at_utc,
            "chunking_strategy": self.chunking_strategy,
            "chunk_size_words": self.chunk_size_words,
            "overlap_words": self.overlap_words,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "start_word": self.start_word,
            "end_word": self.end_word,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "word_count": self.word_count,
            "character_count": len(self.text),
        }

        if self.page_start is not None:
            metadata["page_start"] = self.page_start

        if self.page_end is not None:
            metadata["page_end"] = self.page_end

        if self.source_page_count is not None:
            metadata["source_page_count"] = self.source_page_count

        return metadata
