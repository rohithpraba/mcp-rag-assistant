"""Data models shared by the document-ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OriginType = Literal["local_file", "url"]


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A validated source after its text has been extracted.

    A source document represents one logical input, such as a local file
    or public URL. Chunking happens later and may produce many chunks from
    one source document.
    """

    workspace_id: str
    source_id: str
    origin_type: OriginType
    media_type: str
    source_name: str
    source_uri: str
    content_hash: str
    byte_size: int
    ingested_at_utc: str
    text: str

    def metadata(self) -> dict[str, str | int]:
        """Return metadata without including the complete extracted text."""
        return {
            "workspace_id": self.workspace_id,
            "source_id": self.source_id,
            "origin_type": self.origin_type,
            "media_type": self.media_type,
            "source_name": self.source_name,
            "source_uri": self.source_uri,
            "content_hash": self.content_hash,
            "byte_size": self.byte_size,
            "character_count": len(self.text),
            "ingested_at_utc": self.ingested_at_utc,
        }
