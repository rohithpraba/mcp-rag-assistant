"""Page-aware extraction of text-based local PDF documents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from .local_file import (
    calculate_content_hash,
    create_local_source_id,
    normalize_text,
    validate_workspace_id,
)
from .models import SourceDocument


DEFAULT_MAX_PDF_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PdfPageSpan:
    """Character range occupied by one non-empty PDF page."""

    page_number: int
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class PdfSourceDocument:
    """An extracted PDF source plus page-position information."""

    source: SourceDocument
    page_count: int
    text_page_count: int
    page_spans: tuple[PdfPageSpan, ...]

    def metadata(self) -> dict[str, str | int]:
        """Return PDF and source-level metadata."""
        metadata = self.source.metadata()

        metadata.update(
            {
                "page_count": self.page_count,
                "text_page_count": self.text_page_count,
            }
        )

        return metadata


def load_local_pdf(
    path: str | Path,
    *,
    workspace_id: str = "default",
    max_bytes: int = DEFAULT_MAX_PDF_BYTES,
) -> PdfSourceDocument:
    """Validate and extract a text-based PDF.

    Image-only PDFs are rejected because OCR is not part of the
    initial core implementation.
    """
    workspace = validate_workspace_id(workspace_id)

    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")

    source_path = Path(path).expanduser()

    if not source_path.exists():
        raise FileNotFoundError(
            f"source file does not exist: {source_path}"
        )

    if not source_path.is_file():
        raise ValueError(
            f"source path is not a regular file: {source_path}"
        )

    resolved_path = source_path.resolve(strict=True)

    if resolved_path.suffix.lower() != ".pdf":
        raise ValueError("PDF source must use the .pdf extension")

    reported_size = resolved_path.stat().st_size

    if reported_size > max_bytes:
        raise ValueError(
            f"PDF exceeds the {max_bytes}-byte size limit"
        )

    raw_bytes = resolved_path.read_bytes()

    if len(raw_bytes) > max_bytes:
        raise ValueError(
            f"PDF exceeds the {max_bytes}-byte size limit"
        )

    try:
        pdf = pymupdf.open(
            stream=raw_bytes,
            filetype="pdf",
        )
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise ValueError(
            "source file is not a readable PDF"
        ) from error

    try:
        if pdf.needs_pass:
            raise ValueError(
                "password-protected PDFs are not supported"
            )

        page_count = int(pdf.page_count)

        if page_count <= 0:
            raise ValueError("PDF contains no pages")

        text_parts: list[str] = []
        hash_parts: list[str] = []
        page_spans: list[PdfPageSpan] = []
        character_cursor = 0

        for page_index in range(page_count):
            page = pdf.load_page(page_index)

            page_text = normalize_text(
                page.get_text(
                    "text",
                    sort=True,
                )
            )

            if not page_text:
                continue

            if text_parts:
                text_parts.append("\n\n")
                character_cursor += 2

            start_char = character_cursor

            text_parts.append(page_text)
            character_cursor += len(page_text)

            page_number = page_index + 1

            page_spans.append(
                PdfPageSpan(
                    page_number=page_number,
                    start_char=start_char,
                    end_char=character_cursor,
                )
            )

            hash_parts.append(
                f"PAGE:{page_number}\n{page_text}"
            )

        combined_text = "".join(text_parts)

        if not combined_text:
            raise ValueError(
                "PDF contains no extractable text; "
                "image-only or scanned PDFs require OCR"
            )

        # Include page structure so moving identical text between
        # pages still produces a new source content version.
        hash_input = (
            f"PAGE_COUNT:{page_count}\n"
            + "\n\f\n".join(hash_parts)
        )

        ingested_at = (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

        source = SourceDocument(
            workspace_id=workspace,
            source_id=create_local_source_id(
                workspace_id=workspace,
                resolved_path=resolved_path,
            ),
            origin_type="local_file",
            media_type="application/pdf",
            source_name=resolved_path.name,
            source_uri=resolved_path.as_uri(),
            content_hash=calculate_content_hash(
                hash_input
            ),
            byte_size=len(raw_bytes),
            ingested_at_utc=ingested_at,
            text=combined_text,
        )

        return PdfSourceDocument(
            source=source,
            page_count=page_count,
            text_page_count=len(page_spans),
            page_spans=tuple(page_spans),
        )
    finally:
        pdf.close()
