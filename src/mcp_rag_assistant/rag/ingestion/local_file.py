"""Ingest UTF-8 text and Markdown files from the local filesystem."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .models import SourceDocument


DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024

SUPPORTED_MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}

_WORKSPACE_ID_PATTERN = re.compile(
    r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$"
)


def validate_workspace_id(workspace_id: str) -> str:
    """Validate and return a workspace identifier.

    Workspace identifiers are deliberately restricted so they remain safe
    to use in database keys, command-line arguments, logs, and URLs.
    """
    candidate = workspace_id.strip()

    if not _WORKSPACE_ID_PATTERN.fullmatch(candidate):
        raise ValueError(
            "workspace_id must contain 1-64 lowercase letters, numbers, "
            "hyphens, or underscores; it must start and end with a letter "
            "or number"
        )

    return candidate


def normalize_text(text: str) -> str:
    """Normalize line endings and remove outer whitespace."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def calculate_content_hash(text: str) -> str:
    """Return the SHA-256 digest of normalized extracted text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity_uri(path: Path) -> str:
    """Return a platform-normalized file URI for identity calculation."""
    normalized_path = Path(os.path.normcase(str(path)))
    return normalized_path.as_uri()


def create_local_source_id(
    workspace_id: str,
    resolved_path: Path,
) -> str:
    """Create a stable identifier for a file inside one workspace.

    The identifier depends on the workspace and the file's canonical
    location, but not on its current contents. As a result, updating the
    file changes its content hash without changing its source identity.
    """
    identity_value = (
        f"{workspace_id}\0local_file\0{_identity_uri(resolved_path)}"
    )

    digest = hashlib.sha256(identity_value.encode("utf-8")).hexdigest()

    return f"src_{digest[:32]}"


def load_local_text_file(
    path: str | Path,
    workspace_id: str = "default",
    max_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> SourceDocument:
    """Validate and ingest a UTF-8 text or Markdown file.

    Args:
        path:
            Path to a local .txt, .md, or .markdown file.
        workspace_id:
            Identifier of the knowledge workspace receiving the source.
        max_bytes:
            Maximum accepted file size in bytes.

    Returns:
        A SourceDocument containing extracted text and source metadata.

    Raises:
        FileNotFoundError:
            When the supplied path does not exist.
        ValueError:
            When the path is not a supported, valid, non-empty UTF-8 file.
    """
    workspace = validate_workspace_id(workspace_id)

    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")

    source_path = Path(path).expanduser()

    if not source_path.exists():
        raise FileNotFoundError(f"source file does not exist: {source_path}")

    if not source_path.is_file():
        raise ValueError(f"source path is not a regular file: {source_path}")

    resolved_path = source_path.resolve(strict=True)
    suffix = resolved_path.suffix.lower()

    try:
        media_type = SUPPORTED_MEDIA_TYPES[suffix]
    except KeyError as error:
        supported = ", ".join(sorted(SUPPORTED_MEDIA_TYPES))
        raise ValueError(
            f"unsupported file type '{suffix}'; supported types: {supported}"
        ) from error

    reported_size = resolved_path.stat().st_size

    if reported_size > max_bytes:
        raise ValueError(
            f"source file exceeds the {max_bytes}-byte size limit"
        )

    raw_bytes = resolved_path.read_bytes()

    # Check again in case the file changed between stat() and read_bytes().
    if len(raw_bytes) > max_bytes:
        raise ValueError(
            f"source file exceeds the {max_bytes}-byte size limit"
        )

    try:
        decoded_text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            "source file must use UTF-8 text encoding"
        ) from error

    normalized_text = normalize_text(decoded_text)

    if not normalized_text:
        raise ValueError(
            "source file contains no non-whitespace text"
        )

    ingested_at = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    return SourceDocument(
        workspace_id=workspace,
        source_id=create_local_source_id(
            workspace_id=workspace,
            resolved_path=resolved_path,
        ),
        origin_type="local_file",
        media_type=media_type,
        source_name=resolved_path.name,
        source_uri=resolved_path.as_uri(),
        content_hash=calculate_content_hash(normalized_text),
        byte_size=len(raw_bytes),
        ingested_at_utc=ingested_at,
        text=normalized_text,
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate a local text or Markdown file and display its "
            "ingestion metadata."
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
        "--preview-chars",
        type=int,
        default=300,
        help="Number of extracted characters to preview. Default: 300.",
    )

    return parser.parse_args()


def main() -> None:
    """Run local-file ingestion from the command line."""
    arguments = parse_arguments()

    if arguments.preview_chars <= 0:
        raise SystemExit("--preview-chars must be greater than zero")

    try:
        document = load_local_text_file(
            path=arguments.path,
            workspace_id=arguments.workspace,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"ingestion failed: {error}") from error

    print(json.dumps(document.metadata(), indent=2, ensure_ascii=False))

    preview = document.text[: arguments.preview_chars]

    if len(document.text) > arguments.preview_chars:
        preview += "..."

    print("\nExtracted text preview:")
    print(preview)


if __name__ == "__main__":
    main()
