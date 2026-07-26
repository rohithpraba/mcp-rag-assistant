"""Deterministically refresh the fixed demo workspace."""

from __future__ import annotations

import os
from pathlib import Path

from ..rag.index_local_file import index_local_file

WORKSPACE = "public-demo"
SOURCES = (
    Path("README.md"), Path("docs/ARCHITECTURE.md"),
    Path("docs/EVALUATION.md"), Path("docs/DECISIONS.md"),
)


def main() -> None:
    database = Path(os.getenv("DEMO_DATABASE_PATH", "indexes/chroma"))
    for source in SOURCES:
        result = index_local_file(source, workspace_id=WORKSPACE, database_path=database)
        print(f"{source.name}: chunks={result.current_chunk_count} changed={result.record_set_changed}")


if __name__ == "__main__":
    main()
