"""Model Context Protocol interface for the local RAG assistant."""

from typing import Any

__all__ = ["create_server"]


def create_server(*args: Any, **kwargs: Any) -> Any:
    """Create the server without importing it during package startup."""
    from .server import create_server as _create_server

    return _create_server(*args, **kwargs)
