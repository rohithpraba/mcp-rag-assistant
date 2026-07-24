"""Local stdio MCP server for retrieval and grounded answering."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ResourceError, ToolError

from ..rag.ingestion.local_file import validate_workspace_id

if TYPE_CHECKING:
    from ..rag.ask_workspace import WorkspaceAnswerResult
    from ..rag.search_workspace import WorkspaceSearchResult


DEFAULT_DATABASE_PATH = Path("indexes/chroma")
DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_MAX_CONTEXT_CHARACTERS = 6000
DEFAULT_SEARCH_TIMEOUT_SECONDS = 30.0
DEFAULT_ANSWER_TIMEOUT_SECONDS = 300.0
MAX_QUERY_CHARACTERS = 4000
MAX_SOURCE_ID_CHARACTERS = 256
MAX_TOP_K = 20

T = TypeVar("T")

SearchBackend = Callable[..., "WorkspaceSearchResult"]
AnswerBackend = Callable[..., "WorkspaceAnswerResult"]
SourceInfoBackend = Callable[..., dict[str, object]]


@dataclass(frozen=True, slots=True)
class ServerSettings:
    """Runtime settings for the local MCP adapter."""

    database_path: Path = DEFAULT_DATABASE_PATH
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str = "cpu"
    ollama_model: str = "gemma3:latest"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: float = 240.0
    search_timeout_seconds: float = DEFAULT_SEARCH_TIMEOUT_SECONDS
    answer_timeout_seconds: float = DEFAULT_ANSWER_TIMEOUT_SECONDS
    max_context_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS
    max_output_tokens: int = 300


def _clean_text(
    value: str,
    *,
    field_name: str,
    maximum_characters: int,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    cleaned = value.strip()

    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")

    if len(cleaned) > maximum_characters:
        raise ValueError(
            f"{field_name} must not exceed "
            f"{maximum_characters} characters"
        )

    return cleaned


def _clean_workspace_id(value: str) -> str:
    try:
        return validate_workspace_id(value)
    except (TypeError, ValueError) as error:
        raise ValueError("workspace_id is invalid") from error


def _clean_source_id(value: str | None) -> str | None:
    if value is None:
        return None

    return _clean_text(
        value,
        field_name="source_id",
        maximum_characters=MAX_SOURCE_ID_CHARACTERS,
    )


def _clean_top_k(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("top_k must be an integer")

    if not 1 <= value <= MAX_TOP_K:
        raise ValueError(
            f"top_k must be between 1 and {MAX_TOP_K}"
        )

    return value


async def _run_with_timeout(
    operation: Callable[[], T],
    *,
    timeout_seconds: float,
) -> T:
    return await asyncio.wait_for(
        asyncio.to_thread(operation),
        timeout=timeout_seconds,
    )


def _public_metadata(
    metadata: dict[str, object],
) -> dict[str, object]:
    allowed_fields = (
        "source_id",
        "source_name",
        "source_type",
        "page_start",
        "page_end",
        "chunk_index",
        "chunk_count",
        "content_hash",
    )

    return {
        field: metadata[field]
        for field in allowed_fields
        if field in metadata
    }


def _search_payload(
    result: "WorkspaceSearchResult",
) -> dict[str, object]:
    from ..rag.retrieval.service import format_chunk_citation

    return {
        "workspace_id": result.workspace_id,
        "query": result.retrieval.query,
        "source_filter": result.retrieval.source_id,
        "requested_top_k": result.retrieval.requested_top_k,
        "result_count": len(result.retrieval.results),
        "stored_chunk_count": result.stored_chunk_count,
        "results": [
            {
                "rank": chunk.rank,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "similarity": chunk.similarity,
                "distance": chunk.distance,
                "citation": format_chunk_citation(chunk),
                "metadata": _public_metadata(chunk.metadata),
            }
            for chunk in result.retrieval.results
        ],
    }


def _answer_payload(
    result: "WorkspaceAnswerResult",
) -> dict[str, object]:
    answer = result.grounded_answer

    return {
        "workspace_id": result.workspace_id,
        "question": answer.question,
        "answer": answer.answer,
        "insufficient_evidence": answer.insufficient_evidence,
        "citation_status": answer.citation_status,
        "cited_labels": list(answer.cited_labels),
        "unknown_citation_labels": list(
            answer.unknown_citation_labels
        ),
        "context_character_count": (
            answer.context_character_count
        ),
        "stored_chunk_count": result.stored_chunk_count,
        "sources": [
            {
                "label": source.label,
                "citation": source.citation,
                "retrieval_rank": source.retrieval_rank,
                "chunk_id": source.chunk_id,
                "text": source.text,
                "similarity": source.similarity,
                "distance": source.distance,
                "truncated": source.truncated,
                "metadata": _public_metadata(source.metadata),
            }
            for source in answer.sources
        ],
    }


def _default_source_info(
    *,
    workspace_id: str,
    source_id: str,
    settings: ServerSettings,
) -> dict[str, object]:
    from ..rag.embeddings.sentence_transformer import (
        SentenceTransformerEmbedder,
    )
    from ..rag.storage.chroma_store import ChromaVectorStore

    embedder = SentenceTransformerEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    store = ChromaVectorStore(
        database_path=settings.database_path,
        workspace_id=workspace_id,
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.dimension,
    )
    chunk_ids = store.get_source_chunk_ids(source_id)

    return {
        "workspace_id": workspace_id,
        "source_id": source_id,
        "chunk_count": len(chunk_ids),
        "chunk_ids": chunk_ids,
    }


def _default_search_backend(
    query: str,
    **kwargs: object,
) -> "WorkspaceSearchResult":
    from ..rag.search_workspace import search_workspace

    return search_workspace(query, **kwargs)


def _default_answer_backend(
    question: str,
    **kwargs: object,
) -> "WorkspaceAnswerResult":
    from ..rag.ask_workspace import answer_workspace

    return answer_workspace(question, **kwargs)


def create_server(
    *,
    settings: ServerSettings | None = None,
    search_backend: SearchBackend = _default_search_backend,
    answer_backend: AnswerBackend = _default_answer_backend,
    source_info_backend: SourceInfoBackend | None = None,
) -> FastMCP:
    """Create an MCP server with injectable deterministic backends."""
    active_settings = settings or ServerSettings()
    active_source_info = (
        source_info_backend or _default_source_info
    )

    server = FastMCP(
        name="mcp-rag-assistant",
        instructions=(
            "Search indexed local workspaces and answer questions "
            "using only retrieved evidence."
        ),
    )

    @server.tool(
        name="search_documents",
        description=(
            "Search semantically similar chunks in an existing "
            "local knowledge workspace."
        ),
        structured_output=True,
    )
    async def search_documents(
        query: str,
        workspace_id: str,
        top_k: int = 3,
        source_id: str | None = None,
    ) -> dict[str, object]:
        try:
            cleaned_query = _clean_text(
                query,
                field_name="query",
                maximum_characters=MAX_QUERY_CHARACTERS,
            )
            cleaned_workspace = _clean_workspace_id(
                workspace_id
            )
            cleaned_top_k = _clean_top_k(top_k)
            cleaned_source = _clean_source_id(source_id)
        except ValueError as error:
            raise ToolError(
                f"Invalid search request: {error}"
            ) from None

        try:
            result = await _run_with_timeout(
                lambda: search_backend(
                    cleaned_query,
                    workspace_id=cleaned_workspace,
                    database_path=active_settings.database_path,
                    top_k=cleaned_top_k,
                    source_id=cleaned_source,
                    model_name=active_settings.embedding_model,
                    device=active_settings.embedding_device,
                ),
                timeout_seconds=(
                    active_settings.search_timeout_seconds
                ),
            )
        except TimeoutError:
            raise ToolError(
                "Document search timed out"
            ) from None
        except Exception:
            raise ToolError(
                "Document search is unavailable"
            ) from None

        return _search_payload(result)

    @server.tool(
        name="answer_question",
        description=(
            "Answer a question using retrieved workspace evidence "
            "and return the supporting source chunks."
        ),
        structured_output=True,
    )
    async def answer_question_tool(
        question: str,
        workspace_id: str,
        top_k: int = 3,
        source_id: str | None = None,
    ) -> dict[str, object]:
        try:
            cleaned_question = _clean_text(
                question,
                field_name="question",
                maximum_characters=MAX_QUERY_CHARACTERS,
            )
            cleaned_workspace = _clean_workspace_id(
                workspace_id
            )
            cleaned_top_k = _clean_top_k(top_k)
            cleaned_source = _clean_source_id(source_id)
        except ValueError as error:
            raise ToolError(
                f"Invalid answer request: {error}"
            ) from None

        try:
            result = await _run_with_timeout(
                lambda: answer_backend(
                    cleaned_question,
                    workspace_id=cleaned_workspace,
                    database_path=active_settings.database_path,
                    top_k=cleaned_top_k,
                    source_id=cleaned_source,
                    embedding_model=(
                        active_settings.embedding_model
                    ),
                    embedding_device=(
                        active_settings.embedding_device
                    ),
                    ollama_model=active_settings.ollama_model,
                    ollama_base_url=(
                        active_settings.ollama_base_url
                    ),
                    timeout_seconds=(
                        active_settings.ollama_timeout_seconds
                    ),
                    max_context_characters=(
                        active_settings.max_context_characters
                    ),
                    temperature=0.0,
                    max_output_tokens=(
                        active_settings.max_output_tokens
                    ),
                ),
                timeout_seconds=(
                    active_settings.answer_timeout_seconds
                ),
            )
        except TimeoutError:
            raise ToolError(
                "Grounded answering timed out"
            ) from None
        except Exception:
            raise ToolError(
                "Grounded answering is unavailable"
            ) from None

        return _answer_payload(result)

    @server.resource(
        "rag://workspaces/{workspace_id}/sources/{source_id}",
        name="source_chunks",
        description=(
            "Return chunk identifiers stored for one source in a "
            "local knowledge workspace."
        ),
        mime_type="application/json",
    )
    async def source_chunks(
        workspace_id: str,
        source_id: str,
    ) -> str:
        try:
            cleaned_workspace = _clean_workspace_id(
                workspace_id
            )
            cleaned_source = _clean_source_id(source_id)
            assert cleaned_source is not None
        except ValueError as error:
            raise ResourceError(
                f"Invalid source resource request: {error}"
            ) from None

        try:
            payload = await _run_with_timeout(
                lambda: active_source_info(
                    workspace_id=cleaned_workspace,
                    source_id=cleaned_source,
                    settings=active_settings,
                ),
                timeout_seconds=(
                    active_settings.search_timeout_seconds
                ),
            )
        except TimeoutError:
            raise ResourceError(
                "Source information request timed out"
            ) from None
        except Exception:
            raise ResourceError(
                "Source information is unavailable"
            ) from None

        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )

    @server.prompt(
        name="grounded_answer",
        description=(
            "Prepare a reusable prompt for answering exclusively "
            "from retrieved workspace evidence."
        ),
    )
    def grounded_answer_prompt(
        question: str,
        workspace_id: str,
        source_id: str = "",
    ) -> str:
        cleaned_question = _clean_text(
            question,
            field_name="question",
            maximum_characters=MAX_QUERY_CHARACTERS,
        )
        cleaned_workspace = _clean_workspace_id(workspace_id)
        cleaned_source = (
            _clean_source_id(source_id)
            if source_id
            else None
        )
        source_instruction = (
            f" Restrict retrieval to source_id "
            f"{cleaned_source!r}."
            if cleaned_source
            else ""
        )

        return (
            "Use the answer_question tool and rely only on its "
            "grounded answer and returned sources. Preserve its "
            "citations and insufficient-evidence response exactly.\n"
            f"Use workspace_id "
            f"{cleaned_workspace!r}.{source_instruction}\n"
            f"Question: {cleaned_question}"
        )

    return server


mcp = create_server()


def main() -> None:
    """Run the server over local standard input/output."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
