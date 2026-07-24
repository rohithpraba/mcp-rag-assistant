"""Deterministic tests for the local MCP interface."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp.exceptions import ToolError

from mcp_rag_assistant.mcp_server.server import (
    ServerSettings,
    create_server,
)
from mcp_rag_assistant.rag.answering.service import (
    GroundedAnswer,
    LabeledSource,
)
from mcp_rag_assistant.rag.ask_workspace import (
    WorkspaceAnswerResult,
)
from mcp_rag_assistant.rag.retrieval.service import (
    RetrievalResponse,
)
from mcp_rag_assistant.rag.search_workspace import (
    WorkspaceSearchResult,
)
from mcp_rag_assistant.rag.storage.chroma_store import (
    RetrievedChunk,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(coroutine: object) -> object:
    """Run one SDK coroutine without an async pytest dependency."""
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        chunk_id="chk_public",
        text="The retry limit is five.",
        distance=0.1,
        similarity=0.9,
        metadata={
            "source_id": "src_public",
            "source_name": "guide.md",
            "source_uri": "C:/private/guide.md",
            "chunk_index": 0,
            "chunk_count": 1,
        },
    )


def retrieval() -> RetrievalResponse:
    return RetrievalResponse(
        query="What is the retry limit?",
        requested_top_k=3,
        source_id=None,
        results=(retrieved_chunk(),),
    )


def search_result() -> WorkspaceSearchResult:
    return WorkspaceSearchResult(
        workspace_id="demo",
        stored_chunk_count=1,
        embedding_model="test-embedder",
        embedding_dimension=3,
        retrieval=retrieval(),
    )


def answer_result() -> WorkspaceAnswerResult:
    source = LabeledSource(
        label="S1",
        citation="[guide.md, chunk 1/1]",
        retrieval_rank=1,
        chunk_id="chk_public",
        text="The retry limit is five.",
        similarity=0.9,
        distance=0.1,
        metadata=retrieved_chunk().metadata,
        truncated=False,
    )
    answer = GroundedAnswer(
        question="What is the retry limit?",
        answer="The retry limit is five [S1].",
        insufficient_evidence=False,
        citation_status="valid",
        cited_labels=("S1",),
        unknown_citation_labels=(),
        context_character_count=len(source.text),
        sources=(source,),
        retrieval=retrieval(),
        generation=None,
    )

    return WorkspaceAnswerResult(
        workspace_id="demo",
        stored_chunk_count=1,
        embedding_model="test-embedder",
        embedding_dimension=3,
        grounded_answer=answer,
    )


def test_server_registers_phase3_capabilities() -> None:
    server = create_server()

    tools = run(server.list_tools())
    templates = run(server.list_resource_templates())
    prompts = run(server.list_prompts())

    assert [tool.name for tool in tools] == [
        "search_documents",
        "answer_question",
    ]
    assert [template.name for template in templates] == [
        "source_chunks"
    ]
    assert [prompt.name for prompt in prompts] == [
        "grounded_answer"
    ]


def test_search_documents_reuses_workspace_search() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_search(
        query: str,
        **kwargs: object,
    ) -> WorkspaceSearchResult:
        calls.append((query, kwargs))
        return search_result()

    server = create_server(search_backend=fake_search)
    tool_result = run(
        server.call_tool(
            "search_documents",
            {
                "query": "  What is the retry limit?  ",
                "workspace_id": "demo",
                "top_k": 3,
            },
        )
    )
    assert isinstance(tool_result, tuple)
    _, payload = tool_result

    assert isinstance(payload, dict)
    assert payload["result_count"] == 1
    assert payload["results"][0]["citation"] == (
        "[guide.md, chunk 1/1]"
    )
    assert "source_uri" not in payload["results"][0]["metadata"]
    assert calls[0][0] == "What is the retry limit?"
    assert calls[0][1]["workspace_id"] == "demo"


def test_answer_question_reuses_grounded_answering() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_answer(
        question: str,
        **kwargs: object,
    ) -> WorkspaceAnswerResult:
        calls.append((question, kwargs))
        return answer_result()

    server = create_server(answer_backend=fake_answer)
    tool_result = run(
        server.call_tool(
            "answer_question",
            {
                "question": "What is the retry limit?",
                "workspace_id": "demo",
            },
        )
    )
    assert isinstance(tool_result, tuple)
    _, payload = tool_result

    assert isinstance(payload, dict)
    assert payload["answer"] == "The retry limit is five [S1]."
    assert payload["citation_status"] == "valid"
    assert payload["cited_labels"] == ["S1"]
    assert calls[0][1]["temperature"] == 0.0


@pytest.mark.parametrize(
    "arguments, expected_message",
    [
        (
            {"query": " ", "workspace_id": "demo"},
            "query must not be empty",
        ),
        (
            {
                "query": "valid",
                "workspace_id": "demo",
                "top_k": 21,
            },
            "top_k must be between",
        ),
        (
            {"query": "valid", "workspace_id": "../private"},
            "workspace_id is invalid",
        ),
    ],
)
def test_search_validation_is_reported_safely(
    arguments: dict[str, object],
    expected_message: str,
) -> None:
    server = create_server()

    with pytest.raises(ToolError, match=expected_message):
        run(server.call_tool("search_documents", arguments))


def test_backend_errors_do_not_leak_internal_details() -> None:
    def failing_search(
        query: str,
        **kwargs: object,
    ) -> WorkspaceSearchResult:
        raise RuntimeError(
            "C:/private/index: connection token=do-not-leak"
        )

    server = create_server(search_backend=failing_search)

    with pytest.raises(
        ToolError,
        match="Document search is unavailable",
    ) as captured:
        run(
            server.call_tool(
                "search_documents",
                {"query": "valid", "workspace_id": "demo"},
            )
        )

    assert "private" not in str(captured.value)
    assert "token" not in str(captured.value)


def test_search_timeout_is_sanitized() -> None:
    def slow_search(
        query: str,
        **kwargs: object,
    ) -> WorkspaceSearchResult:
        time.sleep(0.05)
        return search_result()

    settings = ServerSettings(
        database_path=Path("unused"),
        search_timeout_seconds=0.001,
    )
    server = create_server(
        settings=settings,
        search_backend=slow_search,
    )

    with pytest.raises(
        ToolError,
        match="Document search timed out",
    ):
        run(
            server.call_tool(
                "search_documents",
                {"query": "valid", "workspace_id": "demo"},
            )
        )


def test_source_resource_returns_deterministic_json() -> None:
    def fake_source_info(
        **kwargs: object,
    ) -> dict[str, object]:
        assert kwargs["workspace_id"] == "demo"
        assert kwargs["source_id"] == "src_public"
        return {
            "workspace_id": "demo",
            "source_id": "src_public",
            "chunk_count": 2,
            "chunk_ids": ["chk_1", "chk_2"],
        }

    server = create_server(
        source_info_backend=fake_source_info
    )
    contents = run(
        server.read_resource(
            "rag://workspaces/demo/sources/src_public"
        )
    )
    items = list(contents)

    assert len(items) == 1
    assert json.loads(items[0].content) == {
        "workspace_id": "demo",
        "source_id": "src_public",
        "chunk_count": 2,
        "chunk_ids": ["chk_1", "chk_2"],
    }


def test_source_resource_sanitizes_backend_errors() -> None:
    def failing_source_info(
        **kwargs: object,
    ) -> dict[str, object]:
        raise RuntimeError("C:/private/index")

    server = create_server(
        source_info_backend=failing_source_info
    )

    with pytest.raises(
        ValueError,
        match="Source information is unavailable",
    ) as captured:
        run(
            server.read_resource(
                "rag://workspaces/demo/sources/src_public"
            )
        )

    assert "private" not in str(captured.value)


def test_grounded_prompt_reuses_grounding_rules() -> None:
    server = create_server()
    result = run(
        server.get_prompt(
            "grounded_answer",
            {
                "question": "What is supported?",
                "workspace_id": "demo",
            },
        )
    )
    text = result.messages[0].content.text

    assert "rely only on its grounded answer" in text
    assert "insufficient-evidence response exactly" in text
    assert "answer_question tool" in text
    assert "What is supported?" in text


def test_stdio_protocol_discovery_and_prompt() -> None:
    async def exercise_protocol() -> str:
        child_environment = os.environ.copy()
        child_environment["PYTHONPATH"] = str(
            PROJECT_ROOT / "src"
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "mcp_rag_assistant.mcp_server.server",
            ],
            cwd=PROJECT_ROOT,
            env=child_environment,
        )

        with tempfile.TemporaryFile(
            mode="w+",
            encoding="utf-8",
        ) as diagnostics:
            try:
                async with asyncio.timeout(30):
                    async with stdio_client(
                        parameters,
                        errlog=diagnostics,
                    ) as (read_stream, write_stream):
                        async with ClientSession(
                            read_stream,
                            write_stream,
                            read_timeout_seconds=timedelta(
                                seconds=20
                            ),
                        ) as session:
                            initialized = (
                                await session.initialize()
                            )
                            tools = await session.list_tools()
                            prompts = await session.list_prompts()
                            templates = (
                                await session
                                .list_resource_templates()
                            )
                            prompt = await session.get_prompt(
                                "grounded_answer",
                                {
                                    "question": (
                                        "What is supported?"
                                    ),
                                    "workspace_id": "demo",
                                },
                            )
            except* anyio.BrokenResourceError:
                # mcp 1.28.1 can race its Windows stdout reader
                # against receive-stream closure after clean shutdown.
                pass

            diagnostics.seek(0)
            captured_diagnostics = diagnostics.read()

        assert initialized.serverInfo.name == (
            "mcp-rag-assistant"
        ), captured_diagnostics
        assert {tool.name for tool in tools.tools} == {
            "search_documents",
            "answer_question",
        }, captured_diagnostics
        assert {item.name for item in prompts.prompts} == {
            "grounded_answer"
        }, captured_diagnostics
        assert {
            str(item.uriTemplate)
            for item in templates.resourceTemplates
        } == {
            (
                "rag://workspaces/{workspace_id}"
                "/sources/{source_id}"
            )
        }, captured_diagnostics
        assert prompt.messages, captured_diagnostics
        assert prompt.messages[0].content.type == "text"
        assert "What is supported?" in (
            prompt.messages[0].content.text
        )

        return captured_diagnostics

    diagnostics = run(exercise_protocol())

    assert isinstance(diagnostics, str)
    assert "Traceback" not in diagnostics
    assert "RuntimeWarning" not in diagnostics
