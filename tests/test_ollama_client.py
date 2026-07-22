"""Tests for the reusable Ollama chat client."""

from __future__ import annotations

import pytest

from mcp_rag_assistant.rag.generation.ollama_client import (
    OllamaChatClient,
    OllamaResponseError,
)


class RecordingTransport:
    """Return a fixed response while recording the request."""

    def __init__(
        self,
        response: dict[str, object],
    ) -> None:
        self.response = response
        self.calls: list[
            tuple[
                str,
                dict[str, object],
                float,
            ]
        ] = []

    def __call__(
        self,
        url: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        self.calls.append(
            (
                url,
                payload,
                timeout_seconds,
            )
        )

        return self.response


def successful_response() -> dict[str, object]:
    """Return a realistic non-streaming Ollama response."""
    return {
        "model": "gemma3:latest",
        "message": {
            "role": "assistant",
            "content": "Grounded answer [S1].",
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": 2_500_000_000,
        "load_duration": 500_000_000,
        "prompt_eval_count": 40,
        "eval_count": 8,
    }


def test_chat_builds_request_and_parses_response() -> None:
    transport = RecordingTransport(
        successful_response()
    )

    client = OllamaChatClient(
        model="gemma3:latest",
        base_url="http://localhost:11434/",
        timeout_seconds=30,
        keep_alive="5m",
        transport=transport,
    )

    result = client.chat(
        system_prompt="Use only supplied evidence.",
        user_prompt="Question and context.",
        temperature=0,
        max_output_tokens=120,
    )

    assert len(transport.calls) == 1

    url, payload, timeout = transport.calls[0]

    assert url == "http://localhost:11434/api/chat"
    assert timeout == 30.0

    assert payload["model"] == "gemma3:latest"
    assert payload["stream"] is False
    assert payload["keep_alive"] == "5m"

    assert payload["messages"] == [
        {
            "role": "system",
            "content": "Use only supplied evidence.",
        },
        {
            "role": "user",
            "content": "Question and context.",
        },
    ]

    assert payload["options"] == {
        "temperature": 0.0,
        "num_predict": 120,
    }

    assert result.model == "gemma3:latest"
    assert result.content == "Grounded answer [S1]."
    assert result.done is True
    assert result.done_reason == "stop"
    assert result.total_duration_seconds == 2.5
    assert result.load_duration_seconds == 0.5
    assert result.prompt_tokens == 40
    assert result.output_tokens == 8


@pytest.mark.parametrize(
    ("system_prompt", "user_prompt"),
    [
        ("   ", "Valid user prompt"),
        ("Valid system prompt", "   "),
    ],
)
def test_blank_prompts_are_rejected(
    system_prompt: str,
    user_prompt: str,
) -> None:
    client = OllamaChatClient(
        transport=RecordingTransport(
            successful_response()
        )
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )


def test_nonpositive_output_limit_is_rejected() -> None:
    client = OllamaChatClient(
        transport=RecordingTransport(
            successful_response()
        )
    )

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        client.chat(
            system_prompt="System",
            user_prompt="Question",
            max_output_tokens=0,
        )


def test_negative_temperature_is_rejected() -> None:
    client = OllamaChatClient(
        transport=RecordingTransport(
            successful_response()
        )
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        client.chat(
            system_prompt="System",
            user_prompt="Question",
            temperature=-0.1,
        )


def test_api_error_is_rejected() -> None:
    client = OllamaChatClient(
        transport=RecordingTransport(
            {
                "error": "model not found",
            }
        )
    )

    with pytest.raises(
        OllamaResponseError,
        match="model not found",
    ):
        client.chat(
            system_prompt="System",
            user_prompt="Question",
        )


def test_missing_message_content_is_rejected() -> None:
    client = OllamaChatClient(
        transport=RecordingTransport(
            {
                "model": "gemma3:latest",
                "message": {
                    "role": "assistant",
                },
                "done": True,
            }
        )
    )

    with pytest.raises(
        OllamaResponseError,
        match="no text content",
    ):
        client.chat(
            system_prompt="System",
            user_prompt="Question",
        )


def test_invalid_base_url_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="valid HTTP or HTTPS URL",
    ):
        OllamaChatClient(
            base_url="localhost:11434",
        )
