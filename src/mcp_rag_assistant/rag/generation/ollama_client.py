"""Typed client for Ollama's local non-streaming chat API."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


JsonTransport = Callable[
    [str, dict[str, object], float],
    dict[str, object],
]


class OllamaError(RuntimeError):
    """Base exception for Ollama client failures."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama server cannot be reached."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an invalid or failed response."""


@dataclass(frozen=True, slots=True)
class OllamaChatResult:
    """Validated result from one completed Ollama chat request."""

    model: str
    content: str
    done: bool
    done_reason: str | None

    total_duration_seconds: float | None
    load_duration_seconds: float | None

    prompt_tokens: int | None
    output_tokens: int | None


def _nanoseconds_to_seconds(
    value: object,
) -> float | None:
    """Convert an optional non-negative nanosecond value to seconds."""
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and value >= 0
    ):
        return float(value) / 1_000_000_000

    return None


def _optional_nonnegative_int(
    value: object,
) -> int | None:
    """Return an integer metric when it is present and valid."""
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    ):
        return value

    return None


def _default_json_transport(
    url: str,
    payload: dict[str, object],
    timeout_seconds: float,
) -> dict[str, object]:
    """Send one JSON POST request and return a decoded object."""
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            decoded = json.load(response)
    except HTTPError as error:
        response_body = error.read().decode(
            "utf-8",
            errors="replace",
        )

        raise OllamaResponseError(
            f"Ollama returned HTTP {error.code}: "
            f"{response_body}"
        ) from error
    except URLError as error:
        raise OllamaConnectionError(
            f"Could not connect to Ollama: {error.reason}"
        ) from error
    except TimeoutError as error:
        raise OllamaConnectionError(
            "The Ollama request timed out"
        ) from error
    except json.JSONDecodeError as error:
        raise OllamaResponseError(
            "Ollama returned invalid JSON"
        ) from error

    if not isinstance(decoded, dict):
        raise OllamaResponseError(
            "Ollama response must be a JSON object"
        )

    return decoded


class OllamaChatClient:
    """Call a local Ollama model through the chat API."""

    def __init__(
        self,
        *,
        model: str = "gemma3:latest",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 300.0,
        keep_alive: str | int | float = "5m",
        transport: JsonTransport | None = None,
    ) -> None:
        cleaned_model = model.strip()

        if not cleaned_model:
            raise ValueError("model must not be empty")

        cleaned_base_url = base_url.strip().rstrip("/")
        parsed_url = urlparse(cleaned_base_url)

        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
        ):
            raise ValueError(
                "base_url must be a valid HTTP or HTTPS URL"
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero"
            )

        if isinstance(keep_alive, str):
            if not keep_alive.strip():
                raise ValueError(
                    "keep_alive must not be empty"
                )
        elif (
            not isinstance(keep_alive, int | float)
            or isinstance(keep_alive, bool)
        ):
            raise TypeError(
                "keep_alive must be a duration string or number"
            )

        self.model = cleaned_model
        self.base_url = cleaned_base_url
        self.timeout_seconds = float(timeout_seconds)
        self.keep_alive = keep_alive

        self._transport = (
            transport
            if transport is not None
            else _default_json_transport
        )

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 300,
    ) -> OllamaChatResult:
        """Generate one non-streaming chat response."""
        if not isinstance(system_prompt, str):
            raise TypeError(
                "system_prompt must be a string"
            )

        if not isinstance(user_prompt, str):
            raise TypeError(
                "user_prompt must be a string"
            )

        cleaned_system_prompt = system_prompt.strip()
        cleaned_user_prompt = user_prompt.strip()

        if not cleaned_system_prompt:
            raise ValueError(
                "system_prompt must not be empty"
            )

        if not cleaned_user_prompt:
            raise ValueError(
                "user_prompt must not be empty"
            )

        if (
            not isinstance(temperature, int | float)
            or isinstance(temperature, bool)
            or temperature < 0
        ):
            raise ValueError(
                "temperature must be a non-negative number"
            )

        if (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or max_output_tokens <= 0
        ):
            raise ValueError(
                "max_output_tokens must be a positive integer"
            )

        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": cleaned_system_prompt,
                },
                {
                    "role": "user",
                    "content": cleaned_user_prompt,
                },
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": float(temperature),
                "num_predict": max_output_tokens,
            },
        }

        response = self._transport(
            f"{self.base_url}/api/chat",
            payload,
            self.timeout_seconds,
        )

        api_error = response.get("error")

        if api_error:
            raise OllamaResponseError(
                f"Ollama API error: {api_error}"
            )

        message = response.get("message")

        if not isinstance(message, dict):
            raise OllamaResponseError(
                "Ollama response is missing message data"
            )

        content = message.get("content")

        if not isinstance(content, str):
            raise OllamaResponseError(
                "Ollama response message has no text content"
            )

        response_model = response.get(
            "model",
            self.model,
        )

        if not isinstance(response_model, str):
            raise OllamaResponseError(
                "Ollama response model must be a string"
            )

        done = response.get("done")

        if not isinstance(done, bool):
            raise OllamaResponseError(
                "Ollama response is missing completion status"
            )

        done_reason_value = response.get("done_reason")

        if (
            done_reason_value is not None
            and not isinstance(done_reason_value, str)
        ):
            raise OllamaResponseError(
                "Ollama done_reason must be a string or null"
            )

        return OllamaChatResult(
            model=response_model,
            content=content.strip(),
            done=done,
            done_reason=done_reason_value,
            total_duration_seconds=_nanoseconds_to_seconds(
                response.get("total_duration")
            ),
            load_duration_seconds=_nanoseconds_to_seconds(
                response.get("load_duration")
            ),
            prompt_tokens=_optional_nonnegative_int(
                response.get("prompt_eval_count")
            ),
            output_tokens=_optional_nonnegative_int(
                response.get("eval_count")
            ),
        )
