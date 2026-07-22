"""Tests for controlled public HTML and PDF URL ingestion."""

from __future__ import annotations

from pathlib import Path

import httpx
import pymupdf
import pytest

from mcp_rag_assistant.rag.ingestion.models import (
    SourceDocument,
)
from mcp_rag_assistant.rag.ingestion.pdf_file import (
    PdfSourceDocument,
)
from mcp_rag_assistant.rag.ingestion.public_url import (
    UrlFetchError,
    load_public_url,
    validate_public_url,
)


def public_resolver(
    hostname: str,
    port: int,
) -> tuple[str, ...]:
    """Return a deterministic globally routable test address."""
    return ("93.184.216.34",)


def test_canonical_url_removes_fragment_and_default_port() -> None:
    canonical = validate_public_url(
        "HTTPS://Example.COM:443/docs?x=1#section",
        resolver=public_resolver,
    )

    assert canonical == (
        "https://example.com/docs?x=1"
    )


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1/",
        "http://user:password@example.com/",
        "https://example.com:8443/",
    ],
)
def test_unsafe_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        validate_public_url(
            url,
            resolver=public_resolver,
        )


def test_html_url_extracts_main_content_and_provenance() -> None:
    html = b"""
    <html>
      <head>
        <title>Official Test Documentation</title>
        <script>dangerous_script()</script>
      </head>
      <body>
        <nav>Repeated navigation</nav>
        <main>
          <h1>Dynamic knowledge</h1>
          <p>A public page can become a retrievable source.</p>
        </main>
      </body>
    </html>
    """

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/html; charset=utf-8",
            },
            content=html,
        )

    document = load_public_url(
        "https://example.com/docs#section",
        workspace_id="demo",
        resolver=public_resolver,
        transport=httpx.MockTransport(handler),
    )

    assert isinstance(document, SourceDocument)
    assert document.origin_type == "url"
    assert document.media_type == "text/html"
    assert document.source_name == (
        "Official Test Documentation"
    )
    assert document.requested_uri == (
        "https://example.com/docs"
    )
    assert document.source_uri == (
        "https://example.com/docs"
    )
    assert document.redirect_count == 0

    assert "Dynamic knowledge" in document.text
    assert "retrievable source" in document.text
    assert "dangerous_script" not in document.text
    assert "Repeated navigation" not in document.text


def test_redirect_target_is_revalidated_and_recorded() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        if request.url.host == "example.com":
            return httpx.Response(
                302,
                headers={
                    "location": (
                        "https://docs.example.com/final#part"
                    ),
                },
            )

        return httpx.Response(
            200,
            headers={
                "content-type": "text/html; charset=utf-8",
            },
            content=(
                b"<html><title>Final Page</title>"
                b"<main>Final redirected documentation.</main>"
                b"</html>"
            ),
        )

    document = load_public_url(
        "https://example.com/start",
        workspace_id="demo",
        resolver=public_resolver,
        transport=httpx.MockTransport(handler),
    )

    assert isinstance(document, SourceDocument)
    assert document.requested_uri == (
        "https://example.com/start"
    )
    assert document.source_uri == (
        "https://docs.example.com/final"
    )
    assert document.redirect_count == 1


def test_download_size_limit_is_enforced() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/html",
            },
            content=(
                b"<html><main>"
                + (b"x" * 100)
                + b"</main></html>"
            ),
        )

    with pytest.raises(
        UrlFetchError,
        match="size limit",
    ):
        load_public_url(
            "https://example.com/large",
            max_bytes=32,
            resolver=public_resolver,
            transport=httpx.MockTransport(handler),
        )


def test_same_url_keeps_source_id_when_content_changes() -> None:
    def load_version(text: str) -> SourceDocument:
        def handler(
            request: httpx.Request,
        ) -> httpx.Response:
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/html; charset=utf-8",
                },
                content=(
                    "<html><title>Versioned Page</title>"
                    f"<main>{text}</main></html>"
                ).encode("utf-8"),
            )

        result = load_public_url(
            "https://example.com/versioned",
            workspace_id="demo",
            resolver=public_resolver,
            transport=httpx.MockTransport(handler),
        )

        assert isinstance(result, SourceDocument)
        return result

    first = load_version(
        "The first source version contains useful text."
    )

    second = load_version(
        "The second source version contains changed text."
    )

    assert first.source_id == second.source_id
    assert first.content_hash != second.content_hash


def test_direct_pdf_url_preserves_page_metadata(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "remote.pdf"

    document = pymupdf.open()

    try:
        document.insert_page(
            -1,
            text="Remote PDF evidence appears on page one.",
        )
        document.save(pdf_path)
    finally:
        document.close()

    pdf_bytes = pdf_path.read_bytes()

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "application/pdf",
            },
            content=pdf_bytes,
        )

    loaded = load_public_url(
        "https://example.com/guide.pdf",
        workspace_id="demo",
        resolver=public_resolver,
        transport=httpx.MockTransport(handler),
    )

    assert isinstance(loaded, PdfSourceDocument)
    assert loaded.page_count == 1
    assert loaded.text_page_count == 1
    assert loaded.source.origin_type == "url"
    assert loaded.source.media_type == "application/pdf"
    assert loaded.source.source_name == "guide.pdf"
    assert loaded.source.requested_uri == (
        "https://example.com/guide.pdf"
    )
    assert loaded.source.source_uri == (
        "https://example.com/guide.pdf"
    )
    assert "Remote PDF evidence" in loaded.source.text
