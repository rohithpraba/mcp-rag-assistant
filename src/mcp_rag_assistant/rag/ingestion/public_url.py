"""Controlled ingestion of public static HTML and direct PDF URLs."""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.parse import (
    unquote,
    urljoin,
    urlsplit,
    urlunsplit,
)

import httpx
from bs4 import BeautifulSoup

from .local_file import (
    calculate_content_hash,
    normalize_text,
    validate_workspace_id,
)
from .models import SourceDocument
from .pdf_file import PdfSourceDocument, load_local_pdf


DEFAULT_MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_REDIRECTS = 3

HTML_MEDIA_TYPES = {
    "text/html",
    "application/xhtml+xml",
}

REDIRECT_STATUS_CODES = {
    301,
    302,
    303,
    307,
    308,
}

Resolver = Callable[
    [str, int],
    Sequence[str],
]


class UrlFetchError(RuntimeError):
    """Raised when a public URL cannot be fetched safely."""


@dataclass(frozen=True, slots=True)
class FetchedResource:
    """Validated bytes downloaded from one public URL."""

    requested_url: str
    final_url: str
    media_type: str
    encoding: str | None
    body: bytes
    redirect_count: int


def _utc_timestamp() -> str:
    """Return a stable UTC timestamp representation."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def resolve_host_addresses(
    hostname: str,
    port: int,
) -> tuple[str, ...]:
    """Resolve IPv4 and IPv6 addresses for a hostname."""
    try:
        results = socket.getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as error:
        raise ValueError(
            f"URL hostname could not be resolved: {hostname}"
        ) from error

    addresses = {
        str(result[4][0]).split("%", 1)[0]
        for result in results
    }

    if not addresses:
        raise ValueError(
            f"URL hostname resolved to no addresses: {hostname}"
        )

    return tuple(sorted(addresses))


def _require_global_address(address_text: str) -> None:
    """Reject addresses that are not publicly routable."""
    try:
        address = ipaddress.ip_address(
            address_text.split("%", 1)[0]
        )
    except ValueError as error:
        raise ValueError(
            f"URL resolved to an invalid IP address: {address_text}"
        ) from error

    if not address.is_global:
        raise ValueError(
            "URL host resolved to a non-public address: "
            f"{address}"
        )


def validate_public_url(
    url: str,
    *,
    resolver: Resolver = resolve_host_addresses,
) -> str:
    """Validate and canonicalize a public HTTP or HTTPS URL."""
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    candidate = url.strip()

    if not candidate:
        raise ValueError("url must not be empty")

    if any(
        ord(character) < 32 or ord(character) == 127
        for character in candidate
    ):
        raise ValueError(
            "url must not contain control characters"
        )

    try:
        parsed = urlsplit(candidate)
    except ValueError as error:
        raise ValueError("url could not be parsed") from error

    scheme = parsed.scheme.lower()

    if scheme not in {"http", "https"}:
        raise ValueError(
            "only http and https URLs are supported"
        )

    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            "URLs containing credentials are not supported"
        )

    hostname = parsed.hostname

    if not hostname:
        raise ValueError("url must contain a hostname")

    if "%" in hostname:
        raise ValueError(
            "scoped or zone-qualified IP addresses are not supported"
        )

    try:
        explicit_port = parsed.port
    except ValueError as error:
        raise ValueError("url contains an invalid port") from error

    default_port = 443 if scheme == "https" else 80

    if (
        explicit_port is not None
        and explicit_port != default_port
    ):
        raise ValueError(
            "only default HTTP and HTTPS ports are supported"
        )

    hostname = hostname.rstrip(".").lower()

    if (
        hostname == "localhost"
        or hostname.endswith(".localhost")
    ):
        raise ValueError("localhost URLs are not supported")

    try:
        literal_address = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            canonical_hostname = (
                hostname.encode("idna")
                .decode("ascii")
                .lower()
            )
        except UnicodeError as error:
            raise ValueError(
                "url hostname could not be IDNA-normalized"
            ) from error

        resolved_addresses = tuple(
            resolver(
                canonical_hostname,
                default_port,
            )
        )

        if not resolved_addresses:
            raise ValueError(
                "url hostname resolved to no addresses"
            )
    else:
        canonical_hostname = str(literal_address)
        resolved_addresses = (
            canonical_hostname,
        )

    for address_text in resolved_addresses:
        _require_global_address(address_text)

    if ":" in canonical_hostname:
        netloc = f"[{canonical_hostname}]"
    else:
        netloc = canonical_hostname

    path = parsed.path or "/"

    return urlunsplit(
        (
            scheme,
            netloc,
            path,
            parsed.query,
            "",
        )
    )


def create_url_source_id(
    workspace_id: str,
    canonical_requested_url: str,
) -> str:
    """Create a stable source ID from workspace and canonical URL."""
    workspace = validate_workspace_id(workspace_id)

    identity_value = (
        f"{workspace}\0url\0{canonical_requested_url}"
    )

    digest = hashlib.sha256(
        identity_value.encode("utf-8")
    ).hexdigest()

    return f"src_{digest[:32]}"


def _response_media_type(
    response: httpx.Response,
) -> str:
    """Extract the response media type without parameters."""
    return (
        response.headers.get("content-type", "")
        .split(";", 1)[0]
        .strip()
        .lower()
    )


def fetch_public_resource(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    resolver: Resolver = resolve_host_addresses,
    transport: httpx.BaseTransport | None = None,
) -> FetchedResource:
    """Fetch one public HTML or PDF resource with strict limits."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")

    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds must be greater than zero"
        )

    if max_redirects < 0:
        raise ValueError(
            "max_redirects must be zero or greater"
        )

    requested_url = validate_public_url(
        url,
        resolver=resolver,
    )

    current_url = requested_url
    redirect_count = 0

    try:
        with httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
            headers={
                "User-Agent": (
                    "mcp-rag-assistant/0.1 "
                    "(controlled-document-ingestion)"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,"
                    "application/pdf"
                ),
            },
        ) as client:
            while True:
                # Revalidate every destination, including redirects.
                current_url = validate_public_url(
                    current_url,
                    resolver=resolver,
                )

                with client.stream(
                    "GET",
                    current_url,
                ) as response:
                    if (
                        response.status_code
                        in REDIRECT_STATUS_CODES
                    ):
                        if redirect_count >= max_redirects:
                            raise UrlFetchError(
                                "URL exceeded the redirect limit"
                            )

                        location = response.headers.get(
                            "location"
                        )

                        if not location:
                            raise UrlFetchError(
                                "redirect response has no location"
                            )

                        redirected_url = urljoin(
                            current_url,
                            location,
                        )

                        current_url = validate_public_url(
                            redirected_url,
                            resolver=resolver,
                        )

                        redirect_count += 1
                        continue

                    if 300 <= response.status_code < 400:
                        raise UrlFetchError(
                            "unsupported redirect response: "
                            f"HTTP {response.status_code}"
                        )

                    response.raise_for_status()

                    content_length = response.headers.get(
                        "content-length"
                    )

                    if content_length is not None:
                        try:
                            declared_length = int(
                                content_length
                            )
                        except ValueError as error:
                            raise UrlFetchError(
                                "response contains an invalid "
                                "Content-Length header"
                            ) from error

                        if declared_length < 0:
                            raise UrlFetchError(
                                "response contains a negative "
                                "Content-Length header"
                            )

                        if declared_length > max_bytes:
                            raise UrlFetchError(
                                "response exceeds the download "
                                "size limit"
                            )

                    body = bytearray()

                    for data in response.iter_bytes():
                        body.extend(data)

                        if len(body) > max_bytes:
                            raise UrlFetchError(
                                "response exceeds the download "
                                "size limit"
                            )

                    downloaded_body = bytes(body)

                    if not downloaded_body:
                        raise UrlFetchError(
                            "URL returned an empty response"
                        )

                    declared_media_type = (
                        _response_media_type(response)
                    )

                    if downloaded_body.startswith(b"%PDF-"):
                        media_type = "application/pdf"
                    elif declared_media_type in HTML_MEDIA_TYPES:
                        media_type = declared_media_type
                    elif declared_media_type == "application/pdf":
                        raise UrlFetchError(
                            "response claims to be a PDF but "
                            "has no PDF signature"
                        )
                    else:
                        raise UrlFetchError(
                            "unsupported response media type: "
                            f"{declared_media_type or 'missing'}"
                        )

                    encoding = (
                        response.encoding
                        if media_type in HTML_MEDIA_TYPES
                        else None
                    )

                    return FetchedResource(
                        requested_url=requested_url,
                        final_url=current_url,
                        media_type=media_type,
                        encoding=encoding,
                        body=downloaded_body,
                        redirect_count=redirect_count,
                    )
    except httpx.HTTPStatusError as error:
        raise UrlFetchError(
            "URL returned HTTP "
            f"{error.response.status_code}"
        ) from error
    except httpx.TimeoutException as error:
        raise UrlFetchError(
            "URL request timed out"
        ) from error
    except httpx.RequestError as error:
        raise UrlFetchError(
            f"URL request failed: {error}"
        ) from error


def _name_from_url(
    url: str,
    *,
    default_name: str,
) -> str:
    """Create a readable source name from a URL path."""
    parsed = urlsplit(url)

    path_name = PurePosixPath(
        unquote(parsed.path)
    ).name.strip()

    return path_name or default_name


def _extract_html_document(
    resource: FetchedResource,
    *,
    workspace_id: str,
) -> SourceDocument:
    """Convert downloaded static HTML into normalized source text."""
    soup = BeautifulSoup(
        resource.body,
        "html.parser",
        from_encoding=resource.encoding,
    )

    title = ""

    if soup.title is not None:
        title = re.sub(
            r"\s+",
            " ",
            soup.title.get_text(
                " ",
                strip=True,
            ),
        ).strip()

    for tag_name in (
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "iframe",
        "form",
    ):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    root = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    raw_text = root.get_text(
        "\n",
        strip=True,
    )

    lines: list[str] = []

    for raw_line in raw_text.splitlines():
        cleaned_line = re.sub(
            r"[ \t\f\v]+",
            " ",
            raw_line,
        ).strip()

        if not cleaned_line:
            continue

        # Consecutive duplicate navigation labels add little value.
        if lines and lines[-1] == cleaned_line:
            continue

        lines.append(cleaned_line)

    if title and (
        not lines
        or lines[0].casefold() != title.casefold()
    ):
        lines.insert(0, title)

    extracted_text = normalize_text(
        "\n".join(lines)
    )

    if len(extracted_text) < 20:
        raise ValueError(
            "HTML page contains insufficient extractable text"
        )

    source_name = (
        title[:200]
        if title
        else _name_from_url(
            resource.final_url,
            default_name=urlsplit(
                resource.final_url
            ).hostname
            or "web-page",
        )
    )

    workspace = validate_workspace_id(workspace_id)

    return SourceDocument(
        workspace_id=workspace,
        source_id=create_url_source_id(
            workspace,
            resource.requested_url,
        ),
        origin_type="url",
        media_type=resource.media_type,
        source_name=source_name,
        source_uri=resource.final_url,
        content_hash=calculate_content_hash(
            extracted_text
        ),
        byte_size=len(resource.body),
        ingested_at_utc=_utc_timestamp(),
        text=extracted_text,
        requested_uri=resource.requested_url,
        redirect_count=resource.redirect_count,
    )


def _extract_pdf_document(
    resource: FetchedResource,
    *,
    workspace_id: str,
    max_bytes: int,
) -> PdfSourceDocument:
    """Reuse the page-aware local PDF extractor for remote bytes."""
    with TemporaryDirectory() as directory:
        temporary_path = (
            Path(directory)
            / "downloaded-resource.pdf"
        )

        temporary_path.write_bytes(
            resource.body
        )

        local_document = load_local_pdf(
            temporary_path,
            workspace_id=workspace_id,
            max_bytes=max_bytes,
        )

    source_name = _name_from_url(
        resource.final_url,
        default_name="remote.pdf",
    )

    if not source_name.lower().endswith(".pdf"):
        source_name += ".pdf"

    remote_source = replace(
        local_document.source,
        source_id=create_url_source_id(
            workspace_id,
            resource.requested_url,
        ),
        origin_type="url",
        source_name=source_name,
        source_uri=resource.final_url,
        requested_uri=resource.requested_url,
        redirect_count=resource.redirect_count,
    )

    return replace(
        local_document,
        source=remote_source,
    )


def load_public_url(
    url: str,
    *,
    workspace_id: str = "default",
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    resolver: Resolver = resolve_host_addresses,
    transport: httpx.BaseTransport | None = None,
) -> SourceDocument | PdfSourceDocument:
    """Load a public static HTML page or direct PDF URL."""
    resource = fetch_public_resource(
        url,
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
        max_redirects=max_redirects,
        resolver=resolver,
        transport=transport,
    )

    if resource.media_type == "application/pdf":
        return _extract_pdf_document(
            resource,
            workspace_id=workspace_id,
            max_bytes=max_bytes,
        )

    return _extract_html_document(
        resource,
        workspace_id=workspace_id,
    )
