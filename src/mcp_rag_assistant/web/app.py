"""FastAPI application for the fixed, read-only public demo."""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..rag.answering.service import LabeledSource
from ..rag.ask_workspace import answer_workspace
from ..rag.retrieval.service import format_chunk_citation
from ..rag.search_workspace import search_workspace
from ..rag.storage.chroma_store import workspace_collection_name

WORKSPACE = "public-demo"
DEMO_SOURCES = ("README.md", "ARCHITECTURE.md", "EVALUATION.md", "DECISIONS.md")
STATIC = Path(__file__).with_name("static")


@dataclass(frozen=True)
class Settings:
    database_path: Path = Path(os.getenv("DEMO_DATABASE_PATH", "indexes/chroma"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3:latest")
    github_url: str = os.getenv("GITHUB_REPOSITORY_URL", "")
    generation_timeout: float = float(os.getenv("GENERATION_TIMEOUT_SECONDS", "120"))
    rate_limit: int = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
    rate_window: float = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    max_body: int = int(os.getenv("MAX_REQUEST_BODY_BYTES", "2048"))
    trust_cloudflare: bool = os.getenv("TRUST_CLOUDFLARE", "false").lower() == "true"
    public_demo_mode: bool = os.getenv("PUBLIC_DEMO_MODE", "true").lower() == "true"


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input must not be empty")
        return value.strip()


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)

    @field_validator("question")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input must not be empty")
        return value.strip()


def _safe_metadata(metadata: dict[str, object]) -> dict[str, object]:
    allowed = {"source_name", "chunk_index", "chunk_count", "page_start", "page_end"}
    return {key: value for key, value in metadata.items() if key in allowed}


def _source_payload(source: LabeledSource) -> dict[str, object]:
    return {
        "label": source.label, "citation": source.citation,
        "rank": source.retrieval_rank, "text": source.text,
        "similarity": source.similarity, "truncated": source.truncated,
        "metadata": _safe_metadata(source.metadata),
    }


def create_app(
    settings: Settings | None = None,
    search_fn: Callable[..., Any] = search_workspace,
    answer_fn: Callable[..., Any] = answer_workspace,
    readiness_fn: Callable[[Settings], dict[str, object]] | None = None,
) -> FastAPI:
    cfg = settings or Settings()
    app = FastAPI(title="Public RAG Demo", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.state.generation = asyncio.Semaphore(1)
    app.state.clients = defaultdict(deque)

    def client_id(request: Request) -> str:
        if cfg.trust_cloudflare:
            value = request.headers.get("CF-Connecting-IP")
            if value:
                return value
        return request.client.host if request.client else "unknown"

    @app.middleware("http")
    async def controls(request: Request, call_next: Callable[..., Any]) -> Any:
        if request.method in {"POST", "PUT", "PATCH"}:
            length = request.headers.get("content-length")
            if length:
                try:
                    declared_size = int(length)
                except ValueError:
                    return JSONResponse({"detail": "invalid content length"}, status_code=400)
                if declared_size > cfg.max_body:
                    return JSONResponse({"detail": "request body too large"}, status_code=413)
            body = await request.body()
            if len(body) > cfg.max_body:
                return JSONResponse({"detail": "request body too large"}, status_code=413)
        now = time.monotonic()
        bucket = app.state.clients[client_id(request)]
        while bucket and bucket[0] <= now - cfg.rate_window:
            bucket.popleft()
        if len(bucket) >= cfg.rate_limit:
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
        bucket.append(now)
        return await call_next(request)

    @app.exception_handler(Exception)
    async def sanitized_error(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse({"detail": "demo backend unavailable"}, status_code=503)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready() -> JSONResponse:
        result = await asyncio.to_thread(readiness_fn or _readiness, cfg)
        return JSONResponse(result, status_code=200 if result["ready"] else 503)

    @app.get("/api/v1/demo/sources")
    async def sources() -> dict[str, object]:
        return {
            "workspace": WORKSPACE,
            "github_repository_url": cfg.github_url,
            "sources": [{"source_name": name} for name in DEMO_SOURCES],
        }

    @app.post("/api/v1/search")
    async def search(body: Query) -> dict[str, object]:
        try:
            result = await asyncio.to_thread(
                search_fn, body.query, workspace_id=WORKSPACE,
                database_path=cfg.database_path, top_k=body.top_k,
            )
        except Exception as error:
            raise HTTPException(503, "search backend unavailable") from error
        chunks = []
        for item in result.retrieval.results:
            chunks.append({
                "rank": item.rank, "text": item.text, "similarity": item.similarity,
                "citation": format_chunk_citation(item),
                "source_name": item.metadata.get("source_name", "unknown"),
                "metadata": _safe_metadata(item.metadata),
            })
        return {"workspace": WORKSPACE, "query": result.retrieval.query, "results": chunks}

    @app.post("/api/v1/ask")
    async def ask(body: Question) -> dict[str, object]:
        if app.state.generation.locked():
            raise HTTPException(429, "generation already in progress")
        async with app.state.generation:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        answer_fn, body.question, workspace_id=WORKSPACE,
                        database_path=cfg.database_path, top_k=body.top_k,
                        ollama_model=cfg.ollama_model, ollama_base_url=cfg.ollama_base_url,
                        timeout_seconds=cfg.generation_timeout,
                    ),
                    timeout=cfg.generation_timeout,
                )
            except TimeoutError as error:
                raise HTTPException(504, "generation timed out") from error
            except Exception as error:
                raise HTTPException(503, "answer backend unavailable") from error
        answer = result.grounded_answer
        return {
            "workspace": WORKSPACE, "answer": answer.answer,
            "insufficient_evidence": answer.insufficient_evidence,
            "citation_status": answer.citation_status,
            "citations": [s.citation for s in answer.sources],
            "sources": [_source_payload(s) for s in answer.sources],
            "generation_latency_seconds": (
                answer.generation.total_duration_seconds if answer.generation else None
            ),
        }

    return app


def _readiness(settings: Settings) -> dict[str, object]:
    workspace_exists = settings.database_path.exists()
    chunk_count = 0
    if workspace_exists:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(settings.database_path))
            collection = client.get_collection(workspace_collection_name(WORKSPACE))
            chunk_count = collection.count()
        except Exception:
            chunk_count = 0
    ollama = False
    try:
        with urlopen(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2) as response:
            ollama = response.status == 200
    except Exception:
        pass
    return {
        "ready": workspace_exists and chunk_count > 0 and ollama,
        "workspace_exists": workspace_exists, "indexed_chunks": chunk_count,
        "ollama_reachable": ollama,
    }


app = create_app()
