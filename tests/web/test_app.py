from types import SimpleNamespace

from fastapi.testclient import TestClient

from mcp_rag_assistant.rag.storage.chroma_store import RetrievedChunk
from mcp_rag_assistant.web.app import Settings, WORKSPACE, create_app


def chunk():
    return RetrievedChunk(
        rank=1, chunk_id="c1", text="Evidence", similarity=.9, distance=.1,
        metadata={"source_name": "README.md", "source_uri": r"C:\private\README.md",
                  "chunk_index": 0, "chunk_count": 1},
    )


def search_fn(query, **kwargs):
    assert kwargs["workspace_id"] == WORKSPACE
    return SimpleNamespace(retrieval=SimpleNamespace(query=query, results=(chunk(),)))


def answer_fn(question, **kwargs):
    assert kwargs["workspace_id"] == WORKSPACE
    source = SimpleNamespace(
        label="S1", citation="[README.md, chunk 1/1]", retrieval_rank=1,
        text="Evidence", similarity=.9, truncated=False,
        metadata=chunk().metadata,
    )
    grounded = SimpleNamespace(
        answer="Grounded [S1]", insufficient_evidence=False,
        citation_status="valid", sources=(source,), generation=None,
    )
    return SimpleNamespace(grounded_answer=grounded)


def client(**settings):
    values = {"rate_limit": 100, **settings}
    cfg = Settings(**values)
    return TestClient(create_app(cfg, search_fn, answer_fn, lambda _: {
        "ready": False, "workspace_exists": False, "indexed_chunks": 0,
        "ollama_reachable": False,
    }), raise_server_exceptions=False)


def test_ui_health_and_unavailable_readiness():
    with client() as c:
        assert "MCP-Powered" in c.get("/").text
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/readyz").status_code == 503


def test_search_and_answer_are_sanitized_and_fixed():
    with client() as c:
        search = c.post("/api/v1/search", json={"query": "tech", "top_k": 3})
        answer = c.post("/api/v1/ask", json={"question": "tech", "top_k": 3})
    assert search.status_code == answer.status_code == 200
    assert WORKSPACE in search.text and WORKSPACE in answer.text
    assert "C:\\\\private" not in search.text + answer.text


def test_validation_and_no_mutations():
    with client() as c:
        for payload in ({"query": ""}, {"query": "x" * 501}, {"query": "x", "top_k": 6}):
            assert c.post("/api/v1/search", json=payload).status_code == 422
        assert c.post("/api/v1/ingest", json={}).status_code == 404
        assert c.delete("/api/v1/demo").status_code == 404


def test_backend_error_is_sanitized():
    def broken(*args, **kwargs):
        raise RuntimeError(r"C:\secret\failure")
    with TestClient(create_app(Settings(rate_limit=100), broken, answer_fn),
                    raise_server_exceptions=False) as c:
        response = c.post("/api/v1/search", json={"query": "x"})
    assert response.status_code == 503
    assert "secret" not in response.text


def test_rate_limit():
    with client(rate_limit=1) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/healthz").status_code == 429


def test_generation_concurrency_limit():
    app = create_app(Settings(rate_limit=100), search_fn, answer_fn)
    app.state.generation._value = 0
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.post("/api/v1/ask", json={"question": "x"}).status_code == 429
