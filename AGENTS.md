# Repository Instructions

## Structure

- `src/mcp_rag_assistant/rag/`: ingestion, chunking, embeddings, storage,
  retrieval, grounding, Ollama, and CLIs
- `src/mcp_rag_assistant/finetune/`: data, LoRA training, smoke testing, and
  evaluation
- `src/mcp_rag_assistant/mcp_server/`: official-SDK local stdio adapter
- `tests/`: deterministic RAG, fine-tuning, MCP, and protocol tests
- `data/evaluation/`, `data/finetune/evaluation/`: small committed evidence
- `docs/`: project and portfolio documentation

## Commands

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

Git Bash:

```bash
PYTHONPATH=src python -m pytest -q
```

Focused checks:

```bash
PYTHONPATH=src python -m pytest -q tests/finetune
PYTHONPATH=src python -m pytest -q tests/test_mcp_server.py
python -m pip check
git diff --check
```

## Testing and boundaries

- Run focused tests for touched components and the full suite.
- Mock models, Ollama, network, and indexes in unit tests.
- Do not retrain models or rebuild recorded benchmarks for ordinary work.
- MCP discovery must need no Ollama, index, model load, internet, or secrets.
- Reuse `rag/retrieval/service.py` and `rag/answering/service.py`; never
  duplicate RAG or grounding logic.
- RAG owns dynamic knowledge; LoRA targets behaviour; MCP is a thin local
  stdio adapter.

## Protected content

Do not modify or commit `.venv/`, `models/`, `outputs/`, `indexes/`,
`checkpoints/`, caches, multi-gigabyte merged models, credentials, tokens, or
`.env` files. Never add model binaries to Git. Treat generated artifacts as
read-only unless a task explicitly authorizes a narrow operation.

Do not add a dependency without explaining what and why, then obtaining
approval. Never print or copy secrets. Do not claim production readiness.
Pair Phase 1 metrics with “8 cases over 2 controlled documents” and tuned
metrics with “34 synthetic held-out cases.”

## Deferred work

Tuned Gemma 2 Windows Ollama deployment, GGUF conversion, UI, Docker, OCR,
recursive crawling, reranking, remote MCP, authentication, hosted deployment,
and the full learning handbook are optional/deferred.
