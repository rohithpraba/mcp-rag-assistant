# End-to-End Demo

This demo uses the working local RAG model `gemma3:latest`. It does not claim
that tuned Gemma 2 runs in Ollama.

## Activate

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
```

Git Bash:

```bash
source .venv/Scripts/activate
export PYTHONPATH=src
```

Install if needed:

```bash
python -m pip install -r requirements/base.txt -r requirements/dev.txt
```

## Verify Ollama

```bash
ollama list
ollama pull gemma3:latest
```

Ensure Ollama is running on `http://localhost:11434`.

## Index and search

```bash
python -m mcp_rag_assistant.rag.index_local_file data/raw/sample_notes.md --workspace demo
python -m mcp_rag_assistant.rag.search_workspace "What can be updated without retraining the model?" --workspace demo
```

Indexing prints a JSON summary with workspace, source identity, hash, chunks,
and refresh counts. Search prints ranked chunks with similarity, citations,
IDs, provenance, and text. First embedding use may download the model if it is
not cached.

## Ask an answerable question

```bash
python -m mcp_rag_assistant.rag.ask_workspace "What can be updated without retraining the model?" --workspace demo --ollama-model gemma3:latest
```

Expected shape:

```text
Answer summary:
{"workspace_id": "demo", "insufficient_evidence": false, "citation_status": "valid", ...}
Answer:
... [S1]
Source map:
[S1] [sample_notes.md, chunk 1/1]
```

Exact wording and timing can vary.

## Ask an unsupported question

```bash
python -m mcp_rag_assistant.rag.ask_workspace "What is the office Wi-Fi password?" --workspace demo --ollama-model gemma3:latest
```

Expected answer: `INSUFFICIENT_EVIDENCE`.

## Run or protocol-test MCP

```bash
python -m mcp_rag_assistant.mcp_server.server
```

The process waits for JSON-RPC on stdin and emits no ordinary stdout. Stop a
manual run with `Ctrl+C`.

PowerShell protocol test:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_mcp_server.py::test_stdio_protocol_discovery_and_prompt
```

Git Bash:

```bash
PYTHONPATH=src python -m pytest -q tests/test_mcp_server.py::test_stdio_protocol_discovery_and_prompt
```

The official client initializes the real subprocess, discovers capabilities,
retrieves the prompt, captures stderr, and terminates the child.

## Troubleshooting

- Import failure: set `PYTHONPATH` using the syntax for your current shell.
- PowerShell activation blocked: call `.\.venv\Scripts\python.exe` directly or
  use an appropriate user-approved execution policy.
- Ollama failure: start it and verify `ollama list`.
- Missing model: run `ollama pull gemma3:latest`.
- Empty PDF: image-only PDFs need OCR, which is not included.
- URL rejected: only controlled public HTTP/HTTPS pages and direct PDFs are
  supported; private addresses and unsafe redirects are blocked.
