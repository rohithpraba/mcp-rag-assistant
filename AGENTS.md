# Repository instructions

## Scope

Maintain the existing local-first RAG, MCP, web-demo, and optional Gemma LoRA evaluation architecture. Prefer small, reviewable changes over module moves or redesigns.

## Protected material

Never commit credentials, `.env` files, model weights, checkpoints, vector stores, databases, private notes, employer or client material, resumes, or unrelated intake files. The private shipment Data Engineering learning package and all extracted contents are outside this repository and must never be copied, summarized as completed work, or referenced in public project content.

## Claims and evidence

- Describe this project as a portfolio/research implementation, not a production service.
- Preserve exact `INSUFFICIENT_EVIDENCE` behaviour.
- Use the phrase `validated source labels`; label validation does not prove semantic entailment of every generated sentence.
- Keep evaluation sizes attached to results: 8 controlled RAG cases and 34 synthetic held-out tuning cases.
- Do not use `hallucination-free`, `production-ready`, `enterprise-grade`, or universal 100% performance wording.
- Keep the tuned adapter and Ollama deployment status distinct.

## Development boundaries

- Core RAG and MCP tests must not require Ollama, model downloads, internet access, or LoRA training.
- Training dependencies must remain optional and separate from normal runtime and development installation.
- Mock external model and network calls in unit tests.
- Do not expose arbitrary indexing or administrative operations through the fixed public demo.
- Do not redesign the current Ollama, Chroma, MCP, or FastAPI architecture without separate approval.

## Validation

Before and after material code changes, run the repository's supported equivalents of:

```bash
python -m pip install -e ".[dev,web]"
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
python -m pip check
```

Where Docker is available and no secret is required:

```bash
docker build -t mcp-rag-assistant:test .
```

Report unavailable checks honestly. Do not represent an unexecuted check as passing.

## Git and review

- Work on the approved feature branch.
- Preserve existing tests and recorded evaluation artifacts.
- Never force-push, rewrite history, rename, archive, delete, or merge the repository automatically.
- Review every diff for secrets, private paths, unsupported claims, and accidental generated artifacts.
