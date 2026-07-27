# Contributing

This repository is maintained as a focused portfolio/research implementation. Small, evidence-backed changes are preferred over broad rewrites.

## Development setup

Use Python 3.11 or later:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[web,dev]"
```

Training dependencies are optional:

```bash
python -m pip install -e ".[training]"
```

Do not download models or rerun LoRA training merely to validate a core code change.

## Validation

Run the checks applicable to the change:

```bash
python -m ruff check src tests
python -m pytest -q
python -m build
python -m pip check
```

Where Docker is available:

```bash
docker build -t mcp-rag-assistant:test .
```

Core tests must remain independent of Ollama, model downloads, public share links, and paid services. Mock external model and network calls in unit tests.

## Pull-request expectations

A pull request should explain:

- the problem being addressed;
- the files and behaviour changed;
- tests that were run and their exact result;
- any unavailable validation;
- claim, security, or compatibility implications.

Do not remove evaluation-set sizes or limitation wording from reported metrics. Do not claim production readiness, universal hallucination prevention, or guaranteed semantic correctness.

## Protected material

Never commit credentials, `.env` files, private documents, employer or client material, local databases, vector stores, model weights, checkpoints, or unrelated project intake files.

## Architecture changes

Changes that replace Ollama, Chroma, MCP transport, the grounding contract, or the public-demo boundary require a separate design discussion before implementation.
