# Project Status

## Phase 1 — RAG: complete

The repository implements TXT/Markdown, page-aware PDF, static public HTML,
and direct public PDF ingestion; stable source identity and content hashing;
deterministic overlapping chunks; Sentence Transformer embeddings; persistent
Chroma workspaces; source refresh and stale-chunk deletion; semantic
retrieval; grounded local Ollama answers; citations; and exact
`INSUFFICIENT_EVIDENCE` abstention.

The recorded Phase 1 benchmark contains **8 cases over 2 controlled
documents**. Its fixed benchmark was committed in `ed9f08f`.

## Phase 2 — LoRA training and PEFT evaluation: complete

The dataset contains 96 training, 22 validation, and 34 synthetic held-out
test examples. Standard FP16 LoRA training for `google/gemma-2-2b-it` and
direct PEFT evaluation are complete. The tuned adapter scored 100% on the
**34 synthetic held-out cases**; this is not a claim of universal performance.

Local tuned-model deployment through Windows Ollama is deferred because the
direct-adapter and merged-Safetensors routes failed in that environment. No
GGUF conversion or deployment retry is part of the completed core.

Relevant commits:

- `32f661c` — Gemma 2 LoRA training pipeline
- `186dbde` — final PEFT evaluation source and evidence

## Phase 3 — MCP: complete

Commit `31f324b` adds the official `mcp==1.28.1` SDK, local stdio FastMCP
server, `search_documents` and `answer_question` tools, a source-chunk
resource template, grounded-answer prompt, deterministic tests, and a real
protocol subprocess handshake.

## Phase 4 — portfolio packaging: complete

The root README and focused architecture, evaluation, demo, decision, status,
and interview documents complete the portfolio package without changing
application behaviour.

## Validation status

- Full repository suite: 103 tests passing
- Focused fine-tuning suite: 13 tests passing
- Focused MCP suite: 12 tests passing, including real stdio discovery
- Installed dependencies: `pip check` clean

## Optional future extensions

Larger human-reviewed evaluations, cross-platform tuned-model packaging, OCR,
recursive crawling, reranking, authenticated remote MCP, UI, containers, and
deployment automation remain optional rather than unfinished core work.
