# Architecture Decisions

## ADR 1 — Manual NumPy retrieval before Chroma

The first search path used explicit NumPy cosine similarity and top-k
selection so vector mathematics stayed inspectable and testable before
persistence. The application later moved to Chroma.

## ADR 2 — Chroma as persistent vector store

Chroma provides local persistence, metadata filters, collection isolation, and
cosine search without a hosted service. Collection metadata records embedding
model, dimension, schema, and distance space so incompatibilities fail early.

## ADR 3 — Dynamic knowledge base rather than Spark-only scope

Domain-neutral ingestion and retrieval better demonstrate reusable source
lifecycle. Controlled documents still support deterministic evaluation without
hard-coding the application to one technical domain.

## ADR 4 — Stable source ID versus content hash

`source_id` remains stable for a logical source location. `content_hash`
changes with normalized content. This enables refresh, provenance, version
detection, and stale-chunk deletion.

## ADR 5 — Deterministic overlapping chunking

Word windows are inspectable and reproducible. Overlap preserves boundary
context, while deterministic IDs make unchanged indexing idempotent.

## ADR 6 — RAG for knowledge, LoRA for behaviour

Changing facts belong in retrievable sources. LoRA targets stable response
behaviour: grounding, abstention, citation discipline, exact terms, source
conflicts, and ignoring instructions embedded in source text.

## ADR 7 — Standard LoRA instead of QLoRA

A proof run showed FP16 LoRA fit on a Tesla T4 with about 8.70 GiB peak CUDA
allocation. Quantization complexity was unnecessary for this experiment.

## ADR 8 — MCP stdio as core transport

Local stdio is the smallest official MCP transport and fits a local,
secret-free assistant. It demonstrates tools, resources, prompts, schemas,
timeouts, and protocol behaviour without network exposure or authentication.

## ADR 9 — Defer tuned Ollama deployment

Direct PEFT evaluation works. Direct-adapter and merged-Safetensors imports
failed in the local Windows Ollama path-handling workflow. An unverified
Modelfile was removed rather than presented as working, and GGUF conversion
was kept outside the critical path.

## ADR 10 — Exclude optional features

UI, Docker, remote HTTP MCP, OCR, recursive crawling, reranking, and production
deployment add breadth without strengthening the completed RAG–LoRA–MCP core.
They remain optional extensions.
