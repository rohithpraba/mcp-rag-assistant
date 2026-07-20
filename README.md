# MCP-Powered RAG Assistant with a Fine-Tuned Domain LLM

A local-first AI assistant combining retrieval-augmented generation,
parameter-efficient LLM fine-tuning, and Model Context Protocol integration.

## Project goals

- Build a document ingestion and retrieval pipeline.
- Generate grounded answers with source citations.
- Fine-tune a small language model using LoRA or QLoRA.
- Compare the base, RAG, fine-tuned, and fine-tuned-plus-RAG systems.
- Expose retrieval and generation capabilities through an MCP server.
- Evaluate retrieval quality, answer grounding, and system latency.

## Planned architecture

1. Ingest official domain documentation.
2. Clean documents and divide them into metadata-rich chunks.
3. Convert chunks into embeddings.
4. Store and retrieve embeddings using a local vector store.
5. Pass retrieved evidence to a local language model.
6. Fine-tune model behaviour using parameter-efficient fine-tuning.
7. Expose project capabilities through MCP tools, resources, and prompts.

## Repository structure

- `src/mcp_rag_assistant/rag/` — ingestion, chunking, embeddings, and retrieval
- `src/mcp_rag_assistant/finetune/` — dataset preparation and fine-tuning utilities
- `src/mcp_rag_assistant/mcp_server/` — MCP server implementation
- `data/raw/` — source documents
- `data/processed/` — generated document chunks
- `data/evaluation/` — labelled evaluation questions
- `tests/` — automated tests
- `configs/` — application and experiment configuration
- `docs/` — architecture, experiments, terminology, and interview preparation
- `notebooks/` — learning and experimentation notebooks

## Current status

Phase 0 — Environment and repository setup.

## Cost requirement

The core system uses local software, open models, open-source libraries, and
free compute options. No paid LLM or embedding API is required.
