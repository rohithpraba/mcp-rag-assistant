# Resume and Interview Notes

## Resume bullets

- Built a local-first Python RAG assistant that ingests TXT/Markdown,
  page-aware PDFs, and controlled public URLs; maintains deterministic source
  lifecycle in Chroma; and generates cited Ollama answers with explicit
  insufficient-evidence abstention.
- Trained a standard FP16 LoRA adapter for Gemma 2 2B on 96 behaviour examples
  and improved measured accuracy from 85.29% to 100% on a **34-case synthetic
  held-out benchmark**, retaining 100% citation and exact-term scores on that
  controlled set.
- Optional: Exposed search and grounded answering through an official Python
  MCP stdio server with tools, a resource template, a prompt, and a real
  subprocess protocol test; the repository has 103 passing tests.

## 60-second walkthrough

“I built a local knowledge assistant around three boundaries. The RAG pipeline
validates TXT, Markdown, PDF, or controlled URL sources, creates stable source
IDs and content hashes, chunks deterministically, embeds with a Sentence
Transformer, and maintains workspaces in Chroma. At query time it retrieves
evidence, sends JSON-serialized untrusted context to local Ollama, and validates
citations or returns `INSUFFICIENT_EVIDENCE`.

I used LoRA for behaviour rather than knowledge. Gemma 2 2B was trained on
grounding, abstention, exact terms, conflicts, and indirect prompt injection.
The adapter scored 100% on a 34-case synthetic held-out benchmark versus
85.29% for the untuned model; I do not generalize that small result.

Finally, I wrapped the existing services in a local stdio MCP server with two
tools, one resource template, and one prompt. A real official-client test
launches the subprocess and verifies discovery.”

## Architecture in one paragraph

Ingestion owns validation and provenance; chunking and embeddings create
deterministic vector records; Chroma owns workspace state and source
replacement; retrieval owns ranking; answering owns context labels, grounding,
abstention, and citations; fine-tuning owns behaviour experiments; MCP remains
a thin validated adapter over search and answering.

## Exact benchmark wording

- “Phase 1 achieved 100% Hit Rate@3, MRR 1.0, and 100%
  answerability/citation/exact-term metrics on **8 cases over 2 controlled
  documents**.”
- “The tuned adapter achieved 100% behavioural accuracy on **34 synthetic
  held-out cases**, versus 85.29% for the untuned model on the same cases.”
- “These controlled benchmarks support regression evidence, not universal
  performance or production-readiness claims.”

## Five likely interview questions

### Why both RAG and fine-tuning?

RAG handles dynamic facts and provenance. LoRA targets stable behaviour.
Encoding changing knowledge in weights makes updates and citations harder.

### How are stale vectors avoided?

The source ID stays stable, content hash changes, and deterministic chunk IDs
encode version and position. Refresh upserts the current complete set, then
deletes stale IDs.

### How is document prompt injection limited?

Source text is JSON data, the grounding rules mark it untrusted, answers are
restricted to evidence, and citations are checked. The 34-case synthetic
result is evidence, not proof against arbitrary attacks.

### Why standard LoRA rather than QLoRA?

FP16 LoRA fit on a T4 with about 8.70 GiB peak CUDA allocation, so extra
quantization complexity was unnecessary.

### What remains before production?

Larger human/adversarial evaluation, authentication, remote transport,
observability, deployment, and model lifecycle controls. Tuned local Ollama
packaging is also deferred after Windows import failures.
