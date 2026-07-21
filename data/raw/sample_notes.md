# Dynamic Knowledge Base

A dynamic knowledge base allows sources to be added, updated, and removed
without retraining the generative language model.

The ingestion pipeline validates a source, extracts its text, preserves its
provenance, and prepares it for chunking and retrieval.

Retrieval-augmented generation can then retrieve evidence from the currently
indexed sources and place that evidence into the language model's context.
