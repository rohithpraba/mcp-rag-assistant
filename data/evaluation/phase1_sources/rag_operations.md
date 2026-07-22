# Dynamic Knowledge Base Operations

A dynamic knowledge base accepts new sources after deployment without retraining the generative language model.

Within a workspace, source_id is the stable identifier for one logical source. When the source text changes, source_id remains the same while content_hash changes. content_hash is a SHA-256 fingerprint of normalized extracted content and identifies the current source version.

A source refresh upserts current chunks and deletes stale chunks from the previous version. A stale chunk is an indexed record that no longer belongs to the current authoritative source version.

Top-k semantic retrieval returns the nearest available chunks according to the configured vector distance. It does not guarantee that those chunks are relevant.

When supplied evidence does not support an answer, the grounded assistant should return INSUFFICIENT_EVIDENCE rather than answer from outside knowledge.
