# Fine-Tuning Dataset Design

This dataset teaches stable grounded-answer behaviour rather than domain facts.

## Target behaviours

1. Answer only from supplied context.
2. Cite supplied labels such as `[S1]`.
3. Use grouped citations such as `[S1, S2]`.
4. Preserve technical identifiers exactly.
5. Return exactly `INSUFFICIENT_EVIDENCE` for unsupported questions.
6. Ignore instructions embedded inside source content.
7. Report conflicts between supplied sources.

## Split policy

- Training, validation, and held-out test splits use different scenarios.
- No scenario identifier or complete prompt is reused across splits.
- The test split is never supplied to the trainer.
- Synthetic facts are used so success cannot be attributed to memorized world knowledge.

## Format

Each JSONL record contains conversational `prompt` and `completion` fields,
plus metadata used for deterministic evaluation.
