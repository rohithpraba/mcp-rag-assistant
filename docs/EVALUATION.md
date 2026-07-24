# Evaluation

## Phase 1 RAG benchmark

The fixed benchmark uses **8 cases over 2 controlled documents** in
`data/evaluation/`: six answerable and two unanswerable questions.

| Metric | Recorded result |
|---|---:|
| Retrieval Hit Rate@3 | 100% |
| Mean reciprocal rank | 1.00 |
| Answerability accuracy | 100% |
| Answerable-response accuracy | 100% |
| Unanswerable abstention accuracy | 100% |
| Citation-label validity | 100% |
| Exact-term accuracy | 100% |

Recorded average latency was approximately 50.30 ms for retrieval, 18.72
seconds for generation, and 20.81 seconds end to end. These local-run timings
are not general performance guarantees.

## Behaviour dataset

The deterministic dataset has 152 examples: 96 training, 22 validation, and
34 synthetic held-out test cases. Categories cover grounded single-source and
multi-source answers, exact terms, abstention, indirect prompt injection, and
source conflicts. Scenario IDs, prompts, and technical facts are separated
across splits to reduce direct leakage.

## Untuned Gemma 2

The untuned `gemma2:2b` baseline on the **34 synthetic held-out cases**:

| Metric | Result |
|---|---:|
| Overall behaviour accuracy | 85.29% |
| Answerability accuracy | 94.12% |
| Abstention accuracy | 66.67% |
| Prompt-injection resistance | 25% |
| Citation validity | 100% |
| Required-citation coverage | 100% |
| Exact-term accuracy | 100% |

## Tuned PEFT adapter

The FP16 LoRA adapter on the same **34 synthetic held-out cases**:

| Metric | Result |
|---|---:|
| Overall behaviour accuracy | 100% |
| Answerability accuracy | 100% |
| Abstention accuracy | 100% |
| Prompt-injection resistance | 100% |
| Citation validity | 100% |
| Required-citation coverage | 100% |
| Exact-term accuracy | 100% |

“100%” means **100% on these 34 synthetic held-out cases**. It is not evidence
of universal prompt-injection resistance, general model quality, or
production readiness.

## LoRA configuration and resources

| Setting | Value |
|---|---:|
| Base model | `google/gemma-2-2b-it` |
| Precision | FP16 |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Epochs | 3 |
| Effective optimizer steps | 72 |
| Trainable parameters | 20,766,720 |
| Trainable proportion | 0.788078% |
| Recorded training runtime | 199.49 seconds |
| Peak training CUDA allocation | 8.70 GiB |
| GPU | Tesla T4 |

The 72 steps follow from 96 records × 3 epochs, batch size 1, and gradient
accumulation 4. Recorded training loss was 0.03818; validation loss was about
0.0000534. Direct PEFT held-out generation used about 5.02 GiB peak CUDA
allocation.

## Distinguishing the measurements

- **Training loss** measures optimization fit on 96 training records.
- **Validation loss** measures token-level fit on 22 development records.
- **Held-out behavioural accuracy** scores generated behaviour on 34 separate
  synthetic test cases with answerability, citation, term, and
  injection-resistance checks.

Low loss does not replace behavioural evaluation, and this controlled
behavioural set does not replace broad external or human evaluation.

## Limitations

- Phase 1 contains only 8 cases and 2 controlled documents.
- Fine-tuning contains only 34 synthetic held-out cases.
- Templates are less diverse than real users and adversarial documents.
- Automated checks do not comprehensively measure usefulness or safety.
- There is no independent human or production-traffic evaluation.
- The tuned adapter was evaluated through PEFT; local Windows Ollama
  deployment remains deferred.
