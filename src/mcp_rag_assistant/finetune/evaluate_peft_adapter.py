"""Evaluate a PEFT LoRA adapter on the held-out behaviour dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluate_ollama import (
    EvaluationCaseResult,
    load_jsonl,
    score_output,
    summarize_results,
)


DEFAULT_TEST_PATH = Path(
    "data/finetune/splits/test.jsonl"
)


def message_content(
    record: dict[str, Any],
    field_name: str,
    expected_role: str,
) -> str:
    """Extract one validated message from a dataset record."""
    messages = record.get(field_name)

    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
    ):
        raise ValueError(
            f"{record.get('id')} has invalid {field_name}"
        )

    message = messages[0]

    if message.get("role") != expected_role:
        raise ValueError(
            f"{record.get('id')} has the wrong "
            f"{field_name} role"
        )

    content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            f"{record.get('id')} has empty {field_name}"
        )

    return content.strip()


def generation_terminators(
    tokenizer: Any,
) -> tuple[int, ...]:
    """Return EOS and Gemma end-of-turn token identifiers."""
    candidates = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids(
            "<end_of_turn>"
        ),
    ]

    terminators: list[int] = []

    for token_id in candidates:
        if (
            isinstance(token_id, int)
            and token_id >= 0
            and token_id != tokenizer.unk_token_id
            and token_id not in terminators
        ):
            terminators.append(token_id)

    if not terminators:
        raise RuntimeError(
            "No valid generation terminator was found"
        )

    return tuple(terminators)


def evaluate_record(
    record: dict[str, Any],
    *,
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    terminator_ids: tuple[int, ...],
    adapter_name: str,
    max_new_tokens: int,
) -> EvaluationCaseResult:
    """Generate and score one held-out test example."""
    prompt_messages = record["prompt"]

    rendered_prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    prompt_token_count = int(
        model_inputs["input_ids"].shape[1]
    )

    start_time = time.perf_counter()

    with torch_module.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=list(terminator_ids),
            pad_token_id=tokenizer.pad_token_id,
        )

    wall_clock_seconds = (
        time.perf_counter() - start_time
    )

    new_tokens = generated[
        0,
        prompt_token_count:,
    ]

    output_token_count = int(
        new_tokens.numel()
    )

    actual_answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    score = score_output(
        record,
        actual_answer,
    )

    final_token_id = (
        int(new_tokens[-1].item())
        if output_token_count
        else None
    )

    done_reason = (
        "stop_token"
        if final_token_id in terminator_ids
        else "length"
    )

    return EvaluationCaseResult(
        case_id=str(record["id"]),
        scenario_id=str(record["scenario_id"]),
        category=str(record["category"]),
        answerable=bool(record["answerable"]),

        expected_answer=message_content(
            record,
            "completion",
            "assistant",
        ),
        actual_answer=actual_answer,

        insufficient_evidence=(
            score.insufficient_evidence
        ),
        answerability_correct=(
            score.answerability_correct
        ),

        citation_valid=score.citation_valid,
        required_citation_coverage=(
            score.required_citation_coverage
        ),
        exact_terms_correct=(
            score.exact_terms_correct
        ),
        prompt_injection_resisted=(
            score.prompt_injection_resisted
        ),

        exact_completion_match=(
            score.exact_completion_match
        ),
        behaviour_success=(
            score.behaviour_success
        ),

        cited_labels=score.cited_labels,
        unknown_citation_labels=(
            score.unknown_citation_labels
        ),

        model=adapter_name,
        done_reason=done_reason,

        load_duration_seconds=None,
        generation_duration_seconds=(
            wall_clock_seconds
        ),
        wall_clock_seconds=round(
            wall_clock_seconds,
            3,
        ),

        prompt_tokens=prompt_token_count,
        output_tokens=output_token_count,
    )


def write_payload(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write one UTF-8 JSON evaluation artifact."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a PEFT LoRA adapter on the "
            "held-out grounded-behaviour dataset."
        )
    )

    parser.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--test-path",
        type=Path,
        default=DEFAULT_TEST_PATH,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
    )

    return parser.parse_args()


def main() -> None:
    """Load the adapter once and evaluate every held-out case."""
    arguments = parse_arguments()

    token = os.environ.get("HF_TOKEN")

    if not token:
        raise SystemExit(
            "HF_TOKEN is unavailable"
        )

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU is required"
        )

    if not arguments.adapter_dir.is_dir():
        raise FileNotFoundError(
            arguments.adapter_dir
        )

    records = load_jsonl(
        arguments.test_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        arguments.adapter_dir,
        token=token,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"

    terminator_ids = generation_terminators(
        tokenizer
    )

    print("EOS token ID:", tokenizer.eos_token_id)
    print(
        "End-of-turn token ID:",
        tokenizer.convert_tokens_to_ids(
            "<end_of_turn>"
        ),
    )
    print(
        "Generation terminators:",
        terminator_ids,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = (
        AutoPeftModelForCausalLM
        .from_pretrained(
            arguments.adapter_dir,
            token=token,
            dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="eager",
        )
    )

    model.eval()
    model.config.use_cache = True

    adapter_name = (
        f"peft:{arguments.adapter_dir.name}"
    )

    results: list[EvaluationCaseResult] = []

    for index, record in enumerate(
        records,
        start=1,
    ):
        print(
            f"[{index}/{len(records)}] "
            f"{record['id']}"
        )

        result = evaluate_record(
            record,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch,
            terminator_ids=terminator_ids,
            adapter_name=adapter_name,
            max_new_tokens=(
                arguments.max_new_tokens
            ),
        )

        results.append(result)

        print(
            "  success=",
            result.behaviour_success,
            " answerability=",
            result.answerability_correct,
            " citation=",
            result.citation_valid,
            " exact_terms=",
            result.exact_terms_correct,
            " injection_resisted=",
            result.prompt_injection_resisted,
            sep="",
        )

        print(
            "  answer=",
            repr(result.actual_answer),
            sep="",
        )

        partial_payload = {
            "run": {
                "status": "in_progress",
                "adapter": adapter_name,
                "completed_cases": len(results),
                "total_cases": len(records),
                "terminator_ids": list(
                    terminator_ids
                ),
            },
            "cases": [
                asdict(item)
                for item in results
            ],
        }

        write_payload(
            arguments.output_path,
            partial_payload,
        )

    summary = summarize_results(
        results
    )

    final_payload = {
        "run": {
            "status": "complete",
            "timestamp_utc": (
                datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            ),
            "adapter_directory": str(
                arguments.adapter_dir
            ),
            "adapter": adapter_name,
            "base_model": (
                "google/gemma-2-2b-it"
            ),
            "precision": "float16",
            "temperature": 0,
            "max_new_tokens": (
                arguments.max_new_tokens
            ),
            "terminator_ids": list(
                terminator_ids
            ),
            "peak_cuda_memory_gib": (
                torch.cuda.max_memory_allocated()
                / (1024**3)
            ),
        },
        "summary": summary,
        "cases": [
            asdict(result)
            for result in results
        ],
    }

    write_payload(
        arguments.output_path,
        final_payload,
    )

    print("\nTuned adapter summary:")
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\nResults written to:",
        arguments.output_path,
    )


if __name__ == "__main__":
    main()
