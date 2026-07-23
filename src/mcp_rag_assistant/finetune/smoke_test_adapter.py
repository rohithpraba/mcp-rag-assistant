"""Load a saved LoRA adapter and generate one held-out response."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_TEST_PATH = Path(
    "data/finetune/splits/test.jsonl"
)

DEFAULT_CASE_ID = (
    "test-indirect_prompt_injection-001-variant-2"
)


def load_case(
    path: Path,
    case_id: str,
) -> dict[str, Any]:
    """Load one requested held-out case."""
    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        record = json.loads(line)

        if record.get("id") == case_id:
            return record

    raise ValueError(
        f"test case was not found: {case_id}"
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser()

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
        "--case-id",
        default=DEFAULT_CASE_ID,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
    )

    return parser.parse_args()


def main() -> None:
    """Load adapter and run one deterministic generation."""
    arguments = parse_arguments()

    token = os.environ.get("HF_TOKEN")

    if not token:
        raise SystemExit("HF_TOKEN is unavailable")

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")

    record = load_case(
        arguments.test_path,
        arguments.case_id,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        arguments.adapter_dir,
        token=token,
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        arguments.adapter_dir,
        token=token,
        dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="eager",
    )

    model.eval()

    rendered_prompt = tokenizer.apply_chat_template(
        record["prompt"],
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        rendered_prompt,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=(
                arguments.max_new_tokens
            ),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[
        0,
        model_inputs["input_ids"].shape[1]:,
    ]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    print("Case ID:", record["id"])
    print(
        "Expected:",
        record["completion"][0]["content"],
    )
    print("Actual:", answer)


if __name__ == "__main__":
    main()
