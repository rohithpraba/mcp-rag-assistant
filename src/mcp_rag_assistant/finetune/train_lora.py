"""Train a standard FP16 LoRA adapter for Gemma 2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load one JSON configuration object."""
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"configuration does not exist: {config_path}"
        )

    payload = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "training configuration must be a JSON object"
        )

    return payload


def sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def package_version(name: str) -> str:
    """Return an installed package version."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def token_length_summary(
    dataset: Any,
    tokenizer: Any,
) -> dict[str, float | int]:
    """Measure formatted prompt-completion token lengths."""
    lengths: list[int] = []

    for record in dataset:
        messages = (
            list(record["prompt"])
            + list(record["completion"])
        )

        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        lengths.append(len(token_ids))

    if not lengths:
        raise ValueError("dataset contains no records")

    return {
        "count": len(lengths),
        "minimum": min(lengths),
        "maximum": max(lengths),
        "average": sum(lengths) / len(lengths),
    }


def trainable_parameter_summary(
    model: Any,
) -> dict[str, int | float]:
    """Count total and trainable parameters."""
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": (
            (trainable / total) * 100
            if total
            else 0.0
        ),
    }


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a standard FP16 LoRA adapter "
            "for the grounded-behaviour dataset."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/finetune/gemma2_lora.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--proof",
        action="store_true",
        help=(
            "Run two optimizer steps using small "
            "train and validation subsets."
        ),
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    """Run standard LoRA supervised fine-tuning."""
    arguments = parse_arguments()
    config = load_json_object(arguments.config)

    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        raise SystemExit(
            "HF_TOKEN is not available in the environment"
        )

    # Heavy training packages are imported only in Colab.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU is required for this training script"
        )

    output_directory = arguments.output_dir.resolve()
    checkpoint_directory = (
        output_directory / "checkpoints"
    )
    adapter_directory = (
        output_directory / "final_adapter"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_id = str(config["model_id"])
    train_path = Path(config["train_path"])
    validation_path = Path(
        config["validation_path"]
    )

    if not train_path.is_file():
        raise FileNotFoundError(train_path)

    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)

    set_seed(int(config["seed"]))

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(validation_path),
        },
    )

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    if arguments.proof:
        train_dataset = train_dataset.select(
            range(min(8, len(train_dataset)))
        )

        validation_dataset = (
            validation_dataset.select(
                range(
                    min(
                        4,
                        len(validation_dataset),
                    )
                )
            )
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
    )

    if tokenizer.chat_template is None:
        raise RuntimeError(
            "base tokenizer has no chat template"
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    train_token_lengths = token_length_summary(
        train_dataset,
        tokenizer,
    )

    validation_token_lengths = token_length_summary(
        validation_dataset,
        tokenizer,
    )

    max_length = int(config["max_length"])

    largest_sequence = max(
        int(train_token_lengths["maximum"]),
        int(validation_token_lengths["maximum"]),
    )

    if largest_sequence > max_length:
        raise RuntimeError(
            "formatted dataset contains a sequence of "
            f"{largest_sequence} tokens, exceeding "
            f"max_length={max_length}"
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / (1024**3),
            2,
        ),
        "GiB",
    )
    print(
        "Train token lengths:",
        json.dumps(
            train_token_lengths,
            indent=2,
        ),
    )
    print(
        "Validation token lengths:",
        json.dumps(
            validation_token_lengths,
            indent=2,
        ),
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        dtype=torch.float16,
        device_map={"": 0},
        attn_implementation=str(
            config["attn_implementation"]
        ),
    )

    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(
            config["lora_dropout"]
        ),
        target_modules=list(
            config["target_modules"]
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )

    proof_mode = arguments.proof

    training_arguments = SFTConfig(
        output_dir=str(checkpoint_directory),

        num_train_epochs=float(
            config["num_train_epochs"]
        ),
        max_steps=2 if proof_mode else -1,

        per_device_train_batch_size=int(
            config[
                "per_device_train_batch_size"
            ]
        ),
        per_device_eval_batch_size=int(
            config[
                "per_device_eval_batch_size"
            ]
        ),
        gradient_accumulation_steps=int(
            config[
                "gradient_accumulation_steps"
            ]
        ),

        learning_rate=float(
            config["learning_rate"]
        ),
        weight_decay=float(
            config["weight_decay"]
        ),
        warmup_ratio=float(
            config["warmup_ratio"]
        ),
        lr_scheduler_type=str(
            config["lr_scheduler_type"]
        ),
        max_grad_norm=float(
            config["max_grad_norm"]
        ),

        optim="adamw_torch_fused",

        fp16=True,
        bf16=False,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        use_cache=False,

        logging_steps=(
            1
            if proof_mode
            else int(config["logging_steps"])
        ),
        logging_first_step=True,

        eval_strategy=(
            "steps"
            if proof_mode
            else "epoch"
        ),
        eval_steps=1 if proof_mode else None,

        save_strategy=(
            "steps"
            if proof_mode
            else "epoch"
        ),
        save_steps=1,
        save_total_limit=int(
            config["save_total_limit"]
        ),

        load_best_model_at_end=(
            not proof_mode
        ),
        metric_for_best_model=(
            "eval_loss"
            if not proof_mode
            else None
        ),
        greater_is_better=False,

        report_to="none",

        seed=int(config["seed"]),
        data_seed=int(config["seed"]),

        max_length=max_length,
        truncation_mode="keep_start",
        packing=False,
        completion_only_loss=True,
        shuffle_dataset=True,
        dataset_num_proc=1,

        loss_type="chunked_nll",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    parameter_summary = (
        trainable_parameter_summary(
            trainer.model
        )
    )

    print(
        "Parameter summary:",
        json.dumps(
            parameter_summary,
            indent=2,
        ),
    )

    train_result = trainer.train(
        resume_from_checkpoint=(
            arguments.resume_from_checkpoint
            or None
        )
    )

    evaluation_metrics = trainer.evaluate()

    trainer.model.save_pretrained(
        adapter_directory,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        adapter_directory
    )

    peak_memory_gib = (
        torch.cuda.max_memory_allocated()
        / (1024**3)
    )

    adapter_files = sorted(
        path.name
        for path in adapter_directory.iterdir()
        if path.is_file()
    )

    summary = {
        "timestamp_utc": (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        ),
        "proof_mode": proof_mode,
        "model_id": model_id,
        "output_directory": str(
            output_directory
        ),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": (
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / (1024**3)
        ),
        "peak_cuda_memory_gib": peak_memory_gib,

        "train_records": len(train_dataset),
        "validation_records": len(
            validation_dataset
        ),

        "train_token_lengths": (
            train_token_lengths
        ),
        "validation_token_lengths": (
            validation_token_lengths
        ),

        "parameter_summary": parameter_summary,

        "train_metrics": dict(
            train_result.metrics
        ),
        "evaluation_metrics": dict(
            evaluation_metrics
        ),

        "adapter_files": adapter_files,

        "dataset_sha256": {
            "train": sha256(train_path),
            "validation": sha256(
                validation_path
            ),
        },

        "package_versions": {
            "torch": package_version("torch"),
            "transformers": package_version(
                "transformers"
            ),
            "trl": package_version("trl"),
            "peft": package_version("peft"),
            "datasets": package_version(
                "datasets"
            ),
            "accelerate": package_version(
                "accelerate"
            ),
        },

        "configuration": config,
    }

    summary_path = (
        output_directory / "training_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nTraining summary:")
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    required_adapter_files = {
        "adapter_config.json",
        "adapter_model.safetensors",
    }

    if not required_adapter_files.issubset(
        set(adapter_files)
    ):
        raise RuntimeError(
            "training completed but required adapter "
            "files were not saved"
        )


if __name__ == "__main__":
    main()
