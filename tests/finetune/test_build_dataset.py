"""Tests for the deterministic fine-tuning dataset builder."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mcp_rag_assistant.finetune.build_dataset import (
    EXPECTED_SPLIT_SIZES,
    INSUFFICIENT_EVIDENCE,
    build_dataset,
)


_CITATION_LABEL = re.compile(r"S[1-9][0-9]*")


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """Load one generated JSON Lines file."""
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def build_and_load(
    directory: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Build all splits and reload them."""
    build_dataset(directory)

    return {
        split: load_jsonl(
            directory / f"{split}.jsonl"
        )
        for split in EXPECTED_SPLIT_SIZES
    }


def test_expected_split_sizes(
    tmp_path: Path,
) -> None:
    splits = build_and_load(
        tmp_path / "dataset"
    )

    assert {
        split: len(records)
        for split, records in splits.items()
    } == EXPECTED_SPLIT_SIZES


def test_ids_scenarios_and_prompts_do_not_leak(
    tmp_path: Path,
) -> None:
    splits = build_and_load(
        tmp_path / "dataset"
    )

    for field_name, extractor in (
        (
            "id",
            lambda record: record["id"],
        ),
        (
            "scenario_id",
            lambda record: record[
                "scenario_id"
            ],
        ),
        (
            "prompt",
            lambda record: record[
                "prompt"
            ][0]["content"],
        ),
    ):
        values_by_split = {
            split: {
                str(extractor(record))
                for record in records
            }
            for split, records in splits.items()
        }

        assert values_by_split[
            "train"
        ].isdisjoint(
            values_by_split["validation"]
        ), field_name

        assert values_by_split[
            "train"
        ].isdisjoint(
            values_by_split["test"]
        ), field_name

        assert values_by_split[
            "validation"
        ].isdisjoint(
            values_by_split["test"]
        ), field_name


def test_conversational_prompt_completion_schema(
    tmp_path: Path,
) -> None:
    splits = build_and_load(
        tmp_path / "dataset"
    )

    for records in splits.values():
        for record in records:
            assert record["prompt"] == [
                {
                    "role": "user",
                    "content": record[
                        "prompt"
                    ][0]["content"],
                }
            ]

            assert record["completion"] == [
                {
                    "role": "assistant",
                    "content": record[
                        "completion"
                    ][0]["content"],
                }
            ]

            assert (
                "CONTEXT_SOURCES_JSON"
                in record["prompt"][0]["content"]
            )


def test_unsupported_examples_use_exact_sentinel(
    tmp_path: Path,
) -> None:
    splits = build_and_load(
        tmp_path / "dataset"
    )

    for records in splits.values():
        for record in records:
            if not record["answerable"]:
                assert (
                    record["completion"][0][
                        "content"
                    ]
                    == INSUFFICIENT_EVIDENCE
                )


def test_answerable_citations_use_only_allowed_labels(
    tmp_path: Path,
) -> None:
    splits = build_and_load(
        tmp_path / "dataset"
    )

    for records in splits.values():
        for record in records:
            if not record["answerable"]:
                continue

            answer = record["completion"][0][
                "content"
            ]

            cited_labels = set(
                _CITATION_LABEL.findall(answer)
            )

            assert cited_labels
            assert cited_labels.issubset(
                set(record["allowed_labels"])
            )


def test_required_terms_are_preserved_exactly(
    tmp_path: Path,
) -> None:
    splits = build_and_load(
        tmp_path / "dataset"
    )

    for records in splits.values():
        for record in records:
            answer = record["completion"][0][
                "content"
            ]

            for required_term in record[
                "required_terms"
            ]:
                assert required_term in answer


def test_dataset_build_is_deterministic(
    tmp_path: Path,
) -> None:
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"

    first_manifest = build_dataset(
        first_directory
    )

    second_manifest = build_dataset(
        second_directory
    )

    assert first_manifest == second_manifest

    for filename in (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "manifest.json",
    ):
        assert (
            first_directory / filename
        ).read_bytes() == (
            second_directory / filename
        ).read_bytes()
