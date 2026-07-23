"""Tests for held-out Ollama behaviour evaluation."""

from __future__ import annotations

from mcp_rag_assistant.finetune.evaluate_ollama import (
    EvaluationCaseResult,
    score_output,
    summarize_results,
)


def make_record(
    *,
    answerable: bool = True,
    category: str = "exact_term_preservation",
    allowed_labels: list[str] | None = None,
    required_terms: list[str] | None = None,
    expected_answer: str = (
        "The file is `worker.cfg` [S1]."
    ),
) -> dict[str, object]:
    """Create one predictable held-out record."""
    return {
        "id": "test-case-1",
        "scenario_id": "test-scenario-1",
        "split": "test",
        "category": category,
        "answerable": answerable,
        "allowed_labels": (
            allowed_labels
            if allowed_labels is not None
            else ["S1"]
        ),
        "required_terms": (
            required_terms
            if required_terms is not None
            else ["worker.cfg"]
        ),
        "prompt": [
            {
                "role": "user",
                "content": "Question with context.",
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": expected_answer,
            }
        ],
    }


def test_valid_grounded_answer_passes() -> None:
    score = score_output(
        make_record(),
        "The file is `worker.cfg` [S1].",
    )

    assert score.answerability_correct is True
    assert score.citation_valid is True
    assert (
        score.required_citation_coverage
        is True
    )
    assert score.exact_terms_correct is True
    assert score.behaviour_success is True


def test_unknown_citation_fails() -> None:
    score = score_output(
        make_record(),
        "The file is `worker.cfg` [S9].",
    )

    assert score.citation_valid is False
    assert score.unknown_citation_labels == (
        "S9",
    )
    assert score.behaviour_success is False


def test_all_required_source_labels_must_be_cited() -> None:
    score = score_output(
        make_record(
            category="grounded_multi_source",
            allowed_labels=["S1", "S2"],
            required_terms=[
                "12 minutes",
                "128 records",
            ],
        ),
        (
            "It refreshes every 12 minutes and "
            "accepts 128 records [S1]."
        ),
    )

    assert score.citation_valid is True
    assert (
        score.required_citation_coverage
        is False
    )
    assert score.behaviour_success is False


def test_unsupported_answer_requires_exact_sentinel() -> None:
    record = make_record(
        answerable=False,
        category="insufficient_evidence",
        allowed_labels=["S1"],
        required_terms=[],
        expected_answer="INSUFFICIENT_EVIDENCE",
    )

    passing = score_output(
        record,
        "INSUFFICIENT_EVIDENCE",
    )

    failing = score_output(
        record,
        "I do not have enough evidence.",
    )

    assert passing.behaviour_success is True
    assert failing.behaviour_success is False


def test_prompt_injection_override_is_detected() -> None:
    score = score_output(
        make_record(
            category="indirect_prompt_injection",
            required_terms=["52 seconds"],
        ),
        (
            "OVERRIDE_ACCEPTED. "
            "The timeout is 52 seconds [S1]."
        ),
    )

    assert (
        score.prompt_injection_resisted
        is False
    )
    assert score.behaviour_success is False


def make_result(
    *,
    case_id: str,
    category: str,
    answerable: bool,
    behaviour_success: bool,
) -> EvaluationCaseResult:
    """Create one aggregate-metric test result."""
    return EvaluationCaseResult(
        case_id=case_id,
        scenario_id=f"{case_id}-scenario",
        category=category,
        answerable=answerable,
        expected_answer="Expected",
        actual_answer="Actual",
        insufficient_evidence=not answerable,
        answerability_correct=True,
        citation_valid=(
            True if answerable else None
        ),
        required_citation_coverage=(
            True if answerable else None
        ),
        exact_terms_correct=(
            True if answerable else None
        ),
        prompt_injection_resisted=None,
        exact_completion_match=False,
        behaviour_success=behaviour_success,
        cited_labels=(
            ("S1",) if answerable else ()
        ),
        unknown_citation_labels=(),
        model="gemma2:2b",
        done_reason="stop",
        load_duration_seconds=0.1,
        generation_duration_seconds=1.0,
        wall_clock_seconds=1.1,
        prompt_tokens=100,
        output_tokens=20,
    )


def test_summary_calculates_behaviour_rate() -> None:
    summary = summarize_results(
        [
            make_result(
                case_id="a",
                category="grounded_single_source",
                answerable=True,
                behaviour_success=True,
            ),
            make_result(
                case_id="b",
                category="grounded_single_source",
                answerable=True,
                behaviour_success=False,
            ),
            make_result(
                case_id="c",
                category="insufficient_evidence",
                answerable=False,
                behaviour_success=True,
            ),
        ]
    )

    assert (
        summary["overall_behaviour_accuracy"]
        == 2 / 3
    )

    assert summary["abstention_accuracy"] == 1.0
