"""Tests for the fixed Phase 1 benchmark metrics."""

from __future__ import annotations

from mcp_rag_assistant.rag.evaluation.phase1_benchmark import (
    BenchmarkCaseResult,
    first_relevant_rank,
    required_terms_satisfied,
    summarize_results,
)
from mcp_rag_assistant.rag.storage.chroma_store import (
    RetrievedChunk,
)


def make_retrieved(
    rank: int,
    source_name: str,
) -> RetrievedChunk:
    """Create a small retrieval result."""
    return RetrievedChunk(
        rank=rank,
        chunk_id=f"chunk-{rank}",
        text="Evidence",
        distance=0.1 * rank,
        similarity=1.0 - (0.1 * rank),
        metadata={
            "source_name": source_name,
        },
    )


def make_case_result(
    *,
    case_id: str,
    answerable: bool,
    rank: int | None,
    answerability_correct: bool,
    citation_correct: bool | None,
    exact_terms_correct: bool | None,
) -> BenchmarkCaseResult:
    """Create a predictable benchmark result."""
    return BenchmarkCaseResult(
        case_id=case_id,
        question="Question",
        answerable=answerable,
        first_relevant_rank=rank,
        retrieval_hit=(
            rank is not None
            if answerable
            else None
        ),
        reciprocal_rank=(
            1.0 / rank
            if answerable and rank is not None
            else (
                0.0
                if answerable
                else None
            )
        ),
        retrieval_latency_ms=10.0,
        answer="Answer",
        insufficient_evidence=(
            not answerable
        ),
        answerability_correct=(
            answerability_correct
        ),
        citation_status=(
            "valid"
            if citation_correct
            else "missing"
        ),
        citation_correct=citation_correct,
        exact_terms_correct=(
            exact_terms_correct
        ),
        generation_latency_seconds=2.0,
        end_to_end_latency_seconds=2.5,
        cited_labels=("S1",),
        top_sources=(),
    )


def test_first_relevant_rank_finds_expected_source() -> None:
    results = [
        make_retrieved(1, "unrelated.md"),
        make_retrieved(2, "expected.md"),
    ]

    assert first_relevant_rank(
        results,
        ("expected.md",),
    ) == 2


def test_first_relevant_rank_returns_none_when_missing() -> None:
    results = [
        make_retrieved(1, "unrelated.md"),
    ]

    assert first_relevant_rank(
        results,
        ("expected.md",),
    ) is None


def test_required_term_groups_allow_alternatives() -> None:
    assert required_terms_satisfied(
        "The file is pyvenv.cfg in Scripts.",
        (
            ("pyvenv.cfg",),
            ("Scripts", "bin"),
        ),
    ) is True


def test_required_term_groups_detect_missing_term() -> None:
    assert required_terms_satisfied(
        "The file is venv.cfg.",
        (
            ("pyvenv.cfg",),
        ),
    ) is False


def test_summary_calculates_core_metrics() -> None:
    results = [
        make_case_result(
            case_id="a",
            answerable=True,
            rank=1,
            answerability_correct=True,
            citation_correct=True,
            exact_terms_correct=True,
        ),
        make_case_result(
            case_id="b",
            answerable=True,
            rank=2,
            answerability_correct=True,
            citation_correct=False,
            exact_terms_correct=False,
        ),
        make_case_result(
            case_id="c",
            answerable=False,
            rank=None,
            answerability_correct=True,
            citation_correct=None,
            exact_terms_correct=None,
        ),
    ]

    summary = summarize_results(results)

    assert summary["retrieval_hit_rate_at_k"] == 1.0
    assert summary["mean_reciprocal_rank"] == 0.75
    assert summary["answerability_accuracy"] == 1.0
    assert summary["citation_validity_rate"] == 0.5
    assert summary["exact_term_accuracy"] == 0.5
