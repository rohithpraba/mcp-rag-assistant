"""Reproducible baseline evaluation for the completed Phase 1 RAG system."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..answering.service import answer_question
from ..chunking.word_window import chunk_document
from ..embeddings.sentence_transformer import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
)
from ..generation.ollama_client import OllamaChatClient
from ..ingestion.local_file import load_local_text_file
from ..retrieval.service import retrieve_chunks
from ..storage.chroma_store import (
    ChromaVectorStore,
    RetrievedChunk,
)


DEFAULT_WORKSPACE_ID = "phase1-benchmark"
DEFAULT_DATABASE_PATH = Path(
    "indexes/chroma_phase1_benchmark"
)
DEFAULT_SOURCE_DIRECTORY = Path(
    "data/evaluation/phase1_sources"
)
DEFAULT_BENCHMARK_PATH = Path(
    "data/evaluation/phase1_benchmark.jsonl"
)
DEFAULT_RESULTS_PATH = Path(
    "data/evaluation/phase1_baseline_results.json"
)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One labelled retrieval and generation evaluation case."""

    case_id: str
    question: str
    answerable: bool
    expected_source_names: tuple[str, ...]
    required_term_groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class BenchmarkCaseResult:
    """Measured result for one benchmark case."""

    case_id: str
    question: str
    answerable: bool

    first_relevant_rank: int | None
    retrieval_hit: bool | None
    reciprocal_rank: float | None
    retrieval_latency_ms: float

    answer: str
    insufficient_evidence: bool
    answerability_correct: bool

    citation_status: str
    citation_correct: bool | None
    exact_terms_correct: bool | None

    generation_latency_seconds: float | None
    end_to_end_latency_seconds: float

    cited_labels: tuple[str, ...]
    top_sources: tuple[dict[str, object], ...]


def _clean_string(value: Any, field_name: str) -> str:
    """Validate and strip one required string field."""
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string"
        )

    cleaned = value.strip()

    if not cleaned:
        raise ValueError(
            f"{field_name} must not be empty"
        )

    return cleaned


def load_benchmark_cases(
    path: str | Path,
) -> tuple[BenchmarkCase, ...]:
    """Load and validate JSON Lines benchmark cases."""
    benchmark_path = Path(path)

    if not benchmark_path.is_file():
        raise FileNotFoundError(
            f"benchmark file does not exist: {benchmark_path}"
        )

    cases: list[BenchmarkCase] = []
    observed_ids: set[str] = set()

    for line_number, raw_line in enumerate(
        benchmark_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON on benchmark line {line_number}"
            ) from error

        if not isinstance(payload, dict):
            raise ValueError(
                f"benchmark line {line_number} must be an object"
            )

        case_id = _clean_string(
            payload.get("case_id"),
            "case_id",
        )

        if case_id in observed_ids:
            raise ValueError(
                f"duplicate benchmark case_id: {case_id}"
            )

        observed_ids.add(case_id)

        question = _clean_string(
            payload.get("question"),
            "question",
        )

        answerable = payload.get("answerable")

        if not isinstance(answerable, bool):
            raise ValueError(
                "answerable must be a boolean"
            )

        raw_source_names = payload.get(
            "expected_source_names",
            [],
        )

        if not isinstance(raw_source_names, list):
            raise ValueError(
                "expected_source_names must be a list"
            )

        expected_source_names = tuple(
            _clean_string(
                value,
                "expected_source_name",
            )
            for value in raw_source_names
        )

        if answerable and not expected_source_names:
            raise ValueError(
                f"answerable case {case_id} requires "
                "an expected source"
            )

        raw_groups = payload.get(
            "required_term_groups",
            [],
        )

        if not isinstance(raw_groups, list):
            raise ValueError(
                "required_term_groups must be a list"
            )

        term_groups: list[tuple[str, ...]] = []

        for group in raw_groups:
            if not isinstance(group, list) or not group:
                raise ValueError(
                    "each required term group must be "
                    "a non-empty list"
                )

            term_groups.append(
                tuple(
                    _clean_string(
                        term,
                        "required_term",
                    )
                    for term in group
                )
            )

        cases.append(
            BenchmarkCase(
                case_id=case_id,
                question=question,
                answerable=answerable,
                expected_source_names=(
                    expected_source_names
                ),
                required_term_groups=tuple(
                    term_groups
                ),
            )
        )

    if not cases:
        raise ValueError(
            "benchmark must contain at least one case"
        )

    return tuple(cases)


def first_relevant_rank(
    results: tuple[RetrievedChunk, ...]
    | list[RetrievedChunk],
    expected_source_names: tuple[str, ...],
) -> int | None:
    """Return the first rank whose source matches the label."""
    expected = set(expected_source_names)

    if not expected:
        return None

    for result in results:
        source_name = result.metadata.get(
            "source_name"
        )

        if source_name in expected:
            return result.rank

    return None


def required_terms_satisfied(
    answer: str,
    required_term_groups: tuple[
        tuple[str, ...],
        ...,
    ],
) -> bool | None:
    """Check exact technical terms, allowing alternatives per group."""
    if not required_term_groups:
        return None

    return all(
        any(
            candidate in answer
            for candidate in group
        )
        for group in required_term_groups
    )


def summarize_results(
    results: tuple[BenchmarkCaseResult, ...]
    | list[BenchmarkCaseResult],
) -> dict[str, int | float | None]:
    """Calculate aggregate benchmark metrics."""
    result_list = list(results)

    if not result_list:
        raise ValueError(
            "results must contain at least one case"
        )

    answerable_results = [
        result
        for result in result_list
        if result.answerable
    ]

    unanswerable_results = [
        result
        for result in result_list
        if not result.answerable
    ]

    exact_term_results = [
        result
        for result in result_list
        if result.exact_terms_correct is not None
    ]

    citation_results = [
        result
        for result in answerable_results
        if result.citation_correct is not None
    ]

    generation_latencies = [
        result.generation_latency_seconds
        for result in result_list
        if result.generation_latency_seconds
        is not None
    ]

    def ratio(
        numerator: int,
        denominator: int,
    ) -> float | None:
        if denominator == 0:
            return None

        return numerator / denominator

    return {
        "case_count": len(result_list),
        "answerable_case_count": len(
            answerable_results
        ),
        "unanswerable_case_count": len(
            unanswerable_results
        ),
        "retrieval_hit_rate_at_k": ratio(
            sum(
                result.retrieval_hit is True
                for result in answerable_results
            ),
            len(answerable_results),
        ),
        "mean_reciprocal_rank": (
            sum(
                result.reciprocal_rank or 0.0
                for result in answerable_results
            )
            / len(answerable_results)
            if answerable_results
            else None
        ),
        "answerability_accuracy": ratio(
            sum(
                result.answerability_correct
                for result in result_list
            ),
            len(result_list),
        ),
        "answerable_response_accuracy": ratio(
            sum(
                result.answerability_correct
                for result in answerable_results
            ),
            len(answerable_results),
        ),
        "unanswerable_abstention_accuracy": ratio(
            sum(
                result.answerability_correct
                for result in unanswerable_results
            ),
            len(unanswerable_results),
        ),
        "citation_validity_rate": ratio(
            sum(
                result.citation_correct is True
                for result in citation_results
            ),
            len(citation_results),
        ),
        "exact_term_accuracy": ratio(
            sum(
                result.exact_terms_correct is True
                for result in exact_term_results
            ),
            len(exact_term_results),
        ),
        "average_retrieval_latency_ms": (
            sum(
                result.retrieval_latency_ms
                for result in result_list
            )
            / len(result_list)
        ),
        "average_generation_latency_seconds": (
            sum(generation_latencies)
            / len(generation_latencies)
            if generation_latencies
            else None
        ),
        "average_end_to_end_latency_seconds": (
            sum(
                result.end_to_end_latency_seconds
                for result in result_list
            )
            / len(result_list)
        ),
    }


def index_benchmark_sources(
    *,
    source_directory: Path,
    database_path: Path,
    workspace_id: str,
    embedder: SentenceTransformerEmbedder,
    chunk_size_words: int,
    overlap_words: int,
) -> tuple[
    ChromaVectorStore,
    tuple[dict[str, object], ...],
]:
    """Create a clean persistent index for benchmark documents."""
    if database_path.exists():
        shutil.rmtree(database_path)

    store = ChromaVectorStore(
        database_path=database_path,
        workspace_id=workspace_id,
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.dimension,
    )

    indexed_sources: list[dict[str, object]] = []

    source_paths = sorted(
        source_directory.glob("*.md")
    )

    if not source_paths:
        raise ValueError(
            "benchmark source directory contains no Markdown files"
        )

    for source_path in source_paths:
        document = load_local_text_file(
            source_path,
            workspace_id=workspace_id,
        )

        chunks = chunk_document(
            document,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )

        embeddings = embedder.embed_documents(
            [
                chunk.text
                for chunk in chunks
            ]
        )

        refresh = store.replace_source_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        indexed_sources.append(
            {
                "source_name": document.source_name,
                "source_id": document.source_id,
                "chunk_count": (
                    refresh.current_chunk_count
                ),
            }
        )

    return store, tuple(indexed_sources)


def evaluate_case(
    case: BenchmarkCase,
    *,
    embedder: SentenceTransformerEmbedder,
    store: ChromaVectorStore,
    generator: OllamaChatClient,
    top_k: int,
) -> BenchmarkCaseResult:
    """Evaluate retrieval and generation for one labelled case."""
    retrieval_start = time.perf_counter()

    retrieval = retrieve_chunks(
        case.question,
        embedder=embedder,
        store=store,
        top_k=top_k,
    )

    retrieval_latency_ms = (
        time.perf_counter() - retrieval_start
    ) * 1000

    relevant_rank = first_relevant_rank(
        retrieval.results,
        case.expected_source_names,
    )

    if case.answerable:
        retrieval_hit: bool | None = (
            relevant_rank is not None
        )

        reciprocal_rank: float | None = (
            1.0 / relevant_rank
            if relevant_rank is not None
            else 0.0
        )
    else:
        retrieval_hit = None
        reciprocal_rank = None

    rag_start = time.perf_counter()

    answer = answer_question(
        case.question,
        embedder=embedder,
        store=store,
        generator=generator,
        top_k=top_k,
        max_context_characters=4000,
        temperature=0,
        max_output_tokens=100,
    )

    end_to_end_latency_seconds = (
        time.perf_counter() - rag_start
    )

    if case.answerable:
        answerability_correct = (
            not answer.insufficient_evidence
        )

        citation_correct: bool | None = (
            answer.citation_status == "valid"
        )
    else:
        answerability_correct = (
            answer.insufficient_evidence
        )

        citation_correct = None

    exact_terms_correct = (
        required_terms_satisfied(
            answer.answer,
            case.required_term_groups,
        )
    )

    generation_latency_seconds = (
        answer.generation.total_duration_seconds
        if answer.generation is not None
        else None
    )

    top_sources = tuple(
        {
            "rank": result.rank,
            "source_name": result.metadata.get(
                "source_name"
            ),
            "chunk_id": result.chunk_id,
            "similarity": round(
                result.similarity,
                6,
            ),
        }
        for result in retrieval.results
    )

    return BenchmarkCaseResult(
        case_id=case.case_id,
        question=case.question,
        answerable=case.answerable,
        first_relevant_rank=relevant_rank,
        retrieval_hit=retrieval_hit,
        reciprocal_rank=reciprocal_rank,
        retrieval_latency_ms=round(
            retrieval_latency_ms,
            3,
        ),
        answer=answer.answer,
        insufficient_evidence=(
            answer.insufficient_evidence
        ),
        answerability_correct=(
            answerability_correct
        ),
        citation_status=answer.citation_status,
        citation_correct=citation_correct,
        exact_terms_correct=(
            exact_terms_correct
        ),
        generation_latency_seconds=(
            generation_latency_seconds
        ),
        end_to_end_latency_seconds=round(
            end_to_end_latency_seconds,
            3,
        ),
        cited_labels=answer.cited_labels,
        top_sources=top_sources,
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed Phase 1 retrieval and grounded-generation "
            "benchmark."
        )
    )

    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
    )

    parser.add_argument(
        "--source-directory",
        type=Path,
        default=DEFAULT_SOURCE_DIRECTORY,
    )

    parser.add_argument(
        "--database-path",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
    )

    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
    )

    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE_ID,
    )

    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
    )

    parser.add_argument(
        "--ollama-model",
        default="gemma3:latest",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--chunk-size-words",
        type=int,
        default=90,
    )

    parser.add_argument(
        "--overlap-words",
        type=int,
        default=15,
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete baseline benchmark."""
    arguments = parse_arguments()

    cases = load_benchmark_cases(
        arguments.benchmark_path
    )

    embedder = SentenceTransformerEmbedder(
        model_name=arguments.embedding_model,
        device="cpu",
    )

    store, indexed_sources = index_benchmark_sources(
        source_directory=arguments.source_directory,
        database_path=arguments.database_path,
        workspace_id=arguments.workspace,
        embedder=embedder,
        chunk_size_words=arguments.chunk_size_words,
        overlap_words=arguments.overlap_words,
    )

    generator = OllamaChatClient(
        model=arguments.ollama_model,
        keep_alive="15m",
        timeout_seconds=300,
    )

    results: list[BenchmarkCaseResult] = []

    for index, case in enumerate(
        cases,
        start=1,
    ):
        print(
            f"[{index}/{len(cases)}] "
            f"Running {case.case_id}..."
        )

        result = evaluate_case(
            case,
            embedder=embedder,
            store=store,
            generator=generator,
            top_k=arguments.top_k,
        )

        results.append(result)

        print(
            "  retrieval_hit=",
            result.retrieval_hit,
            " answerability_correct=",
            result.answerability_correct,
            " citation=",
            result.citation_status,
            " exact_terms=",
            result.exact_terms_correct,
            sep="",
        )

        print(
            "  answer=",
            result.answer,
            sep="",
        )

    result_tuple = tuple(results)
    summary = summarize_results(result_tuple)

    payload = {
        "run": {
            "timestamp_utc": (
                datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            ),
            "workspace_id": arguments.workspace,
            "embedding_model": embedder.model_name,
            "embedding_dimension": embedder.dimension,
            "generation_model": (
                arguments.ollama_model
            ),
            "top_k": arguments.top_k,
            "chunk_size_words": (
                arguments.chunk_size_words
            ),
            "overlap_words": (
                arguments.overlap_words
            ),
            "stored_chunk_count": store.count(),
        },
        "indexed_sources": list(
            indexed_sources
        ),
        "summary": summary,
        "cases": [
            asdict(result)
            for result in result_tuple
        ],
    }

    arguments.results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arguments.results_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nPhase 1 baseline summary:")
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nResults written to:",
        arguments.results_path,
    )


if __name__ == "__main__":
    main()
