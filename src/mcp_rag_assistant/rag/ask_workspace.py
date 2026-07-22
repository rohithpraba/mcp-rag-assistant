"""Answer a question using retrieval and a local Ollama model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .answering.service import (
    DEFAULT_MAX_CONTEXT_CHARACTERS,
    GroundedAnswer,
    answer_question,
)
from .embeddings.sentence_transformer import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
)
from .generation.ollama_client import OllamaChatClient
from .storage.chroma_store import ChromaVectorStore


DEFAULT_DATABASE_PATH = Path("indexes/chroma")


@dataclass(frozen=True, slots=True)
class WorkspaceAnswerResult:
    """Workspace metadata and one complete grounded answer."""

    workspace_id: str
    stored_chunk_count: int
    embedding_model: str
    embedding_dimension: int
    grounded_answer: GroundedAnswer


def answer_workspace(
    question: str,
    *,
    workspace_id: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    top_k: int = 3,
    source_id: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_device: str = "cpu",
    ollama_model: str = "gemma3:latest",
    ollama_base_url: str = "http://localhost:11434",
    keep_alive: str | int | float = "5m",
    timeout_seconds: float = 300.0,
    max_context_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS,
    temperature: float = 0.0,
    max_output_tokens: int = 300,
) -> WorkspaceAnswerResult:
    """Run the full local retrieval-augmented generation path."""
    embedder = SentenceTransformerEmbedder(
        model_name=embedding_model,
        device=embedding_device,
    )

    store = ChromaVectorStore(
        database_path=database_path,
        workspace_id=workspace_id,
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.dimension,
    )

    generator = OllamaChatClient(
        model=ollama_model,
        base_url=ollama_base_url,
        timeout_seconds=timeout_seconds,
        keep_alive=keep_alive,
    )

    grounded_answer = answer_question(
        question,
        embedder=embedder,
        store=store,
        generator=generator,
        top_k=top_k,
        source_id=source_id,
        max_context_characters=max_context_characters,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    return WorkspaceAnswerResult(
        workspace_id=workspace_id,
        stored_chunk_count=store.count(),
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.dimension,
        grounded_answer=grounded_answer,
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Answer a question using retrieval from a persistent "
            "workspace and generation through a local Ollama model."
        )
    )

    parser.add_argument(
        "question",
        nargs="+",
        help="Natural-language question.",
    )

    parser.add_argument(
        "--workspace",
        required=True,
        help="Knowledge workspace identifier.",
    )

    parser.add_argument(
        "--database-path",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=(
            "Local Chroma database directory. "
            "Default: indexes/chroma."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum retrieved chunks. Default: 3.",
    )

    parser.add_argument(
        "--source-id",
        default=None,
        help="Optional exact source-ID retrieval filter.",
    )

    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence Transformer model identifier.",
    )

    parser.add_argument(
        "--embedding-device",
        default="cpu",
        help="Embedding device. Default: cpu.",
    )

    parser.add_argument(
        "--ollama-model",
        default="gemma3:latest",
        help="Local Ollama generation model.",
    )

    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Ollama server base URL.",
    )

    parser.add_argument(
        "--keep-alive",
        default="5m",
        help="Ollama model keep-alive duration. Default: 5m.",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Ollama request timeout. Default: 300.",
    )

    parser.add_argument(
        "--max-context-characters",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARACTERS,
        help=(
            "Maximum retrieved source-text characters. "
            "Default: 6000."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Default: 0.",
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=300,
        help="Maximum generated tokens. Default: 300.",
    )

    return parser.parse_args()


def _optional_duration(
    value: float | None,
) -> str | None:
    """Format an optional duration for JSON output."""
    if value is None:
        return None

    return f"{value:.3f}s"


def main() -> None:
    """Run complete workspace RAG from the command line."""
    arguments = parse_arguments()
    question = " ".join(arguments.question)

    try:
        result = answer_workspace(
            question,
            workspace_id=arguments.workspace,
            database_path=arguments.database_path,
            top_k=arguments.top_k,
            source_id=arguments.source_id,
            embedding_model=arguments.embedding_model,
            embedding_device=arguments.embedding_device,
            ollama_model=arguments.ollama_model,
            ollama_base_url=arguments.ollama_base_url,
            keep_alive=arguments.keep_alive,
            timeout_seconds=arguments.timeout_seconds,
            max_context_characters=(
                arguments.max_context_characters
            ),
            temperature=arguments.temperature,
            max_output_tokens=arguments.max_output_tokens,
        )
    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        raise SystemExit(
            f"answering failed: {error}"
        ) from error

    answer = result.grounded_answer
    generation = answer.generation

    summary = {
        "workspace_id": result.workspace_id,
        "stored_chunk_count": result.stored_chunk_count,
        "embedding_model": result.embedding_model,
        "embedding_dimension": result.embedding_dimension,
        "question": answer.question,
        "requested_top_k": answer.retrieval.requested_top_k,
        "retrieved_result_count": len(
            answer.retrieval.results
        ),
        "context_source_count": len(answer.sources),
        "context_character_count": (
            answer.context_character_count
        ),
        "insufficient_evidence": (
            answer.insufficient_evidence
        ),
        "citation_status": answer.citation_status,
        "cited_labels": list(answer.cited_labels),
        "unknown_citation_labels": list(
            answer.unknown_citation_labels
        ),
        "generation_model": (
            generation.model
            if generation is not None
            else None
        ),
        "load_duration": (
            _optional_duration(
                generation.load_duration_seconds
            )
            if generation is not None
            else None
        ),
        "total_duration": (
            _optional_duration(
                generation.total_duration_seconds
            )
            if generation is not None
            else None
        ),
        "prompt_tokens": (
            generation.prompt_tokens
            if generation is not None
            else None
        ),
        "output_tokens": (
            generation.output_tokens
            if generation is not None
            else None
        ),
    }

    print("RAG summary:")
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nAnswer:")
    print(answer.answer)

    if answer.citation_status == "missing":
        print(
            "\nCitation warning: the generated answer "
            "did not contain a source label."
        )

    if answer.citation_status == "unknown":
        print(
            "\nCitation warning: the generated answer used "
            "one or more unknown source labels."
        )

    if not answer.sources:
        print("\nNo source chunks were supplied to the model.")
        return

    print("\nSource map:")

    for source in answer.sources:
        source_uri = source.metadata.get(
            "source_uri",
            "unknown",
        )

        print(
            f"\n[{source.label}] {source.citation}"
        )
        print(
            f"   Retrieval rank: {source.retrieval_rank}"
        )
        print(
            f"   Similarity: {source.similarity:.4f}"
        )
        print(
            f"   Chunk ID: {source.chunk_id}"
        )
        print(
            f"   Source URI: {source_uri}"
        )
        print(
            f"   Truncated for context: {source.truncated}"
        )
        print("   Context text:")
        print(source.text)


if __name__ == "__main__":
    main()
