"""A small semantic-search application using manual NumPy ranking."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from .vector_math import cosine_similarity_scores, top_k_indices


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


CORPUS: tuple[dict[str, str], ...] = (
    {
        "chunk_id": "concept-rag",
        "title": "Retrieval-augmented generation",
        "source": "project learning corpus",
        "text": (
            "Retrieval-augmented generation retrieves relevant external "
            "information before producing an answer. The retrieved passages "
            "are placed in the language model's input so the answer can be "
            "grounded in supplied evidence."
        ),
    },
    {
        "chunk_id": "concept-embedding",
        "title": "Embedding model",
        "source": "project learning corpus",
        "text": (
            "An embedding model converts text into a fixed-size dense vector. "
            "Texts with related meanings should receive vector representations "
            "that are close according to a selected similarity measure."
        ),
    },
    {
        "chunk_id": "concept-chunking",
        "title": "Document chunking",
        "source": "project learning corpus",
        "text": (
            "Chunking divides a long document into smaller retrievable units. "
            "Chunk size and overlap affect whether retrieval returns focused "
            "information while preserving enough surrounding context."
        ),
    },
    {
        "chunk_id": "concept-finetuning",
        "title": "LLM fine-tuning",
        "source": "project learning corpus",
        "text": (
            "Fine-tuning adapts a pretrained model using task examples. "
            "Parameter-efficient methods can train smaller adapter parameters "
            "instead of updating every parameter in the base model."
        ),
    },
    {
        "chunk_id": "concept-mcp",
        "title": "Model Context Protocol",
        "source": "project learning corpus",
        "text": (
            "The Model Context Protocol provides a standard way for AI "
            "applications to connect to servers that expose tools, resources, "
            "and reusable prompts."
        ),
    },
    {
        "chunk_id": "concept-vector-store",
        "title": "Vector store",
        "source": "project learning corpus",
        "text": (
            "A vector store persists embeddings and associated metadata. "
            "It searches for stored vectors that are nearest to a query vector "
            "according to a configured distance or similarity function."
        ),
    },
    {
        "chunk_id": "concept-generative-model",
        "title": "Generative language model",
        "source": "project learning corpus",
        "text": (
            "A generative language model predicts tokens to produce natural "
            "language. It performs a different job from an embedding model, "
            "which produces vectors used for comparison and retrieval."
        ),
    },
)


@dataclass(frozen=True)
class SearchResult:
    """One ranked semantic-search result."""

    rank: int
    chunk_id: str
    title: str
    source: str
    score: float
    text: str


def semantic_search(
    query: str,
    corpus: Sequence[Mapping[str, str]],
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[SearchResult]:
    """Embed a query and corpus, then retrieve the highest-scoring passages."""
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("query must not be empty")

    if not corpus:
        raise ValueError("corpus must not be empty")

    document_texts = [item["text"] for item in corpus]

    document_embeddings = model.encode_document(
        document_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_embedding = model.encode_query(
        cleaned_query,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    document_matrix = np.asarray(document_embeddings, dtype=np.float32)
    query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

    scores = cosine_similarity_scores(
        query_vector=query_vector,
        document_matrix=document_matrix,
    )

    ranked_indices = top_k_indices(scores=scores, k=top_k)

    results: list[SearchResult] = []

    for rank, index in enumerate(ranked_indices, start=1):
        corpus_item = corpus[int(index)]

        results.append(
            SearchResult(
                rank=rank,
                chunk_id=corpus_item["chunk_id"],
                title=corpus_item["title"],
                source=corpus_item["source"],
                score=float(scores[index]),
                text=corpus_item["text"],
            )
        )

    return results


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Search a small corpus using text embeddings."
    )

    parser.add_argument(
        "query",
        nargs="+",
        help="The natural-language query to search for.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to return. Default: 3.",
    )

    return parser.parse_args()


def main() -> None:
    """Run semantic search from the command line."""
    arguments = parse_arguments()
    query = " ".join(arguments.query)

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    results = semantic_search(
        query=query,
        corpus=CORPUS,
        model=model,
        top_k=arguments.top_k,
    )

    print(f"\nQuery: {query}")
    print(f"Results returned: {len(results)}")

    for result in results:
        print(
            f"\n{result.rank}. {result.title} "
            f"[score={result.score:.4f}]"
        )
        print(f"   Chunk ID: {result.chunk_id}")
        print(f"   Source: {result.source}")
        print(f"   Text: {result.text}")


if __name__ == "__main__":
    main()
