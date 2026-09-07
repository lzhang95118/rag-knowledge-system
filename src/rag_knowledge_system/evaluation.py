import json
from pathlib import Path

from collections.abc import Iterable

from rag_knowledge_system.vector_store import SearchResult

def load_cases(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_key(
        filename: str,
        chunk_index: int
    ) -> tuple[str, int]:
    return filename, chunk_index


def result_key(result: SearchResult) -> tuple[str, int]:
    metadata = result.chunk.metadata
    return chunk_key(
        filename=metadata["filename"],
        chunk_index=metadata["chunk_index"],
    )


def hit_at_k(
    retrieved: Iterable[SearchResult],
    relevant_chunks: set[tuple[str, int]],
    k: int,
) -> float:
    top_k = list(retrieved)[:k]

    return float(
        any(result_key(result) in relevant_chunks for result in top_k)
    )


def recall_at_k(
    retrieved: Iterable[SearchResult],
    relevant_chunks: set[tuple[str, int]],
    k: int,
) -> float:
    if not relevant_chunks:
        return 0.0

    retrieved_keys = {
        result_key(result)
        for result in list(retrieved)[:k]
    }

    matched = retrieved_keys & relevant_chunks

    return len(matched) / len(relevant_chunks)


def reciprocal_rank(
    retrieved: Iterable[SearchResult],
    relevant_chunks: set[tuple[str, int]],
) -> float:
    for rank, result in enumerate(retrieved, start=1):
        if result_key(result) in relevant_chunks:
            return 1.0 / rank

    return 0.0


def evaluate_retriever(
    retriever,
    cases: list[dict],
    k: int = 3,
) -> dict[str, float]:
    hit_scores = []
    recall_scores = []
    rr_scores = []

    for case in cases:
        query = case["query"]

        relevant_chunks = {
            chunk_key(
                filename=chunk["filename"],
                chunk_index=chunk["chunk_index"],
            )
            for chunk in case["relevant_chunks"]
        }

        retrieved = retriever.retrieve(
            query=query,
            top_k=k,
        )

        hit_scores.append(
            hit_at_k(retrieved, relevant_chunks, k)
        )

        recall_scores.append(
            recall_at_k(retrieved, relevant_chunks, k)
        )

        rr_scores.append(
            reciprocal_rank(retrieved, relevant_chunks)
        )

    count = len(cases)

    return {
        f"hit_rate@{k}": sum(hit_scores) / count,
        f"recall@{k}": sum(recall_scores) / count,
        "mrr": sum(rr_scores) / count,
    }