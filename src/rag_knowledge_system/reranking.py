from dataclasses import dataclass

from rag_knowledge_system.hybrid_retrieval import HybridSearchResult


@dataclass
class RerankedResult:
    result: HybridSearchResult
    rerank_score: float


def rerank_results(
    query: str,
    results: list[HybridSearchResult],
) -> list[RerankedResult]:
    reranked = []

    query_lower = query.lower()

    for result in results:
        bonus = 0.0

        if query_lower in result.chunk.content.lower():
            bonus += 0.2

        reranked.append(
            RerankedResult(
                result=result,
                rerank_score=result.final_score + bonus,
            )
        )

    reranked.sort(
        key=lambda item: item.rerank_score,
        reverse=True,
    )

    return reranked