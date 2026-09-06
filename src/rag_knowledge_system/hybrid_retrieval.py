from collections.abc import Mapping
from typing import Any
from dataclasses import dataclass

from rag_knowledge_system.chunking import Chunk
from rag_knowledge_system.embedding import HuggingFaceEmbedding, Embedder
from rag_knowledge_system.hybrid import keyword_overlap_score
from rag_knowledge_system.vector_store import (
    InMemoryVectorStore
)


@dataclass
class HybridSearchResult:
    chunk: Chunk
    semantic_score: float
    keyword_score: float
    final_score: float

class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: InMemoryVectorStore,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[HybridSearchResult]:

        query_vector = self.embedder.embed_texts([query])[0]

        semantic_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=len(self.vector_store.chunks),
            metadata_filter=metadata_filter,
        )

        hybrid_results = []

        for result in semantic_results:
            keyword_score = keyword_overlap_score(
                query,
                result.chunk.content
            )

            final_score = (
                self.semantic_weight * result.score
                + self.keyword_weight * keyword_score
            )

            hybrid_results.append(
                HybridSearchResult(
                    chunk=result.chunk,
                    semantic_score=result.score,
                    keyword_score=keyword_score,
                    final_score=final_score,
                )
            )

        hybrid_results.sort(
            key=lambda result: result.final_score,
            reverse=True,
        )

        return hybrid_results[:top_k]
