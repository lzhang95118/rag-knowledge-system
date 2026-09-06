from dataclasses import dataclass

import numpy as np

from rag_knowledge_system.chunking import Chunk

@dataclass
class SearchResult:
    chunk: Chunk
    score: float

class InMemoryVectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.embeddings: list[np.ndarray] = []

    def add(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        for chunk, embedding in zip(chunks, embeddings):
            self.chunks.append(chunk)
            self.embeddings.append(embedding)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
    ) -> list[SearchResult]:
        results = []

        for chunk, embedding in zip(self.chunks, self.embeddings):
            score = cosine_similarity(query_vector, embedding)

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]

def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    dot_product = np.dot(vector_a, vector_b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    return float(dot_product / (norm_a * norm_b))
