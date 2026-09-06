import pytest
import numpy as np
from uuid import UUID

from rag_knowledge_system.chunking import Chunk
from rag_knowledge_system.hybrid import keyword_overlap_score
from rag_knowledge_system.hybrid_retrieval import HybridRetriever
from rag_knowledge_system.vector_store import InMemoryVectorStore


def test_keyword_overlap_score():
    score = keyword_overlap_score(
        "student attendance policy",
        "student attendance rules",
    )

    assert score == pytest.approx(2 / 3)

def test_keyword_overlap_score_return_zero_for_no_overlap():
    score = keyword_overlap_score(
        "student attendance policy",
        "library opening times",
    )

    assert score == 0.0


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0]])

def test_hybrid_retriever_combines_semantic_and_keyword_scores():
    chunk_a = Chunk(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        content="general university guidance",
        chunk_index=0,
    )

    chunk_b = Chunk(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        content="student attendance policy",
        chunk_index=1,
    )

    embeddings = np.array([
        [0.95, 0.05],
        [0.80, 0.20],
    ])

    store = InMemoryVectorStore()
    store.add(
        chunks=[chunk_a, chunk_b],
        embeddings=embeddings,
    )

    hybrid_retriever = HybridRetriever(
        embedder=FakeEmbedder(),
        vector_store=store,
        semantic_weight=0.5,
        keyword_weight=0.5,
    )

    results = hybrid_retriever.retrieve(
        query="student attendance policy",
        top_k=2,
    )


    assert results[0].chunk.content == "student attendance policy"
