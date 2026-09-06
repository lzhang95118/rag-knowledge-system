from uuid import UUID
import numpy as np

from rag_knowledge_system.embedding import HuggingFaceEmbedding
from rag_knowledge_system.chunking import Chunk
from rag_knowledge_system.hybrid_retrieval import HybridSearchResult
from rag_knowledge_system.reranking import rerank_results
from rag_knowledge_system.vector_store import InMemoryVectorStore

def test_rerank_results():
    chunk_a = Chunk(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        content="student attendance policy",
        chunk_index=0,
    )

    chunk_b = Chunk(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        content="library opening times",
        chunk_index=1,
    )

    results = [
        HybridSearchResult(
            chunk=chunk_a,
            semantic_score=0.6,
            keyword_score=0.5,
            final_score=0.55,
        ),
        HybridSearchResult(
            chunk=chunk_b,
            semantic_score=0.7,
            keyword_score=0.4,
            final_score=0.54,
        ),
    ]

    reranked_results = rerank_results(
        query="attendance policy",
        results=results,
    )

    assert len(reranked_results) == 2
    assert reranked_results[0].result.chunk.content == "student attendance policy"