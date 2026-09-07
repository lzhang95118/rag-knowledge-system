from uuid import UUID

from rag_knowledge_system.evaluation import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_knowledge_system.chunking import Chunk
from rag_knowledge_system.vector_store import SearchResult


def make_result(
        filename: str,
        chunk_index: int,
        score: float = 1.0,
) -> SearchResult:
    chunk = Chunk(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        content=f"chunk{chunk_index}",
        chunk_index=chunk_index,
        metadata={
            "filename": filename,
            "chunk_index": chunk_index,
        },
    )

    return SearchResult(
        chunk=chunk,
        score=score,
    )


def test_hit_at_k_returns_1_when_relevant_chunk_is_found():
    relevant_chunks = [
        make_result("test.txt", 0),
        make_result("test.txt", 1),
        make_result("test.txt", 2),
    ]

    relevant = {
        ("test.txt", 1),
    }

    assert hit_at_k(relevant_chunks, relevant, k=3) == 1.0

def test_hit_at_k_returns_0_when_relevant_chunk_is_not_found():
    relevant_chunks = [
        make_result("test.txt", 0),
        make_result("test.txt", 1),
    ]

    relevant = {
        ("test.txt", 3),
    }

    assert hit_at_k(relevant_chunks, relevant, k=3) == 0.0

def test_recall_at_k_returns_fraction_of_relevant_chunks_found():
    retrieved = [
        make_result("test.txt", 0),
        make_result("test.txt", 2),
        make_result("test.txt", 4),
    ]

    relevant = {
        ("test.txt", 0),
        ("test.txt", 1),
        ("test.txt", 2),
    }

    assert recall_at_k(
        retrieved,
        relevant,
        k=3,
    ) == 2 / 3

def test_reciprocal_rank_returns_first_correct_value():
    relevant_chunks = [
        make_result("test.txt", 0),
        make_result("test.txt", 2),
        make_result("test.txt", 1),
    ]

    relevant = {
        ("test.txt", 2),
    }

    assert reciprocal_rank(relevant_chunks, relevant) == 0.5

def test_reciprocal_rank_returns_0_when_no_relevant_chunks_found():
    relevant_chunks = [
        make_result("test.txt", 0),
        make_result("test.txt", 1),
    ]

    relevant = {
        ("test.txt", 2),
    }

    assert reciprocal_rank(relevant_chunks, relevant) == 0.0