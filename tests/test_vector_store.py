import numpy as np
import pytest

from uuid import UUID

from rag_knowledge_system.chunking import Chunk
from rag_knowledge_system.vector_store import cosine_similarity, InMemoryVectorStore


def test_cosine_similarity_same_vector():
    vector = np.array([1.0, 2.0, 3.0])

    store = cosine_similarity(vector, vector)

    assert store == 1.0

def test_cosine_similarity_orthogonal_vectors():
    vector_a = np.array([1.0, 0.0])
    vector_b = np.array([0.0, 1.0])

    store = cosine_similarity(vector_a, vector_b)

    assert store == 0.0


def test_vector_store_returns_most_similar_chunk():
    chunk_a = Chunk(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        content="attendance policy",
        chunk_index=0,
    )

    chunk_b = Chunk(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        content="library opening times",
        chunk_index=1,
    )

    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    store = InMemoryVectorStore()
    store.add(
        chunks=[chunk_a, chunk_b],
        embeddings=embeddings,
    )

    query_vector = np.array([1.0, 0.0])

    results = store.search(query_vector, top_k=1)

    assert results[0].chunk.content == "attendance policy"
    assert results[0].score == 1.0


def test_vector_store_rejects_mismatched_lengths():
    store = InMemoryVectorStore()

    chunk = Chunk(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        content="attendance policy",
        chunk_index=0,
    )
    embeddings = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    with pytest.raises(ValueError):
        store.add(
            chunks=[chunk],
            embeddings=embeddings,
        )


def test_vector_store_filters_by_metadata():
    chunk_a = Chunk(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        content="attendance policy",
        chunk_index=0,
        metadata={"category": "attendance"},
    )

    chunk_b = Chunk(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        content="library opening times",
        chunk_index=1,
        metadata={"category": "library"},
    )

    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    store = InMemoryVectorStore()
    store.add(
        chunks=[chunk_a, chunk_b],
        embeddings=embeddings,
    )

    query_vector = np.array([1.0, 0.0])
    metadata = {"category": "attendance"}

    results = store.search(query_vector, top_k=2, metadata_filter=metadata)

    assert len(results) == 1
    assert results[0].chunk.content == "attendance policy"