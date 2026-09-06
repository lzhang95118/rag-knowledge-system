from uuid import UUID

from rag_knowledge_system.chunking import Chunk
from rag_knowledge_system.embedding import HuggingFaceEmbedding
from rag_knowledge_system.retrieval import Retriever
from rag_knowledge_system.vector_store import InMemoryVectorStore

def test_retriever_returns_most_similar_chunk():
    Chunk_a = Chunk(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        content="attendance policy",
        chunk_index=0,
    )

    Chunk_b = Chunk(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        content="library opening times",
        chunk_index=1,
    )

    Chunk_c = Chunk(
        document_id=UUID("00000000-0000-0000-0000-000000000001"),
        content="library closing times",
        chunk_index=2,
    )

    embedder = HuggingFaceEmbedding()

    embeddings = embedder.embed_texts([
        Chunk_a.content,
        Chunk_b.content,
        Chunk_c.content,
    ])

    store = InMemoryVectorStore()
    store.add(
        chunks=[Chunk_a, Chunk_b, Chunk_c],
        embeddings=embeddings,
    )

    retriever = Retriever(
        vector_store=store,
        embedder=embedder,
    )

    result = retriever.retrieve(
        query="student attendance",
        top_k=2,
    )

    assert result[0].chunk.content == "attendance policy"