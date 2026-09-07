from rag_knowledge_system.ingestion import load_text_file
from rag_knowledge_system.retrieval import Retriever
from rag_knowledge_system.chunking import (
    chunk_document_by_words,
    chunk_document_by_paragraphs,
)
from rag_knowledge_system.embedding import HuggingFaceEmbedding
from rag_knowledge_system.vector_store import InMemoryVectorStore
from rag_knowledge_system.generation import (
    GenerationResult,
    GroundedGenerator,
)


def build_retriever_from_file(
    file_path: str,
    chunk_size: int = 50,
    overlap: int = 10,
) -> Retriever:

    document = load_text_file(file_path)

    chunks = chunk_document_by_words(
        document=document,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    embedder = HuggingFaceEmbedding()

    embeddings = embedder.embed_texts(
        [chunk.content for chunk in chunks]
    )

    store = InMemoryVectorStore()
    store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=store,
    )

    return retriever


def build_retriever_from_file_with_paragraphs(
    file_path: str,
    max_words: int = 100,
    overlap: int = 0,
) -> Retriever:

    document = load_text_file(file_path)

    chunks = chunk_document_by_paragraphs(
        document=document,
        max_words=max_words,
        overlap=overlap,
    )

    embedder = HuggingFaceEmbedding()

    embeddings = embedder.embed_texts(
        [chunk.content for chunk in chunks]
    )

    store = InMemoryVectorStore()
    store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=store,
    )

    return retriever

def answer_question(
    question: str,
    retriever: Retriever,
    generator: GroundedGenerator,
    top_k: int = 3,
) -> GenerationResult:
    results = retriever.retrieve(
        query=question,
        top_k=top_k,
    )

    return generator.generate(
        question=question,
        results=results,
    )
