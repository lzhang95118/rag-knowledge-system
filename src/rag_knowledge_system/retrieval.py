from collections.abc import Mapping
from typing import Any, Mapping

from rag_knowledge_system.embedding import HuggingFaceEmbedding
from rag_knowledge_system.vector_store import InMemoryVectorStore, SearchResult


class Retriever:
    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        embedder: HuggingFaceEmbedding,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        query_vector = self.embedding_model.embed_texts([query])[0]

        return self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
