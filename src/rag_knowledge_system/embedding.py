from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_knowledge_system.config import DEFAULT_EMBEDDING_MODEL

class Embedder(Protocol):
    def embed_texts(
        self,
        texts: list[str],
    ) -> np.ndarray:
        ...

class HuggingFaceEmbedding:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(
        self,
        texts: list[str],
    ) -> np.ndarray:
        vectors = self.model.encode(texts)
        return vectors