from rag_knowledge_system.config import DEFAULT_EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer
import numpy as np


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