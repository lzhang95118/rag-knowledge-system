from rag_knowledge_system.embedding import HuggingFaceEmbedding


def test_embed_texts_returns_correct_shape():
    embedder = HuggingFaceEmbedding()

    texts = [
        "hello world",
        "student attendance policy",
    ]

    vectors = embedder.embed_texts(texts)

    assert vectors.shape[0] == 2
    assert vectors.shape[1] == 384

def test_similar_texts_produce_different_vectors():
    embedder = HuggingFaceEmbedding()

    vectors = embedder.embed_texts([
        "The student attended class.",
        "The library is closed.",
    ])

    assert not (vectors[0] == vectors[1]).all()