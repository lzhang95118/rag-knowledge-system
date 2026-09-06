from rag_knowledge_system.pipeline import (
    build_retriever_from_file,
    build_retriever_from_file_with_paragraphs,
)

def test_build_retriever_from_file(tmp_path):
    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        """
Students must attend scheduled classes.

The library opens at 8 am and closes at 10 pm.

Attendance below the required threshold may trigger follow-up.
""".strip(),
        encoding="utf-8"
    )

    retriever = build_retriever_from_file(
        file_path=file_path,
        chunk_size=8,
        overlap=2,
    )

    results = retriever.retrieve(
        "student attendance",
        top_k=1,
    )

    assert results[0].chunk.chunk_index == 0



def test_build_retriever_from_file_with_paragraphs(tmp_path):
    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        """
Students must attend scheduled classes.

The library opens at 8 am and closes at 10 pm.

Attendance below the required threshold may trigger follow-up.
""".strip(),
            encoding="utf-8"
    )

    retriever = build_retriever_from_file_with_paragraphs(
        file_path=file_path,
        max_words=100,
        overlap=0,
    )

    results = retriever.retrieve(
        "student attendance",
        top_k=3,
    )

    top_result = results[0]

    assert top_result.chunk.chunk_index in [0, 2]
    assert "library" not in top_result.chunk.content.lower()