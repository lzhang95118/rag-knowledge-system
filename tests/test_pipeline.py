from unittest.mock import Mock

from rag_knowledge_system.generation import (
    GenerationResult,
    GroundedGenerator,
)
from rag_knowledge_system.pipeline import (
    answer_question,
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


def test_answer_question_connects_retrieval_and_generation():
    search_results = [
        Mock(),
    ]

    retriever = Mock()
    retriever.retrieve.return_value = search_results

    expected_result = GenerationResult(
        answer="RAG combines retrieval and generation [Source 1].",
        sources=[],
    )

    generator = Mock(spec=GroundedGenerator)
    generator.generate.return_value = expected_result

    result = answer_question(
        question="What is RAG?",
        retriever=retriever,
        generator=generator,
        top_k=2,
    )

    retriever.retrieve.assert_called_once_with(
        query="What is RAG?",
        top_k=2,
    )

    generator.generate.assert_called_once_with(
        question="What is RAG?",
        results=search_results,
    )

    assert result == expected_result

def test_source_grounded_generation_pipeline(tmp_path):
    file_path = tmp_path / "rag_basics.txt"
    file_path.write_text(
        "Retrieval-augmented generation combines retrieval "
        "with language model generation.",
        encoding="utf-8",
    )

    retriever = build_retriever_from_file(
        str(file_path),
        chunk_size=50,
        overlap=0,
    )

    def fake_llm(prompt: str) -> str:
        assert "Retrieval-augmented generation" in prompt
        assert "[Source 1]" in prompt
        assert "rag_basics.txt" in prompt

        return (
            "Retrieval-augmented generation combines retrieval "
            "with language model generation [Source 1]."
        )

    generator = GroundedGenerator(llm=fake_llm)

    result = answer_question(
        question="What is retrieval-augmented generation?",
        retriever=retriever,
        generator=generator,
        top_k=1,
    )

    assert "[Source 1]" in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].source.endswith("rag_basics.txt")
