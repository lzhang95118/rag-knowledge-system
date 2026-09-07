from uuid import uuid4

from rag_knowledge_system.chunking import Chunk
from rag_knowledge_system.generation import (
    GroundedGenerator,
    build_context,
    build_grounded_prompt,
)
from rag_knowledge_system.vector_store import SearchResult


def make_result(
    content: str,
    source: str,
    chunk_index: int = 0,
    score: float = 0.9,
) -> SearchResult:
    chunk = Chunk(
        document_id=uuid4(),
        content=content,
        chunk_index=chunk_index,
        metadata={
            "source": source,
        },
    )

    return SearchResult(
        chunk=chunk,
        score=score,
    )


def test_build_context_formats_sources():
    results = [
        make_result(
            content="RAG combines retrieval with generation.",
            source="rag_basics.txt",
            chunk_index=0,
        ),
        make_result(
            content="Retrieved documents provide context to the model.",
            source="retrieval.txt",
            chunk_index=1,
        ),
    ]

    context = build_context(results)

    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "rag_basics.txt" in context
    assert "retrieval.txt" in context
    assert "Chunk: 0" in context
    assert "Chunk: 1" in context
    assert "RAG combines retrieval with generation." in context


def test_build_grounded_prompt_contains_question_and_context():
    question = "What is RAG?"
    context = (
        "[Source 1]\n"
        "Source: rag_basics.txt\n"
        "Content: RAG combines retrieval and generation."
    )

    prompt = build_grounded_prompt(
        question=question,
        context=context,
    )

    assert question in prompt
    assert context in prompt
    assert "Do not use outside knowledge." in prompt
    assert "[Source N]" in prompt


def test_generator_returns_grounded_answer_and_sources():
    def fake_llm(prompt: str) -> str:
        assert "RAG combines retrieval with generation." in prompt

        return (
            "RAG combines retrieval with generation "
            "[Source 1]."
        )

    results = [
        make_result(
            content="RAG combines retrieval with generation.",
            source="rag_basics.txt",
            chunk_index=2,
        )
    ]

    generator = GroundedGenerator(llm=fake_llm)

    result = generator.generate(
        question="What is RAG?",
        results=results,
    )

    assert (
        result.answer
        == "RAG combines retrieval with generation [Source 1]."
    )

    assert len(result.sources) == 1

    source = result.sources[0]

    assert source.source_id == 1
    assert source.source == "rag_basics.txt"
    assert source.chunk_index == 2
    assert source.content == (
        "RAG combines retrieval with generation."
    )


def test_generator_handles_empty_results_without_calling_llm():
    def fake_llm(prompt: str) -> str:
        raise AssertionError("LLM should not be called")

    generator = GroundedGenerator(llm=fake_llm)

    result = generator.generate(
        question="What is quantum gravity?",
        results=[],
    )

    assert (
        result.answer
        == "The answer cannot be determined from the provided sources."
    )
    assert result.sources == []