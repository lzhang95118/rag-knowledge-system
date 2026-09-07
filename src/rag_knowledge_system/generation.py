from collections.abc import Callable
from dataclasses import dataclass

from rag_knowledge_system.vector_store import SearchResult


LLMCallable = Callable[[str], str]

UNKNOWN_SOURCE = "unknown"


@dataclass
class Source:
    source_id: int
    source: str
    chunk_index: int
    content: str


@dataclass
class GenerationResult:
    answer: str
    sources: list[Source]


def build_context(results: list[SearchResult]) -> str:
    context_parts = []

    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        source = chunk.metadata.get("source", UNKNOWN_SOURCE)

        context_parts.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"Source: {source}",
                    f"Chunk: {chunk.chunk_index}",
                    f"Content: {chunk.content}",
                ]
            )
        )

    return "\n\n".join(context_parts)


def build_grounded_prompt(
    question: str,
    context: str,
) -> str:
    return f"""You are a source-grounded question answering assistant.

Answer the question using only the supplied context.

Rules:
1. Do not use outside knowledge.
2. Cite supporting information using [Source N].
3. If the supplied context does not contain enough information, say:
   "The answer cannot be determined from the provided sources."

Context:
{context}

Question:
{question}

Answer:"""


class GroundedGenerator:
    def __init__(self, llm: LLMCallable):
        self.llm = llm

    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> GenerationResult:
        if not results:
            return GenerationResult(
                answer=(
                    "The answer cannot be determined "
                    "from the provided sources."
                ),
                sources=[],
            )

        context = build_context(results)

        prompt = build_grounded_prompt(
            question=question,
            context=context,
        )

        answer = self.llm(prompt)

        sources = [
            Source(
                source_id=index,
                source=result.chunk.metadata.get(
                    "source",
                    UNKNOWN_SOURCE,
                ),
                chunk_index=result.chunk.chunk_index,
                content=result.chunk.content,
            )
            for index, result in enumerate(results, start=1)
        ]

        return GenerationResult(
            answer=answer,
            sources=sources,
        )