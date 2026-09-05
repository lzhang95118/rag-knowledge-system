from uuid import UUID, uuid4
from typing import Any, Protocol

from pydantic import BaseModel, Field
from rag_knowledge_system.documents import Document

class Chunk(BaseModel):
    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)



def chunk_document(
        document: Document,
        chunk_size: int,
        overlap: int = 0,
 ) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer")

    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks = []
    step = chunk_size - overlap

    for index, start in enumerate(
        range(0, len(document.content), step)
    ):
        chunk_content = document.content[start:start + chunk_size]

        chunks.append(
            Chunk(
                document_id=document.document_id,
                content=chunk_content,
                chunk_index=index,
                metadata={
                    **document.metadata,
                    "source": document.source,
                    "file_type": document.file_type,
                    "chunk_index": index,
                }
            )
        )

    return chunks


def chunk_document_by_words(
        document: Document,
        chunk_size: int,
        overlap: int = 0,
 ) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = document.content.split()
    chunks = []

    step = chunk_size - overlap

    for index, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start:start + chunk_size]
        chunk_content = " ".join(chunk_words)

        chunks.append(
            Chunk(
                document_id=document.document_id,
                content=chunk_content,
                chunk_index=index,
                metadata={
                    **document.metadata,
                    "source": document.source,
                    "file_type": document.file_type,
                    "chunk_index": index,
                    "chunk_strategy": "word",
                }
            )
        )

    return chunks


class Tokenizer(Protocol):
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        ...


    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        ...


def chunk_document_by_tokens(
        document: Document,
        tokenizer: Tokenizer,
        chunk_size: int,
        overlap: int = 0,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    token_ids = tokenizer.encode(
        document.content,
        add_special_tokens=False,
    )

    chunks = []
    step = chunk_size - overlap

    for index, start in enumerate(
        range(0, len(token_ids), step)
    ):
        chunk_token_ids = token_ids[start:start + chunk_size]

        chunk_content = tokenizer.decode(
            chunk_token_ids,
            skip_special_tokens=True,
        )

        chunks.append(
            Chunk(
                document_id=document.document_id,
                content=chunk_content,
                chunk_index=index,
                metadata={
                    **document.metadata,
                    "source": document.source,
                    "file_type": document.file_type,
                    "chunk_index": index,
                    "chunk_strategy": "token",
                    "tokenizer": tokenizer.__class__.__name__,
                },
            )
        )

    return chunks

