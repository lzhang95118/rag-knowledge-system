import pytest
from rag_knowledge_system.documents import Document
from rag_knowledge_system.chunking import (
    chunk_document, 
    chunk_document_by_words, 
    chunk_document_by_tokens,
    chunk_document_by_paragraphs,
)


def test_chunk_document():
    document = Document(
        content="ABCDEFGHIJ",
        source="example.txt",
        file_type="txt",
    )

    chunk = chunk_document(
        document=document,
        chunk_size=4,
    )

    assert len(chunk) == 3
    assert chunk[0].content == "ABCD"
    assert chunk[1].content == "EFGH"
    assert chunk[2].content == "IJ"


    assert chunk[0].chunk_index == 0
    assert chunk[1].chunk_index == 1
    assert chunk[2].chunk_index == 2


    assert chunk[0].document_id == document.document_id


def test_chunk_document_with_overlap():
    document = Document(
        content="ABCDEFGHIJ",
        source="example.txt",
        file_type="txt",
    )

    chunk = chunk_document(
        document=document,
        chunk_size=4,
        overlap=1,
    )


    assert chunk[0].content == "ABCD"
    assert chunk[1].content == "DEFG"
    assert chunk[2].content == "GHIJ"


def test_chunk_size_must_be_positive():
    document = Document(
        content="AB",
        source="example.txt",
        file_type="txt",
    )

    with pytest.raises(ValueError):
        chunk_document(document,chunk_size=0)


def test_overlap_cannot_be_negative():
    document = Document(
        content="ABC",
        source="example.txt",
        file_type="txt",
    )

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=3, overlap=-1)


def test_overlap_must_be_smaller_than_chunk_size():
    document = Document(
        content="ABC",
        source="example.txt",
        file_type="txt",
    )

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=3, overlap=3)


def test_chunk_inherits_document_metadata():
    document = Document(
        content="ABCDEFGHIJ",
        source="example.txt",
        file_type="txt",
        metadata={"author": "test"},
    )

    chunks = chunk_document(
        document=document,
        chunk_size=4,
    )

    assert chunks[0].metadata["author"] == "test"
    assert chunks[0].metadata["source"] == "example.txt"
    assert chunks[0].metadata["file_type"] == "txt"
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunk_document_by_words():
    document = Document(
        content="one two three four five six seven",
        source="example.txt",
        file_type="txt",
    )

    chunks = chunk_document_by_words(
        document=document,
        chunk_size=3,
        overlap=1,
    )

    assert chunks[0].content == "one two three"
    assert chunks[1].content == "three four five"
    assert chunks[2].content == "five six seven"


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"token{i}" for i in token_ids)



def test_chunk_document_by_tokens():
    document = Document(
        content="one two three four five six",
        source="example.txt",
        file_type="txt",
    )

    tokenizer = FakeTokenizer()

    chunks = chunk_document_by_tokens(
        document=document,
        tokenizer=tokenizer,
        chunk_size=3,
        overlap=1,
    )

    assert len(chunks) == 3
    assert chunks[0].content == "token0 token1 token2"
    assert chunks[1].content == "token2 token3 token4"
    assert chunks[2].content == "token4 token5"
    assert chunks[0].metadata["chunk_strategy"] == "token"


def test_chunk_document_by_paragraphs():
    document = Document(
        content="""
Students must attend scheduled classes.

The library opens at 8 am and closes at 10 pm.

Attendance below the required threshold may trigger follow-up.
""".strip(),
        source="sample.txt",
        file_type="txt",
    )

    chunks = chunk_document_by_paragraphs(
        document=document,
        max_words=100,
    )

    assert len(chunks) == 3

    assert chunks[0].content == "Students must attend scheduled classes."
    assert chunks[1].content == "The library opens at 8 am and closes at 10 pm."
    assert chunks[2].content == "Attendance below the required threshold may trigger follow-up."