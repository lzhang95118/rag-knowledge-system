import pytest
from pydantic import ValidationError

from rag_knowledge_system.documents import Document


def test_document_creation():
    document = Document(
        content="Example document content",
        source="example.txt",
        file_type="txt",
        metadata={"author": "test"},
    )

    assert document.content == "Example document content"
    assert document.source == "example.txt"
    assert document.file_type == "txt"
    assert document.metadata["author"] == "test"
    assert document.document_id is not None

def test_document_rejects_empty_content():
    with pytest.raises(ValidationError):
        Document(
            content="",
            source="example.txt",
            file_type="txt",
        )

