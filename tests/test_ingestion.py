import pytest

from rag_knowledge_system.ingestion import load_text_file

def test_load_text_file(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("Hello RAG", encoding="utf-8")

    document = load_text_file((file_path))

    assert document.content == "Hello RAG"
    assert document.file_type == "txt"
    assert document.metadata["filename"] == "example.txt"


def test_load_text_file_rejects_missing_file(tmp_path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_text_file(file_path)


def test_load_text_file_rejects_empty_file(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_text_file(file_path)


def test_load_text_file_unspported_file_type(tmp_path):
    file_path = tmp_path / "example.csv"
    file_path.write_text("a,b,c", encoding="utf-8")

    with pytest.raises(ValueError):
        load_text_file(file_path)


def test_load_text_markdown_file(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Title\n\nHello RAG", encoding="utf-8")

    document = load_text_file(file_path)

    assert document.content == "# Title\n\nHello RAG"
    assert document.file_type == "md"
    assert document.metadata["filename"] == "notes.md"


def test_load_uppercase_extension(tmp_path):
    file_path = tmp_path / "example.TXT"
    file_path.write_text("Hello RAG", encoding="utf-8")

    document = load_text_file(file_path)

    assert document.file_type == "txt"