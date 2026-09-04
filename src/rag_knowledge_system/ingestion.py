from pathlib import Path

from rag_knowledge_system.documents import Document
from rag_knowledge_system.config import SUPPORTED_TEXT_TYPES


def load_text_file(file_path: str | Path) -> Document:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_type = path.suffix.lstrip(".").lower()

    if file_type not in SUPPORTED_TEXT_TYPES:
        raise ValueError(f"Unsupported file type: {file_type}")

    content = path.read_text(encoding="utf-8")

    if not content.strip():
        raise ValueError(f"File is empty: {file_path}")

    return Document(
        content=content,
        source=str(path),
        file_type=file_type,
        metadata={
            "filename": path.name,
        },
    )