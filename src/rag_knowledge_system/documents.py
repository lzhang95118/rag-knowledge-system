from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    document_id: UUID = Field(default_factory=uuid4)
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    file_type: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
