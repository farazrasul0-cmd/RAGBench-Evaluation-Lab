"""Document chunk schema with deterministic cryptographic hashing."""

import hashlib
import unicodedata
from typing import Any

from pydantic import BaseModel, Field


def generate_chunk_id(doc_id: str, chunk_index: int, content: str) -> str:
    """Generate deterministic SHA-256 chunk ID from doc_id, index, and normalized content."""
    normalized_content = unicodedata.normalize("NFKC", content.strip())
    payload = f"{doc_id}:{chunk_index}:{normalized_content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DocumentChunk(BaseModel):
    """Represents a discrete segmented passage derived from a RawDocument."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    content: str
    token_count: int
    char_count: int
    start_char: int
    end_char: int
    strategy: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        doc_id: str,
        chunk_index: int,
        content: str,
        token_count: int,
        start_char: int,
        end_char: int,
        strategy: str,
        metadata: dict[str, Any] | None = None,
        chunk_id: str | None = None,
    ) -> "DocumentChunk":
        """Factory method computing deterministic SHA-256 chunk_id."""
        cid = chunk_id or generate_chunk_id(doc_id, chunk_index, content)
        return cls(
            chunk_id=cid,
            doc_id=doc_id,
            chunk_index=chunk_index,
            content=content,
            token_count=token_count,
            char_count=len(content),
            start_char=start_char,
            end_char=end_char,
            strategy=strategy,
            metadata=metadata or {},
        )
