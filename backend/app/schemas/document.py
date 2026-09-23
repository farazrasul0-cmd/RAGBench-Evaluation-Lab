"""Raw document schemas with cryptographic checksums."""

import hashlib
import unicodedata
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def compute_checksum(content: str) -> str:
    """Compute SHA-256 checksum of normalized text content."""
    normalized = unicodedata.normalize("NFKC", content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class RawDocument(BaseModel):
    """Represents an ingested raw document before chunking."""

    doc_id: str
    filename: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    domain: str = "general"
    file_type: str = "text"
    checksum: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, __context: Any) -> None:
        """Ensure checksum is calculated if not explicitly provided."""
        if not self.checksum:
            self.checksum = compute_checksum(self.content)

    @classmethod
    def create(
        cls,
        filename: str,
        content: str,
        doc_id: str | None = None,
        domain: str = "general",
        file_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> "RawDocument":
        """Factory constructor ensuring deterministic doc_id and checksum."""
        checksum = compute_checksum(content)
        final_doc_id = doc_id or f"doc_{checksum[:16]}"
        return cls(
            doc_id=final_doc_id,
            filename=filename,
            content=content,
            metadata=metadata or {},
            domain=domain,
            file_type=file_type,
            checksum=checksum,
        )
