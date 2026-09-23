"""Abstract base parser interface for RAGBench."""

import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.schemas.document import RawDocument


class BaseDocumentParser(ABC):
    """Abstract interface for all document format parsers."""

    @abstractmethod
    def parse(self, file_path: Path, **kwargs: Any) -> RawDocument:
        """Parse a document from a local file path."""

    @abstractmethod
    def parse_text(
        self,
        text: str,
        filename: str = "in_memory_doc",
        domain: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RawDocument:
        """Parse raw text into a RawDocument object."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize unicode representation and unify line-endings to LF."""
        normalized = unicodedata.normalize("NFKC", text)
        return normalized.replace("\r\n", "\n").replace("\r", "\n")
