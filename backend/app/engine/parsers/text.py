"""Plain text and source code document parser."""

import re
from pathlib import Path
from typing import Any

from app.core.exceptions import DocumentParsingError
from app.engine.parsers.base import BaseDocumentParser
from app.schemas.document import RawDocument


class PlainTextParser(BaseDocumentParser):
    """Parser for plain-text, source code, and unstructured text documents."""

    NON_PRINTABLE_REGEX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

    def parse(self, file_path: Path, **kwargs: Any) -> RawDocument:
        """Read and parse plain text from disk with encoding fallback."""
        if not file_path.exists():
            raise DocumentParsingError(f"File not found: {file_path}")
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                raw_text = file_path.read_text(encoding="latin-1")
            except Exception as e:
                raise DocumentParsingError(f"Failed to read file {file_path}: {e}") from e
        except Exception as e:
            raise DocumentParsingError(f"Error reading file {file_path}: {e}") from e

        domain = kwargs.get("domain", "general")
        file_type = kwargs.get("file_type", "text")
        return self.parse_text(
            raw_text,
            filename=file_path.name,
            domain=domain,
            file_type=file_type,
            metadata={"source_path": str(file_path)},
        )

    def parse_text(
        self,
        text: str,
        filename: str = "in_memory.txt",
        domain: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RawDocument:
        """Normalize and wrap raw text into RawDocument."""
        normalized = self.normalize_text(text)
        cleaned = self.NON_PRINTABLE_REGEX.sub("", normalized)

        meta: dict[str, Any] = dict(metadata or {})
        lines = cleaned.splitlines()
        meta["line_count"] = len(lines)
        meta["char_count"] = len(cleaned)
        meta["word_count"] = len(cleaned.split())

        file_type = kwargs.get("file_type", "text")

        return RawDocument.create(
            filename=filename,
            content=cleaned.strip(),
            domain=domain,
            file_type=file_type,
            metadata=meta,
        )
