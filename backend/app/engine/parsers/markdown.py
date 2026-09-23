"""Markdown document parser with YAML front-matter extraction."""

import re
from pathlib import Path
from typing import Any

import yaml

from app.core.exceptions import DocumentParsingError
from app.engine.parsers.base import BaseDocumentParser
from app.schemas.document import RawDocument


class MarkdownParser(BaseDocumentParser):
    """Parser for Markdown documents with front-matter and structural metadata extraction."""

    FRONT_MATTER_REGEX = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    HEADER_REGEX = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_FENCE_REGEX = re.compile(r"```[a-zA-Z0-9_-]*\n.*?\n```", re.DOTALL)

    def parse(self, file_path: Path, **kwargs: Any) -> RawDocument:
        """Read and parse a Markdown file from disk."""
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
        return self.parse_text(
            raw_text,
            filename=file_path.name,
            domain=domain,
            metadata={"source_path": str(file_path)},
        )

    def parse_text(
        self,
        text: str,
        filename: str = "in_memory.md",
        domain: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RawDocument:
        """Parse Markdown string, extracting front-matter and structural hierarchy."""
        normalized = self.normalize_text(text)
        meta: dict[str, Any] = dict(metadata or {})
        front_matter: dict[str, Any] = {}
        body = normalized

        # 1. Front-matter extraction
        match = self.FRONT_MATTER_REGEX.match(normalized)
        if match:
            fm_text = match.group(1)
            try:
                loaded = yaml.safe_load(fm_text)
                front_matter = loaded if isinstance(loaded, dict) else {"raw_front_matter": loaded}
            except Exception as e:
                raise DocumentParsingError(f"Malformed YAML front-matter: {e}") from e
            body = normalized[match.end() :]

        meta["front_matter"] = front_matter

        # 2. Header extraction
        headers: list[dict[str, Any]] = []
        for h_match in self.HEADER_REGEX.finditer(body):
            level = len(h_match.group(1))
            title = h_match.group(2).strip()
            headers.append({"level": level, "title": title, "char_offset": h_match.start()})
        meta["headers"] = headers

        # 3. Code fence statistics
        code_blocks = list(self.CODE_FENCE_REGEX.finditer(body))
        meta["code_blocks_count"] = len(code_blocks)

        return RawDocument.create(
            filename=filename,
            content=body.strip(),
            domain=domain,
            file_type="markdown",
            metadata=meta,
        )
