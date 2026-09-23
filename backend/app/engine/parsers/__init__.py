"""Parsers package for RAGBench."""

from pathlib import Path

from app.core.exceptions import UnsupportedFormatError
from app.engine.parsers.base import BaseDocumentParser
from app.engine.parsers.markdown import MarkdownParser
from app.engine.parsers.pdf import PDFParser
from app.engine.parsers.text import PlainTextParser

__all__ = [
    "BaseDocumentParser",
    "MarkdownParser",
    "PDFParser",
    "PlainTextParser",
    "get_parser_for_file",
]


def get_parser_for_file(file_path: Path) -> BaseDocumentParser:
    """Return appropriate parser instance according to file extension."""
    suffix = file_path.suffix.lower()
    if suffix in [".md", ".markdown", ".mdown"]:
        return MarkdownParser()
    if suffix == ".pdf":
        return PDFParser()
    if suffix in [".txt", ".log", ".json", ".yaml", ".yml", ".py", ".ts", ".tsx", ".js"]:
        return PlainTextParser()
    raise UnsupportedFormatError(f"Unsupported file extension: {suffix} for file {file_path.name}")
