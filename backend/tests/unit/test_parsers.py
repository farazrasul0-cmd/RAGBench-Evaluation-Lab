"""Unit tests for document parsers."""

from io import BytesIO
from pathlib import Path

import pypdf
import pytest

from app.core.exceptions import DocumentParsingError, UnsupportedFormatError
from app.engine.parsers import (
    MarkdownParser,
    PDFParser,
    PlainTextParser,
    get_parser_for_file,
)


def test_markdown_parser_with_front_matter() -> None:
    """Verify Markdown parser extracts front-matter, headers, and code fences."""
    sample_md = """---
title: "Evaluation Architecture"
version: 1.0
tags: [rag, nlp]
---

# Introduction to RAGBench
RAGBench is an evaluation platform.

## Benchmark Topologies
Dense and sparse retrieval algorithms are tested.

```python
def test_retriever():
    return "hybrid"
```

### Sub-topic
Details here.
"""
    parser = MarkdownParser()
    doc = parser.parse_text(sample_md, filename="sample.md", domain="academic_research")

    assert doc.filename == "sample.md"
    assert doc.domain == "academic_research"
    assert doc.file_type == "markdown"
    assert doc.checksum != ""
    assert doc.metadata["front_matter"]["title"] == "Evaluation Architecture"
    assert doc.metadata["front_matter"]["version"] == 1.0
    assert doc.metadata["code_blocks_count"] == 1
    assert len(doc.metadata["headers"]) == 3
    assert doc.metadata["headers"][0]["title"] == "Introduction to RAGBench"
    assert doc.metadata["headers"][0]["level"] == 1
    assert doc.metadata["headers"][1]["title"] == "Benchmark Topologies"
    assert doc.metadata["headers"][1]["level"] == 2
    # Verify body does not contain front-matter block
    assert "---" not in doc.content


def test_markdown_parser_malformed_front_matter() -> None:
    """Verify malformed YAML in front-matter raises DocumentParsingError."""
    malformed_md = """---
title: [unclosed list
---
Content
"""
    parser = MarkdownParser()
    with pytest.raises(DocumentParsingError, match="Malformed YAML"):
        parser.parse_text(malformed_md)


def test_pdf_parser_in_memory() -> None:
    """Verify PDF parser extracts text and per-page metadata from synthetic PDF."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)

    pdf_buffer = BytesIO()
    writer.write(pdf_buffer)
    pdf_bytes = pdf_buffer.getvalue()

    parser = PDFParser()
    doc = parser.parse_bytes(pdf_bytes, filename="test.pdf")

    assert doc.file_type == "pdf"
    assert doc.metadata["page_count"] == 2
    assert len(doc.metadata["pages"]) == 2
    assert doc.checksum != ""


def test_pdf_parser_corrupt_bytes() -> None:
    """Verify corrupt bytes raise DocumentParsingError."""
    parser = PDFParser()
    with pytest.raises(DocumentParsingError):
        parser.parse_bytes(b"NOT_A_VALID_PDF_HEADER_12345")


def test_plain_text_parser() -> None:
    """Verify plain text parser handles Unicode NFKC, LF line-endings, and control characters."""
    raw_text = "Line 1\r\nLine 2\rLine 3\x00\x07with special chars"
    parser = PlainTextParser()
    doc = parser.parse_text(raw_text, filename="raw.txt")

    assert doc.file_type == "text"
    assert "\r" not in doc.content
    assert "\x00" not in doc.content
    assert doc.metadata["line_count"] == 3
    assert doc.metadata["word_count"] > 0
    assert doc.checksum != ""


def test_parser_factory() -> None:
    """Verify get_parser_for_file returns correct parser and raises on unsupported extensions."""
    assert isinstance(get_parser_for_file(Path("doc.md")), MarkdownParser)
    assert isinstance(get_parser_for_file(Path("paper.pdf")), PDFParser)
    assert isinstance(get_parser_for_file(Path("code.py")), PlainTextParser)
    assert isinstance(get_parser_for_file(Path("notes.txt")), PlainTextParser)

    with pytest.raises(UnsupportedFormatError, match="Unsupported file extension"):
        get_parser_for_file(Path("archive.xyz123"))
