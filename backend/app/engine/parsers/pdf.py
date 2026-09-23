"""PDF document parser using pypdf."""

from io import BytesIO
from pathlib import Path
from typing import Any

import pypdf

from app.core.exceptions import DocumentParsingError
from app.engine.parsers.base import BaseDocumentParser
from app.schemas.document import RawDocument


class PDFParser(BaseDocumentParser):
    """Parser for PDF files using pypdf with page-level text extraction."""

    def parse(self, file_path: Path, **kwargs: Any) -> RawDocument:
        """Read and parse a PDF file from disk."""
        if not file_path.exists():
            raise DocumentParsingError(f"PDF file not found: {file_path}")
        try:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                return self._extract_from_reader(
                    reader, filename=file_path.name, source_path=str(file_path), **kwargs
                )
        except pypdf.errors.PdfReadError as e:
            raise DocumentParsingError(f"Corrupted or invalid PDF file: {e}") from e
        except DocumentParsingError:
            raise
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse PDF {file_path}: {e}") from e

    def parse_text(
        self,
        text: str,
        filename: str = "in_memory.pdf",
        domain: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RawDocument:
        """Parse raw text representation of a PDF or fallback."""
        norm_text = self.normalize_text(text)
        return RawDocument.create(
            filename=filename,
            content=norm_text.strip(),
            domain=domain,
            file_type="pdf",
            metadata=metadata or {},
        )

    def parse_bytes(
        self,
        data: bytes,
        filename: str = "in_memory.pdf",
        domain: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RawDocument:
        """Parse a PDF directly from in-memory bytes."""
        try:
            reader = pypdf.PdfReader(BytesIO(data))
            return self._extract_from_reader(
                reader, filename=filename, domain=domain, metadata=metadata, **kwargs
            )
        except pypdf.errors.PdfReadError as e:
            raise DocumentParsingError(f"Corrupted or invalid PDF bytes: {e}") from e
        except DocumentParsingError:
            raise
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse PDF bytes: {e}") from e

    def _extract_from_reader(
        self,
        reader: pypdf.PdfReader,
        filename: str,
        source_path: str | None = None,
        domain: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RawDocument:
        """Internal helper to extract text and page metadata from PdfReader."""
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:
                raise DocumentParsingError(
                    f"PDF is encrypted and password was not provided: {e}"
                ) from e

        meta: dict[str, Any] = dict(metadata or {})
        if source_path:
            meta["source_path"] = source_path

        # Document metadata
        info = reader.metadata
        if info:
            meta["author"] = info.author
            meta["title"] = info.title
            meta["creator"] = info.creator

        num_pages = len(reader.pages)
        meta["page_count"] = num_pages

        pages_data: list[dict[str, Any]] = []
        full_text_parts: list[str] = []

        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            norm_page_text = self.normalize_text(page_text).strip()
            pages_data.append(
                {
                    "page_number": idx + 1,
                    "char_count": len(norm_page_text),
                }
            )
            if norm_page_text:
                full_text_parts.append(norm_page_text)

        meta["pages"] = pages_data
        full_content = "\n\n".join(full_text_parts)

        return RawDocument.create(
            filename=filename,
            content=full_content.strip(),
            domain=domain,
            file_type="pdf",
            metadata=meta,
        )
