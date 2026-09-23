"""Domain-specific exceptions for RAGBench."""


class RAGBenchError(Exception):
    """Base exception for all RAGBench domain errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DocumentParsingError(RAGBenchError):
    """Raised when parsing a document fails due to corrupt, unreadable, or invalid content."""


class UnsupportedFormatError(DocumentParsingError):
    """Raised when an unsupported file format or extension is encountered."""


class ChunkingError(RAGBenchError):
    """Raised when text segmentation or chunking fails or invalid parameters are provided."""
