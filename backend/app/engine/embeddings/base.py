"""Abstract base embedding provider interface."""

from abc import ABC, abstractmethod

from app.core.exceptions import RAGBenchError


class EmbeddingError(RAGBenchError):
    """Raised when generating embeddings fails."""


class BaseEmbeddingProvider(ABC):
    """Abstract base class for all embedding representation providers."""

    def __init__(self, model_name: str, dimension: int) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        """Name or HuggingFace ID of the embedding model."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Dimensionality of the dense output vectors."""
        return self._dimension

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embedding vectors for a batch of text passages."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate a dense embedding vector for a single search query."""

    def validate_dimensions(self, embeddings: list[list[float]]) -> None:
        """Validate that all generated vectors match expected dimension."""
        for idx, vec in enumerate(embeddings):
            if len(vec) != self._dimension:
                raise EmbeddingError(
                    f"Vector at index {idx} has dimension {len(vec)}, expected {self._dimension}"
                )
