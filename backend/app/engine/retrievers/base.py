"""Base retriever interface for RAGBench evaluation laboratory."""

from abc import ABC, abstractmethod
from typing import Any

from app.schemas.chunk import DocumentChunk


class BaseRetriever(ABC):
    """Abstract base class for all retrieval engines."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[tuple[DocumentChunk, float]]:
        """Retrieve top_k chunks for a given query along with relevance/similarity scores.

        Returns list of (DocumentChunk, score) tuples sorted in descending order of score.
        """
