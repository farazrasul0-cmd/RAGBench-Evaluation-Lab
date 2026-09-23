"""Base reranker interface for RAGBench evaluation laboratory."""

from abc import ABC, abstractmethod
from typing import Any

from app.schemas.chunk import DocumentChunk, RankedChunk


class BaseReranker(ABC):
    """Abstract interface for neural and lexical passage rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[DocumentChunk],
        top_n: int = 5,
        **kwargs: Any,
    ) -> list[tuple[DocumentChunk, float]]:
        """Rerank candidates against the query, returning top_n (chunk, score) pairs."""
        ...

    def rerank_ranked(
        self,
        query: str,
        candidates: list[DocumentChunk],
        top_n: int = 5,
        **kwargs: Any,
    ) -> list[RankedChunk]:
        """Rerank candidates and return structured RankedChunk models with 1-based ranks."""
        pairs = self.rerank(query=query, candidates=candidates, top_n=top_n, **kwargs)
        return [
            RankedChunk(chunk=chunk, score=score, rank=idx)
            for idx, (chunk, score) in enumerate(pairs, start=1)
        ]
