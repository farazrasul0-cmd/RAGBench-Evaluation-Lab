"""Deterministic mock reranker for testing and offline execution."""

from typing import Any

from app.engine.rerankers.base import BaseReranker
from app.schemas.chunk import DocumentChunk


class DeterministicMockReranker(BaseReranker):
    """Fast deterministic reranker scoring candidates based on keyword match and length."""

    def __init__(self, score_mapping: dict[str, float] | None = None) -> None:
        self.score_mapping = score_mapping or {}

    def rerank(
        self,
        query: str,
        candidates: list[DocumentChunk],
        top_n: int = 5,
        **kwargs: Any,
    ) -> list[tuple[DocumentChunk, float]]:
        """Score each candidate deterministically."""
        if not candidates:
            return []

        query_terms = set(query.lower().split())
        scored: list[tuple[DocumentChunk, float]] = []

        for candidate in candidates:
            if candidate.chunk_id in self.score_mapping:
                score = self.score_mapping[candidate.chunk_id]
            else:
                text_terms = set(candidate.content.lower().split())
                overlap = len(query_terms.intersection(text_terms))
                # Normalized pseudo-score in [0.0, 1.0]
                score = float(overlap) / (len(query_terms) + 1.0)
            scored.append((candidate, score))

        scored.sort(key=lambda x: (x[1], x[0].chunk_id), reverse=True)
        return scored[:top_n]
