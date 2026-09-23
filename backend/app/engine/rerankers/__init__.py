"""Neural and lexical rerankers package."""

from typing import Any

from app.engine.rerankers.base import BaseReranker
from app.engine.rerankers.cross_encoder import CrossEncoderReranker, CrossEncoderRerankerError
from app.engine.rerankers.flashrank import FlashRankReranker, FlashRankRerankerError
from app.engine.rerankers.mock_reranker import DeterministicMockReranker


def get_reranker(
    reranker_type: str = "flashrank",
    **kwargs: Any,
) -> BaseReranker:
    """Factory creating configured passage reranker."""
    reranker_type = reranker_type.lower()
    if reranker_type == "flashrank":
        return FlashRankReranker(**kwargs)
    elif reranker_type == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    elif reranker_type == "mock":
        return DeterministicMockReranker(**kwargs)
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")


__all__ = [
    "BaseReranker",
    "CrossEncoderReranker",
    "CrossEncoderRerankerError",
    "DeterministicMockReranker",
    "FlashRankReranker",
    "FlashRankRerankerError",
    "get_reranker",
]
