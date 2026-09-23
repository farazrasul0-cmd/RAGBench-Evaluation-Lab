"""Sentence-Transformers CrossEncoder reranker with lazy-loading."""

import importlib
from typing import Any

from app.core.exceptions import RAGBenchError
from app.engine.rerankers.base import BaseReranker
from app.schemas.chunk import DocumentChunk


class CrossEncoderRerankerError(RAGBenchError):
    """Raised when CrossEncoder reranking fails."""


class CrossEncoderReranker(BaseReranker):
    """PyTorch/HuggingFace Cross-Encoder reranker."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                st_module = importlib.import_module("sentence_transformers")
                cross_encoder_cls = st_module.CrossEncoder
                kwargs: dict[str, Any] = {"max_length": self.max_length}
                if self.device:
                    kwargs["device"] = self.device
                self._model = cross_encoder_cls(self.model_name, **kwargs)
            except Exception as e:
                raise CrossEncoderRerankerError(
                    f"Failed to load sentence_transformers CrossEncoder: {e}"
                ) from e
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[DocumentChunk],
        top_n: int = 5,
        **kwargs: Any,
    ) -> list[tuple[DocumentChunk, float]]:
        """Rerank candidates using cross-encoder relevance logits."""
        if not candidates:
            return []

        model = self._get_model()
        pairs = [[query, c.content] for c in candidates]
        scores = model.predict(pairs)

        scored: list[tuple[DocumentChunk, float]] = [
            (chunk, float(score)) for chunk, score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda x: (x[1], x[0].chunk_id), reverse=True)
        return scored[:top_n]
