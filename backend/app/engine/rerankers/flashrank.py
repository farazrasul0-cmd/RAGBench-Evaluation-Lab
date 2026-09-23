"""FlashRank lightweight ONNX cross-encoder reranker."""

from typing import Any

from flashrank import Ranker, RerankRequest

from app.core.exceptions import RAGBenchError
from app.engine.rerankers.base import BaseReranker
from app.schemas.chunk import DocumentChunk


class FlashRankRerankerError(RAGBenchError):
    """Raised when FlashRank reranking fails."""


class FlashRankReranker(BaseReranker):
    """CPU-optimized neural reranker powered by FlashRank ONNX runtime."""

    def __init__(
        self,
        model_name: str = "ms-marco-TinyBERT-L-2-v2",
        cache_dir: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        kwargs: dict[str, Any] = {
            "model_name": model_name,
            "max_length": max_length,
        }
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        try:
            self._ranker = Ranker(**kwargs)
        except Exception as e:
            raise FlashRankRerankerError(f"Failed to initialize FlashRank ranker: {e}") from e

    def rerank(
        self,
        query: str,
        candidates: list[DocumentChunk],
        top_n: int = 5,
        **kwargs: Any,
    ) -> list[tuple[DocumentChunk, float]]:
        """Rerank candidates using cross-attention scoring in FlashRank."""
        if not candidates:
            return []

        chunk_lookup: dict[str, DocumentChunk] = {c.chunk_id: c for c in candidates}
        passages: list[dict[str, Any]] = [{"id": c.chunk_id, "text": c.content} for c in candidates]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(rerank_request)

        scored: list[tuple[DocumentChunk, float]] = []
        for item in results:
            cid = str(item["id"])
            score = float(item["score"])
            if cid in chunk_lookup:
                scored.append((chunk_lookup[cid], score))

        # Deterministic sort: descending by score, tie-break by ascending chunk_id
        scored.sort(key=lambda x: (-x[1], x[0].chunk_id))
        return scored[:top_n]
