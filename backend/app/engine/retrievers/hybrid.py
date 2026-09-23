"""Hybrid retrieval engine combining dense vector and lexical BM25 search with score fusion."""

from typing import Any

from app.core.exceptions import RAGBenchError
from app.engine.retrievers.base import BaseRetriever
from app.engine.retrievers.bm25 import BM25Retriever
from app.engine.retrievers.dense import DenseRetriever
from app.schemas.chunk import DocumentChunk


class HybridRetrieverError(RAGBenchError):
    """Raised when hybrid retrieval or fusion fails."""


def reciprocal_rank_fusion(
    rankings: list[list[tuple[DocumentChunk, float]]],
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[DocumentChunk, float]]:
    """Reciprocal Rank Fusion (RRF) algorithm:

    RRF(d) = sum_{m in M} 1 / (k + r_m(d))
    where r_m(d) is the 1-based rank position of document d in ranking m.
    """
    if k < 0:
        raise HybridRetrieverError("RRF constant k must be non-negative.")

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, DocumentChunk] = {}

    for ranking in rankings:
        for rank_idx, (chunk, _) in enumerate(ranking, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            score_increment = 1.0 / (k + rank_idx)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_increment

    fused_results = [(chunk_map[cid], score) for cid, score in rrf_scores.items()]
    # Deterministic sort: descending by score, tie-break by ascending chunk_id
    fused_results.sort(key=lambda x: (-x[1], x[0].chunk_id))
    return fused_results[:top_k]


def relative_score_normalization(
    dense_results: list[tuple[DocumentChunk, float]],
    bm25_results: list[tuple[DocumentChunk, float]],
    alpha: float = 0.5,
    top_k: int = 10,
    epsilon: float = 1e-9,
) -> list[tuple[DocumentChunk, float]]:
    """Relative Score Normalization (RSN):

    s_norm(d) = (s(d) - min_s) / (max_s - min_s + epsilon)
    S_final(d) = alpha * s_dense_norm(d) + (1 - alpha) * s_bm25_norm(d)
    """
    if not 0.0 <= alpha <= 1.0:
        raise HybridRetrieverError("Alpha weighting must be between 0.0 and 1.0 inclusive.")

    if not dense_results and not bm25_results:
        return []

    def normalize(
        results: list[tuple[DocumentChunk, float]],
    ) -> dict[str, float]:
        if not results:
            return {}
        scores = [score for _, score in results]
        min_s = min(scores)
        max_s = max(scores)
        span = max_s - min_s
        if span <= epsilon:
            # If all scores are equal, treat positive scores as 1.0, non-positive as 0.0
            val = 1.0 if max_s > 0.0 else 0.0
            return {chunk.chunk_id: val for chunk, _ in results}
        return {chunk.chunk_id: (score - min_s) / (span + epsilon) for chunk, score in results}

    dense_norm = normalize(dense_results)
    bm25_norm = normalize(bm25_results)

    chunk_map: dict[str, DocumentChunk] = {}
    for chunk, _ in dense_results:
        chunk_map[chunk.chunk_id] = chunk
    for chunk, _ in bm25_results:
        chunk_map[chunk.chunk_id] = chunk

    combined_scores: list[tuple[DocumentChunk, float]] = []
    for cid, chunk in chunk_map.items():
        s_dense = dense_norm.get(cid, 0.0)
        s_bm25 = bm25_norm.get(cid, 0.0)
        final_score = alpha * s_dense + (1.0 - alpha) * s_bm25
        combined_scores.append((chunk, final_score))

    # Deterministic sort: descending by score, tie-break by ascending chunk_id
    combined_scores.sort(key=lambda x: (-x[1], x[0].chunk_id))
    return combined_scores[:top_k]


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval orchestrator executing dense and lexical channels with score fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
    ) -> None:
        if fusion_method not in ("rrf", "rsn"):
            raise HybridRetrieverError(f"Unsupported fusion method: {fusion_method}")
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        candidate_multiplier: int = 2,
        **kwargs: Any,
    ) -> list[tuple[DocumentChunk, float]]:
        """Retrieve candidates from dense and BM25 channels and fuse rankings."""
        num_candidates = max(top_k * candidate_multiplier, 20)
        dense_candidates = self.dense_retriever.retrieve(
            query=query,
            top_k=num_candidates,
            filter_metadata=filter_metadata,
            **kwargs,
        )
        bm25_candidates = self.bm25_retriever.retrieve(
            query=query,
            top_k=num_candidates,
            filter_metadata=filter_metadata,
            **kwargs,
        )

        if self.fusion_method == "rrf":
            return reciprocal_rank_fusion(
                rankings=[dense_candidates, bm25_candidates],
                k=self.rrf_k,
                top_k=top_k,
            )
        else:
            return relative_score_normalization(
                dense_results=dense_candidates,
                bm25_results=bm25_candidates,
                alpha=self.alpha,
                top_k=top_k,
            )
