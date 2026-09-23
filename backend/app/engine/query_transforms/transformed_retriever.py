"""Composable retriever integrating query transformation with multi-channel retrieval."""

from typing import Any

from app.engine.query_transforms.base import BaseQueryTransformer, QueryTransformationResult
from app.engine.retrievers.base import BaseRetriever
from app.engine.retrievers.hybrid import reciprocal_rank_fusion
from app.schemas.chunk import DocumentChunk


class TransformedRetriever(BaseRetriever):
    """Retrieval orchestrator applying query pre-processing prior to passage retrieval.

    For single-query outputs (Identity, HyDE), delegates directly to underlying retriever.
    For multi-query outputs (MultiQuery, StepBack), retrieves top-K for each variant
    and fuses the rankings using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        transformer: BaseQueryTransformer,
        rrf_k: int = 60,
    ) -> None:
        self.retriever = retriever
        self.transformer = transformer
        self.rrf_k = rrf_k
        self.last_transformation: QueryTransformationResult | None = None

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[tuple[DocumentChunk, float]]:
        """Transform query and execute retrieval with RRF fusion for multi-query expansions."""
        trans_res = self.transformer.transform(query=query, **kwargs)
        self.last_transformation = trans_res

        sub_queries = trans_res.transformed_queries
        if not sub_queries:
            return []

        if len(sub_queries) == 1:
            return self.retriever.retrieve(query=sub_queries[0], top_k=top_k, **kwargs)

        # Multi-query expansion: retrieve top candidates for each variant
        candidate_k = max(top_k * 2, 20)
        rankings: list[list[tuple[DocumentChunk, float]]] = []
        for sq in sub_queries:
            sub_res = self.retriever.retrieve(query=sq, top_k=candidate_k, **kwargs)
            rankings.append(sub_res)

        return reciprocal_rank_fusion(rankings=rankings, k=self.rrf_k, top_k=top_k)
