"""Composable retriever decorator combining any BaseRetriever with any BaseReranker."""

from typing import Any

from app.engine.rerankers.base import BaseReranker
from app.engine.retrievers.base import BaseRetriever
from app.schemas.chunk import DocumentChunk


class RerankedRetriever(BaseRetriever):
    """Composable pipeline executing retrieval followed by neural/lexical reranking.

    Enables systematic evaluation across experimental topologies:
    - Dense
    - Dense + Reranker
    - BM25
    - BM25 + Reranker
    - Hybrid
    - Hybrid + Reranker
    without modifying underlying retrieval engines.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        retrieve_top_k: int = 20,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.retrieve_top_k = retrieve_top_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[tuple[DocumentChunk, float]]:
        """Retrieve initial candidates and re-score with configured reranker."""
        candidate_k = max(top_k, self.retrieve_top_k)
        initial_results = self.retriever.retrieve(query=query, top_k=candidate_k, **kwargs)
        candidates = [chunk for chunk, _ in initial_results]
        return self.reranker.rerank(query=query, candidates=candidates, top_n=top_k, **kwargs)
