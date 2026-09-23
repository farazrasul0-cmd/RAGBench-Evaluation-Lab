"""Dense semantic vector retriever using configured EmbeddingProvider and VectorStoreAdapter."""

from typing import Any

from app.engine.embeddings.base import BaseEmbeddingProvider
from app.engine.retrievers.base import BaseRetriever
from app.schemas.chunk import DocumentChunk
from app.services.vector_store import VectorStoreAdapter


class DenseRetriever(BaseRetriever):
    """Dense vector retriever that embeds queries and queries a VectorStoreAdapter."""

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: VectorStoreAdapter,
        collection_name: str,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.collection_name = collection_name

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[DocumentChunk, float]]:
        """Retrieve top_k chunks via dense embedding similarity."""
        query_vector = self.embedding_provider.embed_query(query)
        return self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
