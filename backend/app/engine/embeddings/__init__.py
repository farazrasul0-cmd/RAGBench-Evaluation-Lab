"""Embeddings package for RAGBench."""

from typing import Any

from app.core.exceptions import RAGBenchError
from app.engine.embeddings.base import BaseEmbeddingProvider, EmbeddingError
from app.engine.embeddings.cloud_provider import CloudEmbeddingProvider
from app.engine.embeddings.fastembed_provider import FastEmbedProvider
from app.engine.embeddings.mock_provider import DeterministicMockEmbeddingProvider
from app.engine.embeddings.sentence_trans import SentenceTransformersProvider

__all__ = [
    "BaseEmbeddingProvider",
    "CloudEmbeddingProvider",
    "DeterministicMockEmbeddingProvider",
    "EmbeddingError",
    "FastEmbedProvider",
    "SentenceTransformersProvider",
    "get_embedding_provider",
]


def get_embedding_provider(
    provider: str,
    model_name: str = "BAAI/bge-small-en-v1.5",
    **kwargs: Any,
) -> BaseEmbeddingProvider:
    """Factory creating an embedding provider based on configuration."""
    prov = provider.lower().strip()
    if prov in ["fastembed", "onnx", "local"]:
        return FastEmbedProvider(model_name=model_name, **kwargs)
    if prov in ["sentence_transformers", "sentence-transformers", "pytorch"]:
        return SentenceTransformersProvider(model_name=model_name, **kwargs)
    if prov in ["cloud", "openai", "voyage", "cohere", "litellm"]:
        return CloudEmbeddingProvider(model_name=model_name, **kwargs)
    if prov in ["mock", "test", "deterministic"]:
        return DeterministicMockEmbeddingProvider(model_name=model_name, **kwargs)
    raise RAGBenchError(f"Unknown embedding provider: {provider}")
