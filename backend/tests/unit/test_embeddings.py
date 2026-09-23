"""Unit tests for embedding providers and vector calculations."""

import math

import pytest

from app.engine.embeddings import (
    DeterministicMockEmbeddingProvider,
    FastEmbedProvider,
    get_embedding_provider,
)
from app.engine.embeddings.base import EmbeddingError


def test_mock_embedding_provider_properties() -> None:
    """Verify mock provider dimensions and deterministic output."""
    provider = DeterministicMockEmbeddingProvider(dimension=128)
    assert provider.dimension == 128
    assert "mock" in provider.model_name

    text = "Retrieval-Augmented Generation evaluation laboratory."
    vec1 = provider.embed_query(text)
    vec2 = provider.embed_query(text)

    assert len(vec1) == 128
    assert vec1 == vec2  # 100% deterministic

    # Verify unit normalization: norm == 1.0
    norm = math.sqrt(sum(x * x for x in vec1))
    assert abs(norm - 1.0) < 1e-5


def test_mock_embedding_batch_and_empty() -> None:
    """Verify batch embedding and empty text handling."""
    provider = DeterministicMockEmbeddingProvider(dimension=64)
    texts = ["Document one", "Document two", "Document three"]
    vectors = provider.embed_texts(texts)

    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 64

    assert provider.embed_texts([]) == []


def test_cosine_ranking_orthogonal_vectors() -> None:
    """Verify exact cosine distance ranking on synthetic orthogonal vectors."""
    u = [1.0, 0.0, 0.0, 0.0]
    v = [0.0, 1.0, 0.0, 0.0]
    w = [1.0, 0.0, 0.0, 0.0]

    def cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b)

    assert abs(cosine_sim(u, w) - 1.0) < 1e-9
    assert abs(cosine_sim(u, v) - 0.0) < 1e-9


def test_embedding_factory() -> None:
    """Verify get_embedding_provider returns appropriate instances."""
    mock_prov = get_embedding_provider("mock", dimension=256)
    assert isinstance(mock_prov, DeterministicMockEmbeddingProvider)
    assert mock_prov.dimension == 256

    fast_prov = get_embedding_provider("fastembed", model_name="BAAI/bge-small-en-v1.5")
    assert isinstance(fast_prov, FastEmbedProvider)
    assert fast_prov.dimension == 384


def test_fastembed_provider_dimension() -> None:
    """Verify FastEmbedProvider configuration and dimension lookup."""
    prov = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")
    assert prov.dimension == 384
    assert prov.model_name == "BAAI/bge-small-en-v1.5"


def test_embedding_validation_failure() -> None:
    """Verify validate_dimensions raises EmbeddingError when vector size is wrong."""
    prov = DeterministicMockEmbeddingProvider(dimension=100)
    with pytest.raises(EmbeddingError, match="expected 100"):
        prov.validate_dimensions([[1.0, 2.0, 3.0]])
