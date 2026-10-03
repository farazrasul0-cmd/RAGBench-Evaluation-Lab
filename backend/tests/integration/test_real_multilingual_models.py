"""Real multilingual model integration sanity tests (Phase G Amendment 3).

NOTE (Supervisor Amendment 3):
    This test is a provider/model integration sanity test — it verifies that
    the embedding provider and model weights load, output the correct dimensionality,
    execute deterministically, and perform a simple sanity retrieval on a 5-document
    corpus. It is explicitly NOT presented as evidence of general embedding quality
    or empirical benchmark superiority.

Marked `@pytest.mark.slow` and skipped by default in CI unless RUN_SLOW_TESTS=1
or `--run-slow` is supplied.
"""

import os

import pytest

from app.engine.embeddings.base import BaseEmbeddingProvider
from app.engine.embeddings.mock_provider import DeterministicMockMultilingualEmbeddingProvider
from app.schemas.embedding_config import EmbeddingProviderConfig


@pytest.mark.slow
class TestRealMultilingualModelIntegrationSanity:
    """Provider and model integration sanity verification."""

    @pytest.fixture(autouse=True)
    def check_slow_enabled(self) -> None:
        if not os.environ.get("RUN_SLOW_TESTS"):
            pytest.skip("Skipping slow real-model test. Set RUN_SLOW_TESTS=1 to run.")

    def test_bge_m3_integration_sanity(self) -> None:
        """Sanity check: BAAI/bge-m3 provider loads and produces expected dimensionality."""
        from app.engine.embeddings import get_embedding_provider

        provider = get_embedding_provider("fastembed", model_name="BAAI/bge-m3")
        assert isinstance(provider, BaseEmbeddingProvider)

        # 5-document sanity corpus
        docs = [
            "Climate change accelerates Arctic sea ice loss.",
            "Vaccines prevent millions of childhood deaths annually.",
            "The Sundarbans delta hosts the Royal Bengal tiger.",
            "Transformers use self-attention to model long-range sequence context.",
            "Water chlorination prevents diarrheal infections in rural communities.",
        ]

        embeddings = provider.embed_texts(docs)
        assert len(embeddings) == 5
        # Expected dimension for bge-m3 is 1024
        for vec in embeddings:
            assert len(vec) == 1024

        # Determinism check under fixed model weights
        query = "How does sea ice melt in polar regions?"
        q_vec1 = provider.embed_query(query)
        q_vec2 = provider.embed_query(query)
        assert q_vec1 == q_vec2

        # Sanity retrieval check: query about Arctic sea ice should rank doc 0 highest
        def dot_product(v1: list[float], v2: list[float]) -> float:
            return sum(a * b for a, b in zip(v1, v2, strict=True))

        scores = [dot_product(q_vec1, doc_vec) for doc_vec in embeddings]
        best_doc_idx = scores.index(max(scores))
        assert best_doc_idx == 0, "Doc 0 (Arctic sea ice) should rank highest for Arctic query"

    def test_mock_provider_guaranteed_separate_from_empirical(self) -> None:
        """Contract assertion: Mock provider must have IS_MOCK=True and be identifiable."""
        mock_provider = DeterministicMockMultilingualEmbeddingProvider()
        assert getattr(mock_provider, "IS_MOCK", False) is True

        config = EmbeddingProviderConfig(
            model_identifier="BAAI/bge-m3",
            embedding_dimension=1024,
            embedding_provider="fastembed",
            similarity_metric="cosine",
        )
        assert config.embedding_provider != "mock"
