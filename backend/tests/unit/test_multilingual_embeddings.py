"""Unit tests for multilingual embedding providers and EmbeddingProviderConfig schema."""

from app.engine.embeddings.mock_provider import (
    DeterministicMockEmbeddingProvider,
    DeterministicMockMultilingualEmbeddingProvider,
)
from app.schemas.embedding_config import EmbeddingProviderConfig


class TestDeterministicMockEmbeddingProvider:
    def test_is_not_multilingual_mock(self) -> None:
        provider = DeterministicMockEmbeddingProvider()
        assert provider.IS_MOCK is True

    def test_dimension(self) -> None:
        provider = DeterministicMockEmbeddingProvider(dimension=128)
        vecs = provider.embed_texts(["hello"])
        assert len(vecs[0]) == 128

    def test_deterministic(self) -> None:
        provider = DeterministicMockEmbeddingProvider()
        v1 = provider.embed_query("test query")
        v2 = provider.embed_query("test query")
        assert v1 == v2

    def test_unit_normalized(self) -> None:
        import math

        provider = DeterministicMockEmbeddingProvider()
        vec = provider.embed_query("normalization test")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-5


class TestDeterministicMockMultilingualEmbeddingProvider:
    """Validates the CI/contract mock — not a test of multilingual embedding quality."""

    def test_is_mock_flag(self) -> None:
        provider = DeterministicMockMultilingualEmbeddingProvider()
        assert provider.IS_MOCK is True

    def test_dimension_default_1024(self) -> None:
        provider = DeterministicMockMultilingualEmbeddingProvider()
        vec = provider.embed_query("test")
        assert len(vec) == 1024

    def test_custom_dimension(self) -> None:
        provider = DeterministicMockMultilingualEmbeddingProvider(dimension=256)
        vecs = provider.embed_texts(["text"])
        assert len(vecs[0]) == 256

    def test_deterministic_english(self) -> None:
        provider = DeterministicMockMultilingualEmbeddingProvider()
        v1 = provider.embed_query("The climate is changing rapidly.")
        v2 = provider.embed_query("The climate is changing rapidly.")
        assert v1 == v2

    def test_deterministic_bengali(self) -> None:
        provider = DeterministicMockMultilingualEmbeddingProvider()
        text = (
            "\u099c\u09b2\u09ac\u09be\u09df\u09c1 \u09aa\u09b0\u09bf\u09ac\u09b0\u09cd\u09a4\u09a8"
        )
        v1 = provider.embed_query(text)
        v2 = provider.embed_query(text)
        assert v1 == v2

    def test_unit_normalized(self) -> None:
        import math

        provider = DeterministicMockMultilingualEmbeddingProvider()
        vec = provider.embed_query("normalization test")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-5

    def test_distinct_texts_produce_distinct_vectors(self) -> None:
        provider = DeterministicMockMultilingualEmbeddingProvider()
        v_en = provider.embed_query("The Arctic is warming.")
        bn_query = (
            "\u0986\u09b0\u09cd\u0995\u099f\u09bf\u0995 "
            "\u0989\u09b7\u09cd\u09a3 \u09b9\u099a\u09cd\u099b\u09c7\u0964"
        )
        v_bn = provider.embed_query(bn_query)
        # Different texts must produce different vectors (hash collision unlikely)
        assert v_en != v_bn

    def test_batch_embed_texts(self) -> None:
        provider = DeterministicMockMultilingualEmbeddingProvider()
        texts = ["Hello world", "\u09b9\u09cd\u09af\u09be\u09b2\u09cb", "Climate change"]
        vecs = provider.embed_texts(texts)
        assert len(vecs) == 3
        assert all(len(v) == 1024 for v in vecs)

    def test_is_mock_class_constant(self) -> None:
        """Class-level constant allows tests to gate on mock vs. real provider."""
        assert DeterministicMockMultilingualEmbeddingProvider.IS_MOCK is True
        assert DeterministicMockEmbeddingProvider.IS_MOCK is True


class TestEmbeddingProviderConfig:
    """Validate EmbeddingProviderConfig reproducibility schema (Amendment 2)."""

    def test_all_fields_present(self) -> None:
        config = EmbeddingProviderConfig(
            model_identifier="BAAI/bge-m3",
            model_revision=None,
            embedding_dimension=1024,
            normalize_embeddings=True,
            embedding_provider="fastembed",
            similarity_metric="cosine",
            query_instruction=None,
            document_instruction=None,
        )
        assert config.model_identifier == "BAAI/bge-m3"
        assert config.embedding_dimension == 1024
        assert config.normalize_embeddings is True
        assert config.similarity_metric == "cosine"
        assert config.query_instruction is None
        assert config.document_instruction is None

    def test_e5_style_instructions(self) -> None:
        config = EmbeddingProviderConfig(
            model_identifier="intfloat/multilingual-e5-small",
            embedding_dimension=384,
            embedding_provider="sentence-transformers",
            query_instruction="query: ",
            document_instruction="passage: ",
        )
        assert config.query_instruction == "query: "
        assert config.document_instruction == "passage: "

    def test_model_revision_optional(self) -> None:
        config = EmbeddingProviderConfig(
            model_identifier="BAAI/bge-m3",
            embedding_dimension=1024,
            embedding_provider="fastembed",
        )
        assert config.model_revision is None

    def test_serialization_round_trip(self) -> None:
        config = EmbeddingProviderConfig(
            model_identifier="BAAI/bge-m3",
            model_revision="abc123def456",
            embedding_dimension=1024,
            normalize_embeddings=True,
            embedding_provider="fastembed",
            similarity_metric="cosine",
            query_instruction=None,
            document_instruction=None,
        )
        data = config.model_dump()
        restored = EmbeddingProviderConfig(**data)
        assert restored == config
