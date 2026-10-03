"""Deterministic mock embedding provider for standalone offline unit testing."""

import hashlib
import math
import unicodedata

from app.engine.embeddings.base import BaseEmbeddingProvider


class DeterministicMockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic n-gram hash projection into unit sphere in R^d for offline testing."""

    IS_MOCK: bool = True

    def __init__(self, dimension: int = 384, model_name: str = "mock-deterministic-384d") -> None:
        super().__init__(model_name=model_name, dimension=dimension)

    def _project_text(self, text: str) -> list[float]:
        """Project string to a deterministic unit-normalized float vector."""
        normalized = unicodedata.normalize("NFKC", text.lower().strip())
        vector = [0.0] * self.dimension

        if not normalized:
            vector[0] = 1.0
            return vector

        # Multi-hash hashing of token shingles
        tokens = normalized.split()
        for token in tokens:
            for i in range(len(token)):
                shingle = token[i : i + 3]
                h_val = int(hashlib.md5(shingle.encode("utf-8")).hexdigest(), 16)
                idx = h_val % self.dimension
                sign = 1.0 if ((h_val >> 16) & 1) else -1.0
                vector[idx] += sign

        # L2 normalization
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector

        return [x / norm for x in vector]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Project a batch of texts."""
        results = [self._project_text(t) for t in texts]
        self.validate_dimensions(results)
        return results

    def embed_query(self, query: str) -> list[float]:
        """Project a query."""
        return self._project_text(query)


class DeterministicMockMultilingualEmbeddingProvider(BaseEmbeddingProvider):
    """FOR CI/CONTRACT/INTEGRATION TESTS ONLY.

    Deterministic multilingual mock embedding provider for pipeline validation.

    IMPORTANT: This provider does NOT represent real multilingual semantics.
    It must NEVER appear in empirical RQ3 result tables.  It exists solely to
    allow offline CI tests to exercise the full multilingual retrieval pipeline
    without downloading multi-gigabyte model weights.

    The IS_MOCK class constant allows tests to assert that empirical runs use
    a real provider (``embedding_provider.IS_MOCK is False``).
    """

    IS_MOCK: bool = True

    def __init__(
        self,
        dimension: int = 1024,
        model_name: str = "mock-multilingual-1024d",
    ) -> None:
        super().__init__(model_name=model_name, dimension=dimension)

    def _project_text(self, text: str) -> list[float]:
        """Deterministic SHA-256 projection into R^dimension unit sphere.

        Uses SHA-256 so the vector is stable across Python versions and platforms.
        The distribution is uniform on the unit sphere, which avoids artificial
        retrieval biases introduced by sparse hash projections.
        """
        # Seed with UTF-8 bytes of text; iteration provides enough entropy for dimension
        vector = [0.0] * self.dimension
        seed = text.encode("utf-8")
        for chunk_idx in range(math.ceil(self.dimension * 4 / 32)):
            digest = hashlib.sha256(seed + chunk_idx.to_bytes(4, "little")).digest()
            for byte_idx, byte in enumerate(digest):
                dim_pos = chunk_idx * 32 + byte_idx
                if dim_pos >= self.dimension:
                    break
                # Map byte 0-255 to float in [-1, 1]
                vector[dim_pos] = (byte - 127.5) / 127.5

        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [x / norm for x in vector]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        results = [self._project_text(t) for t in texts]
        self.validate_dimensions(results)
        return results

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self._project_text(query)
