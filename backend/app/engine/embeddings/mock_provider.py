"""Deterministic mock embedding provider for standalone offline unit testing."""

import hashlib
import math
import unicodedata

from app.engine.embeddings.base import BaseEmbeddingProvider


class DeterministicMockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic n-gram hash projection into unit sphere in R^d for offline testing."""

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
