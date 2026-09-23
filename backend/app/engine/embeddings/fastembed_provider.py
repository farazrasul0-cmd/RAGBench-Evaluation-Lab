"""CPU-optimized local embedding provider using fastembed ONNX runtime."""

from typing import Any

import fastembed

from app.engine.embeddings.base import BaseEmbeddingProvider, EmbeddingError


class FastEmbedProvider(BaseEmbeddingProvider):
    """Local ONNX CPU embedding provider using fastembed."""

    KNOWN_DIMENSIONS: dict[str, int] = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "intfloat/multilingual-e5-small": 384,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        dim = self.KNOWN_DIMENSIONS.get(model_name, 384)
        super().__init__(model_name=model_name, dimension=dim)
        self.cache_dir = cache_dir
        self._model: fastembed.TextEmbedding | None = None

    def _get_model(self) -> fastembed.TextEmbedding:
        if self._model is None:
            try:
                self._model = fastembed.TextEmbedding(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                )
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to initialize FastEmbed model '{self.model_name}': {e}"
                ) from e
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate dense vectors for a batch of passages."""
        if not texts:
            return []
        try:
            model = self._get_model()
            embeddings_iter = model.embed(texts)
            results = [e.tolist() for e in embeddings_iter]
            self.validate_dimensions(results)
            return results
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"FastEmbed text embedding failed: {e}") from e

    def embed_query(self, query: str) -> list[float]:
        """Generate dense vector for a single query."""
        results = self.embed_texts([query])
        if not results:
            raise EmbeddingError("Empty embedding returned for query")
        return results[0]
