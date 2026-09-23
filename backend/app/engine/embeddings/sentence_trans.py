"""PyTorch Sentence-Transformers embedding provider."""

import importlib
from typing import Any

from app.engine.embeddings.base import BaseEmbeddingProvider, EmbeddingError


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """Sentence-transformers PyTorch embedding provider."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        dimension: int = 768,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, dimension=dimension)
        self.device = device
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                st_module = importlib.import_module("sentence_transformers")
                model_cls = st_module.SentenceTransformer
                self._model = model_cls(self.model_name, device=self.device)
            except ImportError as e:
                msg = (
                    "sentence-transformers is not installed. Run: pip install sentence-transformers"
                )
                raise EmbeddingError(msg) from e
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load SentenceTransformer '{self.model_name}': {e}"
                ) from e
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate dense vectors using SentenceTransformer."""
        if not texts:
            return []
        try:
            model = self._get_model()
            embeddings = model.encode(texts, normalize_embeddings=True)
            results = [e.tolist() for e in embeddings]
            self.validate_dimensions(results)
            return results
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"SentenceTransformers embedding failed: {e}") from e

    def embed_query(self, query: str) -> list[float]:
        """Generate dense vector for a query."""
        results = self.embed_texts([query])
        return results[0]
