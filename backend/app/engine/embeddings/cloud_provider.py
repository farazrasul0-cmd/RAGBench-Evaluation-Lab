"""Cloud embedding provider using LiteLLM or direct REST APIs."""

from typing import Any

import httpx

from app.engine.embeddings.base import BaseEmbeddingProvider, EmbeddingError


class CloudEmbeddingProvider(BaseEmbeddingProvider):
    """Cloud embedding provider supporting OpenAI, Voyage, Cohere via HTTP."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, dimension=dimension)
        self.api_key = api_key
        self.api_base = api_base

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Request cloud embeddings via REST API."""
        if not texts:
            return []
        if not self.api_key:
            raise EmbeddingError(
                f"Cannot use CloudEmbeddingProvider({self.model_name}): API key is not configured"
            )
        try:
            url = f"{self.api_base}/embeddings"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "input": texts,
                "model": self.model_name,
            }
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    raise EmbeddingError(
                        f"Cloud API returned error {resp.status_code}: {resp.text}"
                    )
                data = resp.json()
                results = [item["embedding"] for item in data["data"]]
                self.validate_dimensions(results)
                return results
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Cloud embedding request failed: {e}") from e

    def embed_query(self, query: str) -> list[float]:
        """Generate cloud embedding for query."""
        results = self.embed_texts([query])
        return results[0]
