"""Base interfaces and data contracts for query transformation strategies."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class QueryTransformationResult(BaseModel):
    """Encapsulates the output of a query transformation pipeline for reproducibility."""

    original_query: str
    strategy: str
    transformed_queries: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseLLMClient(ABC):
    """Abstract interface for LLM text generation in query transformations."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        """Generate completion for given prompt."""
        ...


class MockLLMClient(BaseLLMClient):
    """Deterministic offline mock LLM client for testing query transformations."""

    def __init__(
        self,
        canned_responses: dict[str, str] | None = None,
        default_prefix: str = "Mock passage answering",
    ) -> None:
        self.canned_responses = canned_responses or {}
        self.default_prefix = default_prefix

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        for key, response in self.canned_responses.items():
            if key in prompt:
                return response
        return f"{self.default_prefix}: {prompt[:60]}..."


class BaseQueryTransformer(ABC):
    """Abstract interface for query transformation strategies."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Identifier name of the transformation strategy."""
        ...

    @abstractmethod
    def transform(self, query: str, **kwargs: Any) -> QueryTransformationResult:
        """Transform user query into one or more target retrieval queries."""
        ...
