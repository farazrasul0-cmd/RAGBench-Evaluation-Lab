"""Identity query transformer for baseline control experiments."""

from typing import Any

from app.engine.query_transforms.base import BaseQueryTransformer, QueryTransformationResult


class IdentityTransformer(BaseQueryTransformer):
    """Pass-through query transformer serving as the scientific baseline control."""

    @property
    def strategy_name(self) -> str:
        return "none"

    def transform(self, query: str, **kwargs: Any) -> QueryTransformationResult:
        """Return original query unmodified."""
        return QueryTransformationResult(
            original_query=query,
            strategy=self.strategy_name,
            transformed_queries=[query],
            metadata={},
        )
