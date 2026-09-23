"""Query transformations package."""

from typing import Any

from app.engine.query_transforms.base import (
    BaseLLMClient,
    BaseQueryTransformer,
    MockLLMClient,
    QueryTransformationResult,
)
from app.engine.query_transforms.hyde import HyDETransformer
from app.engine.query_transforms.identity import IdentityTransformer
from app.engine.query_transforms.multi_query import MultiQueryExpander
from app.engine.query_transforms.step_back import StepBackTransformer
from app.engine.query_transforms.transformed_retriever import TransformedRetriever


def get_query_transformer(
    strategy: str,
    llm_client: BaseLLMClient | None = None,
    **kwargs: Any,
) -> BaseQueryTransformer:
    """Factory creating configured query transformer."""
    strat = strategy.lower().strip()
    if strat in ("none", "identity"):
        return IdentityTransformer()
    elif strat == "hyde":
        if llm_client is None:
            raise ValueError("HyDETransformer requires an LLM client.")
        return HyDETransformer(llm_client=llm_client, **kwargs)
    elif strat in ("multi_query", "multiquery"):
        if llm_client is None:
            raise ValueError("MultiQueryExpander requires an LLM client.")
        return MultiQueryExpander(llm_client=llm_client, **kwargs)
    elif strat in ("step_back", "stepback"):
        if llm_client is None:
            raise ValueError("StepBackTransformer requires an LLM client.")
        return StepBackTransformer(llm_client=llm_client, **kwargs)
    else:
        raise ValueError(f"Unknown query transformation strategy: {strategy}")


__all__ = [
    "BaseLLMClient",
    "BaseQueryTransformer",
    "HyDETransformer",
    "IdentityTransformer",
    "MockLLMClient",
    "MultiQueryExpander",
    "QueryTransformationResult",
    "StepBackTransformer",
    "TransformedRetriever",
    "get_query_transformer",
]
