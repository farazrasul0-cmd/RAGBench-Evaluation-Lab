"""Multi-Query expander generating diverse query formulations."""

import re
from typing import Any

from app.engine.query_transforms.base import (
    BaseLLMClient,
    BaseQueryTransformer,
    QueryTransformationResult,
)

DEFAULT_MULTI_QUERY_PROMPT = (
    "You are an expert AI search assistant. Your task is to generate {num_queries} different "
    "versions of the given technical question to retrieve relevant documents from a vector store.\n"
    "By generating multiple perspectives, technical synonyms, and phrasing variations, your "
    "goal is to overcome distance-based similarity search limitations.\n"
    "Provide each alternative query on a new line without numbers or bullet points.\n"
    "Original question: {query}\n\n"
    "Alternative queries:"
)


class MultiQueryExpander(BaseQueryTransformer):
    """Multi-query expander generating diverse query formulations merged via RRF."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        num_queries: int = 3,
        include_original: bool = True,
        prompt_template: str = DEFAULT_MULTI_QUERY_PROMPT,
    ) -> None:
        self.llm_client = llm_client
        self.num_queries = num_queries
        self.include_original = include_original
        self.prompt_template = prompt_template

    @property
    def strategy_name(self) -> str:
        return "multi_query"

    def _parse_queries(self, response: str) -> list[str]:
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        cleaned: list[str] = []
        for line in lines:
            # Strip markdown bullets or numbers like '1. ', '- ', '* '
            c = re.sub(r"^[0-9]+[.\)]\s*", "", line)
            c = re.sub(r"^[-*•]\s*", "", c).strip()
            if c and c not in cleaned:
                cleaned.append(c)
        return cleaned

    def transform(self, query: str, **kwargs: Any) -> QueryTransformationResult:
        prompt = self.prompt_template.format(num_queries=self.num_queries, query=query)
        response = self.llm_client.generate(prompt=prompt, **kwargs)
        sub_queries = self._parse_queries(response)

        # Ensure we have queries, falling back to original
        if not sub_queries:
            sub_queries = [query]

        if self.include_original and query not in sub_queries:
            sub_queries.insert(0, query)

        return QueryTransformationResult(
            original_query=query,
            strategy=self.strategy_name,
            transformed_queries=sub_queries[
                : self.num_queries + (1 if self.include_original else 0)
            ],
            metadata={
                "generated_count": len(sub_queries),
                "include_original": self.include_original,
            },
        )
