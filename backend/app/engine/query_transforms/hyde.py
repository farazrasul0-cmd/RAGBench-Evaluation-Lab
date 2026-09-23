"""Hypothetical Document Embeddings (HyDE) query transformer."""

from typing import Any

from app.engine.query_transforms.base import (
    BaseLLMClient,
    BaseQueryTransformer,
    QueryTransformationResult,
)

DEFAULT_HYDE_PROMPT = (
    "Please write a comprehensive, technically accurate scientific passage that answers "
    "the following question directly.\n"
    "Question: {query}\n\n"
    "Passage:"
)


class HyDETransformer(BaseQueryTransformer):
    """Hypothetical Document Embeddings (HyDE) query transformer.

    Instead of searching using the raw query, prompts an LLM to hallucinate a
    hypothetical technical passage answering the query, which is subsequently embedded.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_template: str = DEFAULT_HYDE_PROMPT,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_template = prompt_template

    @property
    def strategy_name(self) -> str:
        return "hyde"

    def transform(self, query: str, **kwargs: Any) -> QueryTransformationResult:
        prompt = self.prompt_template.format(query=query)
        hypothetical_passage = self.llm_client.generate(prompt=prompt, **kwargs).strip()

        return QueryTransformationResult(
            original_query=query,
            strategy=self.strategy_name,
            transformed_queries=[hypothetical_passage],
            metadata={
                "hypothetical_passage": hypothetical_passage,
                "prompt_template": self.prompt_template,
            },
        )
