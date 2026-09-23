"""Step-Back prompting query transformer abstracting high-level concepts."""

from typing import Any

from app.engine.query_transforms.base import (
    BaseLLMClient,
    BaseQueryTransformer,
    QueryTransformationResult,
)

DEFAULT_STEP_BACK_PROMPT = (
    "You are an expert at high-level knowledge abstraction. Your task is to step back from the "
    "given specific technical question and formulate a more general, conceptual question "
    "that captures the underlying principles, architecture, or background context.\n\n"
    "Specific question: {query}\n\n"
    "Step-back conceptual question:"
)


class StepBackTransformer(BaseQueryTransformer):
    """Step-Back prompting transformer retrieving background concepts alongside specific facts."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        include_original: bool = True,
        prompt_template: str = DEFAULT_STEP_BACK_PROMPT,
    ) -> None:
        self.llm_client = llm_client
        self.include_original = include_original
        self.prompt_template = prompt_template

    @property
    def strategy_name(self) -> str:
        return "step_back"

    def transform(self, query: str, **kwargs: Any) -> QueryTransformationResult:
        prompt = self.prompt_template.format(query=query)
        step_back_q = self.llm_client.generate(prompt=prompt, **kwargs).strip()

        transformed = [step_back_q]
        if self.include_original and query != step_back_q:
            transformed.append(query)

        return QueryTransformationResult(
            original_query=query,
            strategy=self.strategy_name,
            transformed_queries=transformed,
            metadata={
                "step_back_query": step_back_q,
                "include_original": self.include_original,
            },
        )
