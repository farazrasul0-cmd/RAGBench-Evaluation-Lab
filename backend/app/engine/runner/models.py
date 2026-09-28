"""Data models for single-run and matrix pipeline execution."""

from typing import Any

from pydantic import BaseModel, Field


class EvaluationQuery(BaseModel):
    """An individual benchmark query with ground truth for pipeline evaluation."""

    query_id: str = Field(description="Unique stable query identifier")
    query_text: str = Field(description="Raw natural language question")
    expected_answer: str | None = Field(default=None, description="Ground truth answer text")
    ground_truth_chunks: list[str] = Field(
        default_factory=list,
        description="IDs of passage chunks considered relevant for this query",
    )
    ground_truth_docs: list[str] = Field(
        default_factory=list,
        description="IDs of documents considered relevant for this query",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary query metadata (e.g. domain, difficulty, topic)",
    )


class SingleRunResult(BaseModel):
    """Execution summary report for a completed single pipeline run."""

    experiment_run_id: str
    experiment_id: str
    pipeline_config_hash: str
    cache_hash: str
    status: str
    total_queries: int
    completed_queries: int
    failed_queries: int
    mean_metrics: dict[str, float] = Field(default_factory=dict)
    duration_ms: float = 0.0
    cached: bool = False


class MatrixRunResult(BaseModel):
    """Aggregate execution summary across an entire combinatorial experiment matrix."""

    experiment_id: str
    total_configurations: int
    executed_runs: int
    cached_runs: int
    failed_runs: int
    status: str
    results: list[SingleRunResult] = Field(default_factory=list)
    duration_ms: float = 0.0
