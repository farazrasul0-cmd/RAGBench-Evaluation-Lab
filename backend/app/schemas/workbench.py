"""Schemas for Academic Research Workbench API (Phase H)."""

from typing import Any

from pydantic import BaseModel, Field


class MatrixPreviewRequest(BaseModel):
    """Parameter options for matrix sweep generation."""

    name: str = Field(default="Sweep Experiment", description="Experiment sweep name")
    description: str = Field(
        default="Parameter sweep generated via Matrix Builder",
        description="Hypothesis or study notes",
    )
    dataset_id: str = Field(default="multilingual_canonical")
    dataset_version_id: str = Field(default="latest")
    chunking_strategies: list[str] = Field(default_factory=lambda: ["fixed", "recursive"])
    chunk_sizes: list[int] = Field(default_factory=lambda: [200, 400])
    chunk_overlaps: list[int] = Field(default_factory=lambda: [20])
    embedding_models: list[str] = Field(default_factory=lambda: ["BAAI/bge-m3"])
    retrieval_strategies: list[str] = Field(
        default_factory=lambda: ["dense", "bm25", "hybrid"]
    )
    rerankers: list[str] = Field(default_factory=lambda: ["none"])
    generation_models: list[str] = Field(default_factory=lambda: ["mock"])
    top_k_values: list[int] = Field(default_factory=lambda: [5])


class MatrixConfigurationPoint(BaseModel):
    """Single concrete configuration combination."""

    index: int
    chunking: dict[str, Any]
    embedding: dict[str, Any]
    retrieval: dict[str, Any]
    reranker: dict[str, Any]
    generation: dict[str, Any]


class MatrixPreviewResponse(BaseModel):
    """Calculated combinatorial analysis for matrix sweep."""

    total_combinations: int
    complexity_category: str
    estimated_queries_per_run: int
    total_pipeline_points: int
    sample_configurations: list[MatrixConfigurationPoint]


class MatrixYamlResponse(BaseModel):
    """Generated YAML payload ready for execution or download."""

    yaml_string: str
    total_combinations: int
    configuration_hash: str
