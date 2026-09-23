"""Dry-run planning engine for RAGBench experiment matrix execution."""

from typing import Any

from pydantic import BaseModel, Field

from app.engine.orchestrator.cartesian import CartesianExpander
from app.engine.orchestrator.compatibility import validate_component_compatibility
from app.engine.orchestrator.identity import compute_cache_identity
from app.schemas.experiment import ExperimentConfig, PipelineConfig


class PlannedPipelineRun(BaseModel):
    """Represents a planned, single-point pipeline execution within an experiment matrix."""

    index: int = Field(description="Deterministic 0-based sequence index in execution plan")
    pipeline_config_hash: str = Field(description="SHA-256 hash of the pipeline configuration")
    cache_key: str = Field(description="Dataset-version-aware composite cache key")
    cache_hash: str = Field(description="SHA-256 hash of the composite cache key")
    parameter_diff: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter delta relative to experiment baseline configuration",
    )
    topology: str = Field(description="Resolved Phase C retrieval topology name")
    is_compatible: bool = Field(description="Whether component combination is valid")
    compatibility_errors: list[str] = Field(
        default_factory=list,
        description="Validation or compatibility errors",
    )
    compatibility_warnings: list[str] = Field(
        default_factory=list,
        description="Validation or compatibility warnings",
    )
    pipeline_config: PipelineConfig = Field(
        description="Complete strongly typed pipeline configuration",
    )


class DryRunPlan(BaseModel):
    """Comprehensive dry-run execution plan for an experiment."""

    experiment_name: str
    experiment_hash: str
    dataset_id: str
    dataset_version_hash: str
    total_planned_pipelines: int
    total_compatible_pipelines: int
    estimated_query_evaluations: int
    is_fully_compatible: bool
    planned_runs: list[PlannedPipelineRun]


class DryRunPlanner:
    """Generates execution plans and dry-run matrices without executing underlying workloads."""

    @classmethod
    def plan(
        cls,
        config: ExperimentConfig,
        dataset_version_hash: str | None = None,
        query_count: int = 0,
        skip_invalid: bool = False,
    ) -> DryRunPlan:
        """Construct complete dry-run plan for an experiment configuration.

        Analyzes the Cartesian expansion, resolves component topologies, checks
        cross-component compatibility, calculates dataset-version-aware cache identities,
        and estimates evaluation volume without mutating state or running pipelines.
        """
        experiment_hash = config.compute_configuration_hash()
        effective_dataset_version = (
            dataset_version_hash
            if dataset_version_hash is not None and dataset_version_hash.strip()
            else config.dataset.dataset_version_id
        )

        baseline = CartesianExpander.get_baseline_pipeline(config)
        pipelines = CartesianExpander.expand(config, skip_invalid=skip_invalid)

        planned_runs: list[PlannedPipelineRun] = []
        fully_compatible = True

        for idx, pipe in enumerate(pipelines):
            pipe_hash = pipe.compute_configuration_hash()
            cache_id = compute_cache_identity(
                dataset_id=pipe.dataset.dataset_id,
                dataset_version_hash=effective_dataset_version,
                pipeline_config_hash=pipe_hash,
            )
            diff = CartesianExpander.compute_diff(baseline, pipe)
            compat = validate_component_compatibility(pipe)

            if not compat.is_valid:
                fully_compatible = False

            run = PlannedPipelineRun(
                index=idx,
                pipeline_config_hash=pipe_hash,
                cache_key=cache_id.cache_key,
                cache_hash=cache_id.cache_hash,
                parameter_diff=diff,
                topology=compat.topology,
                is_compatible=compat.is_valid,
                compatibility_errors=compat.errors,
                compatibility_warnings=compat.warnings,
                pipeline_config=pipe,
            )
            planned_runs.append(run)

        compatible_count = sum(1 for r in planned_runs if r.is_compatible)
        total_evaluations = len(planned_runs) * max(0, query_count)

        return DryRunPlan(
            experiment_name=config.metadata.name,
            experiment_hash=experiment_hash,
            dataset_id=config.dataset.dataset_id,
            dataset_version_hash=effective_dataset_version,
            total_planned_pipelines=len(planned_runs),
            total_compatible_pipelines=compatible_count,
            estimated_query_evaluations=total_evaluations,
            is_fully_compatible=fully_compatible,
            planned_runs=planned_runs,
        )
