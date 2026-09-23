"""Cartesian experiment orchestrator for RAGBench parameter sweeps."""

import itertools
from typing import Any

from pydantic import ValidationError

from app.schemas.experiment import (
    ChunkingConfig,
    ContextConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentConfig,
    GenerationConfig,
    PipelineConfig,
    QueryTransformConfig,
    RerankerConfig,
    RetrievalConfig,
)


class ConfigurationExpansionError(ValueError):
    """Raised when Cartesian expansion encounters an invalid configuration combination."""


class CartesianExpander:
    """Expands an ExperimentConfig into a deterministic sequence of PipelineConfig points."""

    COMPONENT_CLASSES: dict[str, type] = {
        "chunking": ChunkingConfig,
        "embedding": EmbeddingConfig,
        "retrieval": RetrievalConfig,
        "reranker": RerankerConfig,
        "query_transform": QueryTransformConfig,
        "context": ContextConfig,
        "generation": GenerationConfig,
        "evaluation": EvaluationConfig,
    }

    # Strict canonical component ordering
    COMPONENT_ORDER = (
        "chunking",
        "embedding",
        "retrieval",
        "reranker",
        "query_transform",
        "context",
        "generation",
        "evaluation",
    )

    @classmethod
    def get_baseline_pipeline(cls, config: ExperimentConfig) -> PipelineConfig:
        """Construct the canonical baseline PipelineConfig from an ExperimentConfig."""
        dataset = config.dataset

        component_instances: dict[str, Any] = {"dataset": dataset}
        for comp_name in cls.COMPONENT_ORDER:
            comp_cls = cls.COMPONENT_CLASSES[comp_name]
            raw_dict = getattr(config.parameters, comp_name, {}) or {}

            # Extract scalar baseline values (first item if list)
            baseline_kwargs: dict[str, Any] = {}
            for k, v in raw_dict.items():
                if isinstance(v, list) and not isinstance(v, (str, bytes)):
                    # Special case for evaluation metrics/k_values lists
                    if comp_name == "evaluation" and k in ("metrics", "k_values"):
                        if v and isinstance(v[0], list):
                            baseline_kwargs[k] = v[0]
                        else:
                            baseline_kwargs[k] = v
                    else:
                        baseline_kwargs[k] = v[0] if len(v) > 0 else None
                else:
                    baseline_kwargs[k] = v

            # Map 'sparse' retrieval mode to 'bm25'
            if comp_name == "retrieval" and baseline_kwargs.get("mode") == "sparse":
                baseline_kwargs["mode"] = "bm25"

            component_instances[comp_name] = comp_cls(**baseline_kwargs)

        return PipelineConfig(**component_instances)

    @classmethod
    def expand(
        cls,
        config: ExperimentConfig,
        baseline_pipeline: PipelineConfig | None = None,
        skip_invalid: bool = False,
    ) -> list[PipelineConfig]:
        """Expand experiment configuration into an ordered sequence of PipelineConfigs.

        Guarantees:
        1. Deterministic Cartesian ordering across all sweep dimensions.
        2. Empty sweep produces exactly 1 pipeline point (the baseline).
        3. Single sweep dimension produces points strictly in the declared list order.
        4. Omitted parameters retain baseline pipeline values.
        """
        try:
            baseline = baseline_pipeline or cls.get_baseline_pipeline(config)
        except (ValidationError, ValueError) as err:
            if not skip_invalid:
                raise ConfigurationExpansionError(
                    f"Invalid configuration generated during Cartesian expansion: {err}"
                ) from err
            # Fallback baseline with defaults if skipping invalid points
            baseline = PipelineConfig(
                dataset=config.dataset,
                chunking=ChunkingConfig(),
                embedding=EmbeddingConfig(),
                retrieval=RetrievalConfig(),
                reranker=RerankerConfig(),
                query_transform=QueryTransformConfig(),
                context=ContextConfig(),
                generation=GenerationConfig(),
                evaluation=EvaluationConfig(),
            )

        dataset = config.dataset

        # Collect sweep dimensions across components in canonical order
        dimension_names: list[tuple[str, str]] = []  # (comp_name, field_name)
        dimension_values: list[list[Any]] = []

        for comp_name in cls.COMPONENT_ORDER:
            comp_raw = getattr(config.parameters, comp_name, {}) or {}

            # Sort field names within component alphabetically
            for field_name in sorted(comp_raw.keys()):
                val = comp_raw[field_name]
                # Check if this field represents a sweep list
                is_sweep_list = False
                if comp_name == "evaluation" and field_name in ("metrics", "k_values"):
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
                        is_sweep_list = True
                elif isinstance(val, list) and not isinstance(val, (str, bytes)):
                    is_sweep_list = True

                if is_sweep_list:
                    dimension_names.append((comp_name, field_name))
                    # Map 'sparse' to 'bm25' for retrieval mode
                    if comp_name == "retrieval" and field_name == "mode":
                        normalized_vals = ["bm25" if v == "sparse" else v for v in val]
                        dimension_values.append(normalized_vals)
                    else:
                        dimension_values.append(list(val))

        # Handle empty sweep (no sweep lists specified)
        if not dimension_values:
            return [baseline.model_copy(deep=True)]

        # Generate Cartesian product in deterministic order
        expanded_pipelines: list[PipelineConfig] = []
        for combination in itertools.product(*dimension_values):
            # Start with a copy of baseline components
            comp_data: dict[str, dict[str, Any]] = {}
            for comp_name in cls.COMPONENT_ORDER:
                comp_data[comp_name] = getattr(baseline, comp_name).model_dump()

            # Apply combination overrides
            for (comp_name, field_name), val in zip(dimension_names, combination, strict=True):
                comp_data[comp_name][field_name] = val

            try:
                chunking = ChunkingConfig(**comp_data["chunking"])
                embedding = EmbeddingConfig(**comp_data["embedding"])
                retrieval = RetrievalConfig(**comp_data["retrieval"])
                reranker = RerankerConfig(**comp_data["reranker"])
                query_transform = QueryTransformConfig(**comp_data["query_transform"])
                context = ContextConfig(**comp_data["context"])
                generation = GenerationConfig(**comp_data["generation"])
                evaluation = EvaluationConfig(**comp_data["evaluation"])

                pipe = PipelineConfig(
                    dataset=dataset,
                    chunking=chunking,
                    embedding=embedding,
                    retrieval=retrieval,
                    reranker=reranker,
                    query_transform=query_transform,
                    context=context,
                    generation=generation,
                    evaluation=evaluation,
                )
                expanded_pipelines.append(pipe)

            except (ValidationError, ValueError) as err:
                if skip_invalid:
                    continue
                raise ConfigurationExpansionError(
                    f"Invalid configuration generated during Cartesian expansion: {err}"
                ) from err

        return expanded_pipelines

    @classmethod
    def compute_diff(
        cls,
        baseline: PipelineConfig,
        target: PipelineConfig,
    ) -> dict[str, Any]:
        """Compute human-readable dictionary of parameter deltas between baseline and target."""
        diff: dict[str, Any] = {}

        for comp_name in cls.COMPONENT_ORDER:
            base_comp = getattr(baseline, comp_name)
            target_comp = getattr(target, comp_name)

            for field_name in type(base_comp).model_fields:
                base_val = getattr(base_comp, field_name)
                target_val = getattr(target_comp, field_name)
                if base_val != target_val:
                    diff[f"{comp_name}.{field_name}"] = target_val

        return diff
