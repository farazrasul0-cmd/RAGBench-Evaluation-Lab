"""Component compatibility validation and topology resolution against Phase A-E engines."""

from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.experiment import PipelineConfig


class RetrievalTopology(StrEnum):
    """The six approved Phase C retrieval topologies."""

    DENSE = "Dense"
    BM25 = "BM25"
    HYBRID = "Hybrid"
    DENSE_RERANKER = "Dense + Reranker"
    BM25_RERANKER = "BM25 + Reranker"
    HYBRID_RERANKER = "Hybrid + Reranker"


class CompatibilityResult(BaseModel):
    """Structured report on component resolution and cross-component compatibility."""

    is_valid: bool
    topology: str
    resolved_retriever_class: str
    resolved_reranker_class: str | None = None
    resolved_chunker_class: str = ""
    resolved_transform_class: str = ""
    resolved_context_strategy: str = ""
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


REGISTERED_CHUNKERS: dict[str, str] = {
    "fixed": "FixedTokenChunker",
    "recursive": "RecursiveCharacterChunker",
    "semantic": "SemanticSimilarityChunker",
    "sentence": "SentenceBoundaryChunker",
}

REGISTERED_TRANSFORMS: dict[str, str] = {
    "none": "IdentityTransformer",
    "identity": "IdentityTransformer",
    "hyde": "HyDETransformer",
    "multi_query": "MultiQueryExpander",
    "multiquery": "MultiQueryExpander",
    "step_back": "StepBackTransformer",
    "stepback": "StepBackTransformer",
}

REGISTERED_CONTEXT_STRATEGIES: dict[str, str] = {
    "standard": "reorder_standard",
    "concat": "reorder_standard",
    "lost_in_the_middle": "reorder_lost_in_the_middle",
    "lost_in_middle": "reorder_lost_in_the_middle",
}

SUPPORTED_METRICS: set[str] = {
    "recall",
    "precision",
    "mrr",
    "ndcg",
    "hit",
    "faithfulness",
    "answer_relevance",
    "citations",
}


def resolve_topology(retrieval_mode: str, reranker_enabled: bool) -> tuple[str, str]:
    """Map configuration retrieval mode and reranker state onto Phase C topology.

    Explicitly maps 'sparse' to Phase C's BM25Retriever without creating
    a redundant 'sparse retriever' abstraction.
    """
    mode = retrieval_mode.lower().strip()

    if mode == "dense":
        if reranker_enabled:
            return RetrievalTopology.DENSE_RERANKER.value, "DenseRetriever"
        return RetrievalTopology.DENSE.value, "DenseRetriever"

    elif mode in ("sparse", "bm25"):
        if reranker_enabled:
            return RetrievalTopology.BM25_RERANKER.value, "BM25Retriever"
        return RetrievalTopology.BM25.value, "BM25Retriever"

    elif mode == "hybrid":
        if reranker_enabled:
            return RetrievalTopology.HYBRID_RERANKER.value, "HybridRetriever"
        return RetrievalTopology.HYBRID.value, "HybridRetriever"

    else:
        raise ValueError(
            f"Unsupported retrieval mode '{retrieval_mode}'. "
            f"Must be one of: 'dense', 'sparse', 'bm25', 'hybrid'."
        )


def validate_component_compatibility(pipeline: PipelineConfig) -> CompatibilityResult:
    """Validate a PipelineConfig against registered Phase A-E engine implementations.

    Detects invalid component combinations, unresolvable strategies, and cross-stage
    incompatibilities (e.g. reranker top_n > retrieval top_k).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Resolve retrieval topology and retriever implementation
    topology_name = "Unknown"
    retriever_cls = "Unknown"
    reranker_cls: str | None = None

    try:
        topology_name, retriever_cls = resolve_topology(
            pipeline.retrieval.mode,
            pipeline.reranker.enabled,
        )
    except ValueError as e:
        errors.append(str(e))

    # 2. Reranker compatibility
    if pipeline.reranker.enabled:
        strat = pipeline.reranker.strategy.lower().strip()
        if strat == "none":
            errors.append("reranker.enabled is True but reranker.strategy is 'none'")
        elif strat == "flashrank":
            reranker_cls = "FlashRankReranker"
        elif strat in ("cross_encoder", "cross-encoder"):
            reranker_cls = "CrossEncoderReranker"
        elif strat in ("mock", "deterministic"):
            reranker_cls = "DeterministicMockReranker"
        else:
            errors.append(f"Unregistered reranker strategy '{pipeline.reranker.strategy}'")

        if pipeline.reranker.top_n > pipeline.retrieval.top_k:
            errors.append(
                f"Incompatible top_n / top_k: reranker.top_n ({pipeline.reranker.top_n}) "
                f"cannot exceed retrieval.top_k ({pipeline.retrieval.top_k})"
            )

        if pipeline.reranker.top_n <= 0:
            errors.append(
                f"reranker.top_n must be strictly positive, got {pipeline.reranker.top_n}"
            )

    # 3. Chunker resolution and parameter checks
    chunk_strat = pipeline.chunking.strategy.lower().strip()
    chunker_cls = REGISTERED_CHUNKERS.get(chunk_strat, "")
    if not chunker_cls:
        errors.append(
            f"Unregistered chunking strategy '{pipeline.chunking.strategy}'. "
            f"Registered strategies: {list(REGISTERED_CHUNKERS.keys())}"
        )

    if pipeline.chunking.chunk_overlap >= pipeline.chunking.chunk_size:
        errors.append(
            f"chunk_overlap ({pipeline.chunking.chunk_overlap}) must be strictly less "
            f"than chunk_size ({pipeline.chunking.chunk_size})"
        )

    # 4. Query transform resolution
    transform_strat = pipeline.query_transform.strategy.lower().strip()
    transform_cls = REGISTERED_TRANSFORMS.get(transform_strat, "")
    if not transform_cls:
        errors.append(
            f"Unregistered query transform strategy '{pipeline.query_transform.strategy}'. "
            f"Registered transforms: {list(REGISTERED_TRANSFORMS.keys())}"
        )

    if (
        transform_strat in ("multi_query", "multiquery")
        and pipeline.query_transform.num_queries < 2
    ):
        warnings.append(
            f"multi_query with num_queries={pipeline.query_transform.num_queries} "
            "generates few expansions"
        )

    # 5. Context strategy resolution
    ctx_strat = pipeline.context.reorder_strategy.lower().strip()
    ctx_func = REGISTERED_CONTEXT_STRATEGIES.get(ctx_strat, "")
    if not ctx_func:
        errors.append(
            f"Unregistered context strategy '{pipeline.context.reorder_strategy}'. "
            f"Registered strategies: {list(REGISTERED_CONTEXT_STRATEGIES.keys())}"
        )

    # 6. Evaluation metrics validation
    for metric in pipeline.evaluation.metrics:
        clean_m = metric.lower().strip()
        if clean_m not in SUPPORTED_METRICS:
            errors.append(
                f"Unsupported evaluation metric '{metric}'. "
                f"Supported metrics: {sorted(SUPPORTED_METRICS)}"
            )

    is_valid = len(errors) == 0

    return CompatibilityResult(
        is_valid=is_valid,
        topology=topology_name,
        resolved_retriever_class=retriever_cls,
        resolved_reranker_class=reranker_cls,
        resolved_chunker_class=chunker_cls,
        resolved_transform_class=transform_cls,
        resolved_context_strategy=ctx_func,
        errors=errors,
        warnings=warnings,
    )
