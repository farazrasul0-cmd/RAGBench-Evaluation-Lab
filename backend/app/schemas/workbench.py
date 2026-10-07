import hashlib
from typing import Any

from pydantic import BaseModel, Field


class CanonicalControlSignature(BaseModel):
    """Canonical experimental control signature for scientific equivalence.

    Ensures valid Pareto comparability across runs.
    """

    dataset_id: str = "multilingual_canonical"
    dataset_version_id: str = "dd59f087-86ff-4872-925f-adb03fc8d9a2"
    benchmark_hash: str = "default_benchmark_hash"
    query_population: str = "N=25 matched information units"
    top_k: int = 5
    protocol_version: str = "ragbench-protocol-v1.0"
    metrics_version: str = "metrics-v1.0"
    chunking_strategy: str = "fixed"
    chunk_size: int = 200
    chunk_overlap: int = 20
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    reranker_strategy: str = "none"
    signature_hash: str = ""


def compute_control_signature(config: dict[str, Any], top_k: int = 5) -> CanonicalControlSignature:
    """Derive deterministic canonical control signature from experiment configuration."""
    params = config.get("parameters", {})
    chunk_cfg = params.get("chunking", {})
    emb_cfg = params.get("embedding", {})
    dataset_cfg = config.get("dataset", {})
    rerank_cfg = params.get("reranker", {})

    dataset_id = str(
        dataset_cfg.get("dataset_id") or config.get("dataset_id") or "multilingual_canonical"
    )
    dataset_version_id = str(
        dataset_cfg.get("dataset_version_id")
        or config.get("dataset_version_id")
        or "dd59f087-86ff-4872-925f-adb03fc8d9a2"
    )
    benchmark_hash = str(config.get("benchmark_hash") or "bge_m3_frozen_eval")
    query_pop = str(config.get("query_population") or "N=25 matched information units")
    protocol = str(config.get("evaluation_protocol_version") or "ragbench-protocol-v1.0")
    metrics_ver = str(config.get("metric_definition_version") or "metrics-v1.0")

    c_strat = str(chunk_cfg.get("strategy") or "fixed")
    c_sz = int(chunk_cfg.get("chunk_size") or 200)
    c_ov = int(chunk_cfg.get("chunk_overlap") or 20)

    emb_model = str(emb_cfg.get("model_name") or "BAAI/bge-m3")
    emb_dim = int(emb_cfg.get("dimension") or (1024 if "bge-m3" in emb_model.lower() else 384))
    rerank_strat = str(rerank_cfg.get("strategy") or "none")

    canonical_repr = (
        f"dataset:{dataset_id}|ver:{dataset_version_id}|pop:{query_pop}|"
        f"k:{top_k}|proto:{protocol}|metrics:{metrics_ver}|"
        f"chunk:{c_strat}:{c_sz}:{c_ov}|emb:{emb_model}:{emb_dim}|"
        f"rerank:{rerank_strat}"
    )
    sig_hash = hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()

    return CanonicalControlSignature(
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        benchmark_hash=benchmark_hash,
        query_population=query_pop,
        top_k=top_k,
        protocol_version=protocol,
        metrics_version=metrics_ver,
        chunking_strategy=c_strat,
        chunk_size=c_sz,
        chunk_overlap=c_ov,
        embedding_model=emb_model,
        embedding_dimension=emb_dim,
        reranker_strategy=rerank_strat,
        signature_hash=sig_hash,
    )


class ParetoComparisonValidationRequest(BaseModel):
    """Payload to validate scientific comparability of candidate sweeps."""

    signatures: list[CanonicalControlSignature]


class ParetoComparisonValidationResponse(BaseModel):
    """Authoritative validation verdict determining if sweeps are scientifically comparable."""

    is_comparable: bool
    divergent_dimensions: list[str] = Field(default_factory=list)
    canonical_signature: CanonicalControlSignature | None = None
    validation_notice: str


class MatrixPreviewRequest(BaseModel):
    """Parameter options for matrix sweep generation."""

    name: str = Field(default="Sweep Experiment", description="Experiment sweep name")
    description: str = Field(
        default="Parameter sweep generated via Matrix Builder",
        description="Hypothesis or study notes",
    )
    dataset_id: str = Field(default="multilingual_canonical")
    dataset_version_id: str = Field(default="dd59f087-86ff-4872-925f-adb03fc8d9a2")
    chunking_strategies: list[str] = Field(default_factory=lambda: ["fixed", "sentence"])
    chunk_sizes: list[int] = Field(default_factory=lambda: [200, 512])
    chunk_overlaps: list[int] = Field(default_factory=lambda: [20])
    embedding_models: list[str] = Field(default_factory=lambda: ["BAAI/bge-m3"])
    retrieval_strategies: list[str] = Field(default_factory=lambda: ["dense", "bm25", "hybrid"])
    rerankers: list[str] = Field(default_factory=lambda: ["none"])
    generation_models: list[str] = Field(default_factory=lambda: ["none"])
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
    """Calculated combinatorial analysis and compatibility verification for matrix sweep."""

    total_combinations: int
    complexity_category: str
    estimated_queries_per_run: int
    total_pipeline_points: int
    is_executable: bool = True
    compatibility_status: str = "COMPATIBLE"
    validation_warnings: list[str] = Field(default_factory=list)
    controlled_dimensions: dict[str, Any] = Field(default_factory=dict)
    canonical_control_signature: CanonicalControlSignature | None = None
    sample_configurations: list[MatrixConfigurationPoint]


class MatrixYamlResponse(BaseModel):
    """Generated YAML payload ready for execution or download."""

    yaml_string: str
    total_combinations: int
    configuration_hash: str
    is_executable: bool = True
    validation_warnings: list[str] = Field(default_factory=list)
