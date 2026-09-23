"""Experiment configuration schemas, semantic parameter validation, and canonical hashing."""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================================
# 1. Non-Semantic Metadata (Human-readable descriptions & run metadata)
# ============================================================================


class ExperimentMetadata(BaseModel):
    """Non-semantic experiment metadata that does NOT alter experimental execution."""

    name: str = Field(default="experiment", description="Human-readable experiment name")
    description: str = Field(default="", description="Hypothesis or study description")
    author: str = Field(default="", description="Author or researcher identifier")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    version: str = Field(default="1.0", description="Schema version identifier")


# ============================================================================
# 2. Semantic Component Configurations (Concrete execution points)
# ============================================================================


class DatasetConfig(BaseModel):
    """Dataset identity and file paths participating in evaluation."""

    dataset_id: str = Field(..., min_length=1, description="Dataset identifier")
    dataset_version_id: str = Field(..., min_length=1, description="Dataset version snapshot")
    corpus_path: str | None = Field(default=None, description="Path to raw corpus directory")
    evaluation_qa_path: str | None = Field(default=None, description="Path to evaluation QA JSONL")


class ChunkingConfig(BaseModel):
    """Text segmentation hyperparameters."""

    strategy: str = Field(
        default="fixed", description="Chunking strategy: fixed, recursive, sentence, semantic"
    )
    chunk_size: int = Field(default=512, gt=0, description="Target chunk token length")
    chunk_overlap: int = Field(default=64, ge=0, description="Sliding window token overlap")
    similarity_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Semantic chunker threshold"
    )

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"fixed", "recursive", "sentence", "semantic"}
        strat = v.lower().strip()
        if strat not in allowed:
            raise ValueError(f"Invalid chunking strategy: '{v}'. Must be one of {allowed}")
        return strat

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less "
                f"than chunk_size ({self.chunk_size})"
            )
        return self


class EmbeddingConfig(BaseModel):
    """Vector embedding provider and model hyperparameters."""

    provider: str = Field(default="fastembed", description="fastembed, sentence_transformers, mock")
    model_name: str = Field(default="BAAI/bge-small-en-v1.5", min_length=1)
    dimension: int = Field(default=384, gt=0)
    batch_size: int = Field(default=32, gt=0)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"fastembed", "sentence_transformers", "mock"}
        prov = v.lower().strip()
        if prov not in allowed:
            raise ValueError(f"Invalid embedding provider: '{v}'. Must be one of {allowed}")
        return prov


class RetrievalConfig(BaseModel):
    """Passage retrieval topology hyperparameters."""

    mode: str = Field(default="dense", description="dense, bm25, hybrid")
    top_k: int = Field(default=10, gt=0)
    hybrid_fusion: str = Field(default="rrf", description="rrf, rsn")
    rrf_k: int = Field(default=60, gt=0)
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = {"dense", "bm25", "hybrid"}
        m = v.lower().strip()
        if m not in allowed:
            raise ValueError(f"Invalid retrieval mode: '{v}'. Must be one of {allowed}")
        return m

    @field_validator("hybrid_fusion")
    @classmethod
    def validate_fusion(cls, v: str) -> str:
        allowed = {"rrf", "rsn"}
        f = v.lower().strip()
        if f not in allowed:
            raise ValueError(f"Invalid hybrid fusion method: '{v}'. Must be one of {allowed}")
        return f


class RerankerConfig(BaseModel):
    """Neural / lexical passage reranking hyperparameters."""

    enabled: bool = Field(default=False)
    strategy: str = Field(default="none", description="none, flashrank, cross_encoder, mock")
    model_name: str = Field(default="")
    top_n: int = Field(default=5, gt=0)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"none", "flashrank", "cross_encoder", "mock"}
        s = v.lower().strip()
        if s not in allowed:
            raise ValueError(f"Invalid reranker strategy: '{v}'. Must be one of {allowed}")
        return s


class QueryTransformConfig(BaseModel):
    """Query formulation and expansion hyperparameters."""

    strategy: str = Field(
        default="none", description="none, identity, hyde, multi_query, step_back"
    )
    num_queries: int = Field(default=3, gt=0)
    include_original: bool = Field(default=True)
    prompt_template: str | None = Field(default=None)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"none", "identity", "hyde", "multi_query", "step_back"}
        s = v.lower().strip()
        if s not in allowed:
            raise ValueError(f"Invalid query transform strategy: '{v}'. Must be one of {allowed}")
        return s


class ContextConfig(BaseModel):
    """Context formulation, budget constraints, and ordering."""

    token_budget: int = Field(default=2048, gt=0)
    reorder_strategy: str = Field(default="standard", description="standard, lost_in_the_middle")
    encoding_name: str = Field(default="cl100k_base")

    @field_validator("reorder_strategy")
    @classmethod
    def validate_reorder(cls, v: str) -> str:
        allowed = {"standard", "lost_in_the_middle"}
        s = v.lower().strip()
        if s not in allowed:
            raise ValueError(f"Invalid reorder strategy: '{v}'. Must be one of {allowed}")
        return s


class GenerationConfig(BaseModel):
    """Downstream answer generation hyperparameters."""

    model_name: str = Field(default="mock", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=512, gt=0)


class EvaluationConfig(BaseModel):
    """Quantitative evaluation suite specifications."""

    metrics: list[str] = Field(
        default_factory=lambda: [
            "recall",
            "precision",
            "mrr",
            "ndcg",
            "hit",
            "faithfulness",
            "answer_relevance",
            "citations",
        ]
    )
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[str]) -> list[str]:
        allowed = {
            "recall",
            "precision",
            "mrr",
            "ndcg",
            "hit",
            "faithfulness",
            "answer_relevance",
            "citations",
        }
        for m in v:
            if m.lower().strip() not in allowed:
                raise ValueError(f"Invalid metric '{m}'. Must be one of {allowed}")
        return [m.lower().strip() for m in v]

    @field_validator("k_values")
    @classmethod
    def validate_k_values(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("k_values must contain at least one positive integer cutoff")
        for k in v:
            if k <= 0:
                raise ValueError(f"k cutoff must be strictly positive (> 0), got: {k}")
        return sorted(set(v))


# ============================================================================
# 3. Canonical Serialization & Pipeline Configuration Model
# ============================================================================


def canonical_json_dump(data: dict[str, Any]) -> str:
    """Serialize dictionary to canonical, deterministic JSON.

    Guarantees:
    - Recursive key sorting
    - Deterministic compact separators (no whitespace)
    - Verbatim Unicode preservation (ensure_ascii=False)
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_hash(data: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of canonical JSON."""
    canon = canonical_json_dump(data)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class PipelineConfig(BaseModel):
    """Complete, discrete hyperparameter specification for a single RAG pipeline execution."""

    dataset: DatasetConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    query_transform: QueryTransformConfig = Field(default_factory=QueryTransformConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def to_canonical_dict(self) -> dict[str, Any]:
        """Convert all semantic parameters into a sorted, normalized dictionary."""
        return {
            "dataset": self.dataset.model_dump(),
            "chunking": self.chunking.model_dump(),
            "embedding": self.embedding.model_dump(),
            "retrieval": self.retrieval.model_dump(),
            "reranker": self.reranker.model_dump(),
            "query_transform": self.query_transform.model_dump(),
            "context": self.context.model_dump(),
            "generation": self.generation.model_dump(),
            "evaluation": self.evaluation.model_dump(),
        }

    def canonical_json(self) -> str:
        """Produce deterministic, canonical JSON representation."""
        return canonical_json_dump(self.to_canonical_dict())

    def compute_configuration_hash(self) -> str:
        """Compute immutable SHA-256 configuration hash of semantic hyperparameters."""
        return compute_hash(self.to_canonical_dict())


# ============================================================================
# 4. Sweep Parameters Specification (Multidimensional Cartesian space)
# ============================================================================


class SweepParameters(BaseModel):
    """Specifies parameter values or parameter sweeps across pipeline components."""

    chunking: dict[str, Any] = Field(default_factory=dict)
    embedding: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    reranker: dict[str, Any] = Field(default_factory=dict)
    query_transform: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# 5. Top-Level Experiment Specification (YAML Document Representation)
# ============================================================================


class ExperimentConfig(BaseModel):
    """Top-level evaluation study specification loaded from experiment.yaml."""

    metadata: ExperimentMetadata = Field(default_factory=ExperimentMetadata)
    dataset: DatasetConfig
    parameters: SweepParameters = Field(default_factory=SweepParameters)

    @classmethod
    def from_yaml(cls, yaml_content: str | Path) -> "ExperimentConfig":
        """Parse and validate experiment.yaml string or file path."""
        if isinstance(yaml_content, Path):
            with open(yaml_content, encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
        else:
            raw_data = yaml.safe_load(yaml_content)

        if not isinstance(raw_data, dict):
            raise ValueError("experiment.yaml must contain a top-level mapping")

        # Extract metadata if provided at top-level
        meta_dict = raw_data.get("metadata", {})
        if not isinstance(meta_dict, dict):
            meta_dict = {}

        for top_key in ("name", "description", "author", "tags", "version"):
            if top_key in raw_data and top_key not in meta_dict:
                meta_dict[top_key] = raw_data[top_key]

        metadata = ExperimentMetadata(**meta_dict)

        # Extract dataset
        dataset_data = raw_data.get("dataset")
        if not dataset_data or not isinstance(dataset_data, dict):
            raise ValueError("experiment.yaml must contain a 'dataset' configuration section")
        dataset = DatasetConfig(**dataset_data)

        # Extract parameters
        params_data = raw_data.get("parameters", {})
        if not isinstance(params_data, dict):
            params_data = {}

        # If parameters were defined directly at top-level under component names:
        for comp in (
            "chunking",
            "embedding",
            "retrieval",
            "reranker",
            "query_transform",
            "context",
            "generation",
            "evaluation",
        ):
            if comp in raw_data and comp not in params_data:
                params_data[comp] = raw_data[comp]

        parameters = SweepParameters(**params_data)

        return cls(metadata=metadata, dataset=dataset, parameters=parameters)

    def to_canonical_dict(self) -> dict[str, Any]:
        """Extract semantic experiment specification excluding run metadata."""
        return {
            "dataset": self.dataset.model_dump(),
            "parameters": self.parameters.model_dump(),
        }

    def canonical_json(self) -> str:
        """Deterministic JSON representation of semantic sweep parameters."""
        return canonical_json_dump(self.to_canonical_dict())

    def compute_configuration_hash(self) -> str:
        """Compute SHA-256 hash of canonical semantic experiment configuration."""
        return compute_hash(self.to_canonical_dict())
