"""Unit tests for Experiment and Pipeline configuration schemas and canonical hashing (Phase F1)."""

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.experiment import (
    ChunkingConfig,
    ContextConfig,
    DatasetConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentConfig,
    GenerationConfig,
    PipelineConfig,
    QueryTransformConfig,
    RerankerConfig,
    RetrievalConfig,
    canonical_json_dump,
)


@pytest.fixture
def base_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        dataset=DatasetConfig(dataset_id="msmarco", dataset_version_id="v1.0"),
        chunking=ChunkingConfig(strategy="fixed", chunk_size=512, chunk_overlap=64),
        embedding=EmbeddingConfig(provider="fastembed", model_name="BAAI/bge-small-en-v1.5"),
        retrieval=RetrievalConfig(mode="dense", top_k=10),
        reranker=RerankerConfig(enabled=False, strategy="none"),
        query_transform=QueryTransformConfig(strategy="none"),
        context=ContextConfig(token_budget=2048, reorder_strategy="standard"),
        generation=GenerationConfig(model_name="mock"),
        evaluation=EvaluationConfig(metrics=["recall", "ndcg"], k_values=[1, 3, 5, 10]),
    )


# =========================================================================
# 1. Validation and Rejection Tests
# =========================================================================


def test_chunking_validation_rejections() -> None:
    # Invalid strategy
    with pytest.raises(ValidationError, match="Invalid chunking strategy"):
        ChunkingConfig(strategy="magic_chunker")

    # Chunk overlap >= chunk size
    with pytest.raises(
        ValidationError, match="chunk_overlap .* must be strictly less than chunk_size"
    ):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)

    # Chunk size <= 0
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=0)


def test_retrieval_validation_rejections() -> None:
    # Invalid mode
    with pytest.raises(ValidationError, match="Invalid retrieval mode"):
        RetrievalConfig(mode="inverted_index")

    # Invalid fusion
    with pytest.raises(ValidationError, match="Invalid hybrid fusion method"):
        RetrievalConfig(hybrid_fusion="linear_sum")

    # Top-k <= 0
    with pytest.raises(ValidationError):
        RetrievalConfig(top_k=0)


def test_evaluation_validation_rejections() -> None:
    # Invalid metric name
    with pytest.raises(ValidationError, match="Invalid metric 'hallucination_score'"):
        EvaluationConfig(metrics=["recall", "hallucination_score"])

    # Invalid k values
    with pytest.raises(ValidationError, match="k cutoff must be strictly positive"):
        EvaluationConfig(k_values=[0, 5])

    with pytest.raises(
        ValidationError, match="k_values must contain at least one positive integer"
    ):
        EvaluationConfig(k_values=[])


# =========================================================================
# 2. Canonical JSON Serialization & Hash Determinism
# =========================================================================


def test_canonical_json_ordering_and_separators() -> None:
    data_a = {"b": 2, "a": 1, "nested": {"z": "val_z", "y": "val_y"}}
    data_b = {"nested": {"y": "val_y", "z": "val_z"}, "a": 1, "b": 2}

    json_a = canonical_json_dump(data_a)
    json_b = canonical_json_dump(data_b)

    # 1. Keys sorted recursively
    assert json_a == json_b
    assert json_a == '{"a":1,"b":2,"nested":{"y":"val_y","z":"val_z"}}'

    # 2. No whitespace around separators
    assert ", " not in json_a
    assert ": " not in json_a


def test_canonical_json_unicode_preservation() -> None:
    data = {"language": "?????", "author": "???? ?????"}
    canonical = canonical_json_dump(data)
    # Unicode preserved verbatim (ensure_ascii=False)
    assert "?????" in canonical
    assert "\\u" not in canonical


def test_configuration_hash_reproducibility(base_pipeline_config: PipelineConfig) -> None:
    hashes = [base_pipeline_config.compute_configuration_hash() for _ in range(10)]
    assert len(set(hashes)) == 1
    # Hash matches sha256 of canonical json
    expected = hashlib.sha256(base_pipeline_config.canonical_json().encode("utf-8")).hexdigest()
    assert hashes[0] == expected


# =========================================================================
# 3. Semantic Difference Sensitivity vs. Metadata Insensitivity
# =========================================================================


def test_semantic_parameters_change_hash(base_pipeline_config: PipelineConfig) -> None:
    base_hash = base_pipeline_config.compute_configuration_hash()

    # 1. Chunk size change
    c1 = base_pipeline_config.model_copy(deep=True)
    c1.chunking.chunk_size = 256
    assert c1.compute_configuration_hash() != base_hash

    # 2. Retrieval mode change (dense -> hybrid)
    c2 = base_pipeline_config.model_copy(deep=True)
    c2.retrieval.mode = "hybrid"
    assert c2.compute_configuration_hash() != base_hash

    # 3. Reranker enabled
    c3 = base_pipeline_config.model_copy(deep=True)
    c3.reranker.enabled = True
    c3.reranker.strategy = "flashrank"
    assert c3.compute_configuration_hash() != base_hash

    # 4. Query transform strategy change
    c4 = base_pipeline_config.model_copy(deep=True)
    c4.query_transform.strategy = "hyde"
    assert c4.compute_configuration_hash() != base_hash

    # 5. Context budget change
    c5 = base_pipeline_config.model_copy(deep=True)
    c5.context.token_budget = 4096
    assert c5.compute_configuration_hash() != base_hash

    # 6. Generation model change
    c6 = base_pipeline_config.model_copy(deep=True)
    c6.generation.model_name = "llama-3-8b"
    assert c6.compute_configuration_hash() != base_hash

    # 7. Dataset version change
    c7 = base_pipeline_config.model_copy(deep=True)
    c7.dataset.dataset_version_id = "v2.0"
    assert c7.compute_configuration_hash() != base_hash


def test_non_semantic_metadata_does_not_change_hash() -> None:
    yaml_content_1 = """
metadata:
  name: "Study A"
  description: "Initial hypothesis test on chunking"
  author: "Alice"
  tags: ["preliminary", "v1"]
dataset:
  dataset_id: "squad"
  dataset_version_id: "v1.0"
parameters:
  retrieval:
    mode: "dense"
"""
    yaml_content_2 = """
metadata:
  name: "Study B (Renamed)"
  description: "Completely different documentation text"
  author: "Bob"
  tags: ["production"]
dataset:
  dataset_id: "squad"
  dataset_version_id: "v1.0"
parameters:
  retrieval:
    mode: "dense"
"""
    exp_1 = ExperimentConfig.from_yaml(yaml_content_1)
    exp_2 = ExperimentConfig.from_yaml(yaml_content_2)

    # Metadata differs
    assert exp_1.metadata.name != exp_2.metadata.name
    assert exp_1.metadata.description != exp_2.metadata.description

    # Configuration hash is identical because semantic parameters match
    assert exp_1.compute_configuration_hash() == exp_2.compute_configuration_hash()


# =========================================================================
# 4. YAML File Loading & Sweep Parsing
# =========================================================================


def test_from_yaml_file() -> None:
    temp_yaml = Path(__file__).parent / "_temp_exp.yaml"
    try:
        temp_yaml.write_text(
            """
name: "Retrieval Sweep"
dataset:
  dataset_id: "tech_docs"
  dataset_version_id: "2026.09"
parameters:
  chunking:
    strategy: ["fixed", "recursive"]
    chunk_size: [256, 512]
  retrieval:
    mode: ["dense", "bm25", "hybrid"]
""",
            encoding="utf-8",
        )

        config = ExperimentConfig.from_yaml(temp_yaml)
        assert config.metadata.name == "Retrieval Sweep"
        assert config.dataset.dataset_id == "tech_docs"
        assert config.dataset.dataset_version_id == "2026.09"
        assert config.parameters.chunking["strategy"] == ["fixed", "recursive"]
        assert config.parameters.retrieval["mode"] == ["dense", "bm25", "hybrid"]
        assert len(config.compute_configuration_hash()) == 64
    finally:
        if temp_yaml.exists():
            temp_yaml.unlink()


def test_from_yaml_missing_dataset_raises() -> None:
    invalid_yaml = """
name: "No dataset experiment"
parameters:
  retrieval:
    mode: "dense"
"""
    with pytest.raises(
        ValueError, match="experiment.yaml must contain a 'dataset' configuration section"
    ):
        ExperimentConfig.from_yaml(invalid_yaml)
