"""Unit tests for Phase F2: Cartesian Orchestrator, Topology Resolution, and Dry-Run Planning."""

import pytest

from app.engine.orchestrator import (
    CartesianExpander,
    ConfigurationExpansionError,
    DryRunPlanner,
    RetrievalTopology,
    compute_cache_identity,
    resolve_topology,
    validate_component_compatibility,
)
from app.schemas.experiment import (
    DatasetConfig,
    ExperimentConfig,
    ExperimentMetadata,
    SweepParameters,
)


def create_baseline_experiment() -> ExperimentConfig:
    """Helper creating a baseline experiment configuration for testing."""
    return ExperimentConfig(
        metadata=ExperimentMetadata(
            name="Baseline Benchmark",
            description="Testing F2 Cartesian Orchestration",
            author="Test Runner",
            tags=["unit-test", "f2"],
        ),
        dataset=DatasetConfig(dataset_id="tech_docs_eval", dataset_version_id="v1.0.0"),
        parameters=SweepParameters(
            chunking={"strategy": "recursive", "chunk_size": 512, "chunk_overlap": 50},
            embedding={"provider": "fastembed", "model_name": "BAAI/bge-small-en-v1.5"},
            retrieval={"mode": "hybrid", "top_k": 10},
            reranker={"enabled": True, "strategy": "flashrank", "top_n": 5},
            query_transform={"strategy": "none"},
            context={"token_budget": 2048, "reorder_strategy": "standard"},
            generation={"model_name": "mock"},
            evaluation={"metrics": ["precision", "recall", "ndcg"], "k_values": [5, 10]},
        ),
    )


def test_empty_sweep_produces_single_baseline_point() -> None:
    """Verify that an empty sweep produces exactly 1 baseline pipeline point."""
    exp = create_baseline_experiment()
    expanded = CartesianExpander.expand(exp)
    assert len(expanded) == 1
    assert expanded[0].chunking.chunk_size == 512
    assert expanded[0].retrieval.mode == "hybrid"

    baseline = CartesianExpander.get_baseline_pipeline(exp)
    assert expanded[0].compute_configuration_hash() == baseline.compute_configuration_hash()


def test_single_sweep_dimension_preserves_order() -> None:
    """Verify that sweeping a single dimension preserves the exact declaration order."""
    exp = create_baseline_experiment()
    exp.parameters.chunking["chunk_size"] = [256, 512, 1024]

    expanded = CartesianExpander.expand(exp)
    assert len(expanded) == 3
    assert [p.chunking.chunk_size for p in expanded] == [256, 512, 1024]
    # Other dimensions retain baseline values
    for p in expanded:
        assert p.chunking.chunk_overlap == 50
        assert p.retrieval.mode == "hybrid"


def test_multiple_sweep_dimensions_deterministic_ordering() -> None:
    """Verify deterministic Cartesian ordering across multiple dimensions."""
    exp = create_baseline_experiment()
    exp.parameters.chunking["chunk_size"] = [256, 512]
    exp.parameters.retrieval["mode"] = ["dense", "bm25"]
    exp.parameters.retrieval["top_k"] = [5, 10]

    expanded = CartesianExpander.expand(exp)
    assert len(expanded) == 2 * 2 * 2  # 8 points

    # The canonical dimension order evaluates chunking before retrieval
    # and within retrieval: mode (alphabetical 'm') before top_k (alphabetical 't')
    expected_tuples = [
        (256, "dense", 5),
        (256, "dense", 10),
        (256, "bm25", 5),
        (256, "bm25", 10),
        (512, "dense", 5),
        (512, "dense", 10),
        (512, "bm25", 5),
        (512, "bm25", 10),
    ]

    actual_tuples = [(p.chunking.chunk_size, p.retrieval.mode, p.retrieval.top_k) for p in expanded]
    assert actual_tuples == expected_tuples


def test_cartesian_expansion_invalid_combination_handling() -> None:
    """Verify error raising and skipping behavior on invalid parameter combinations."""
    exp = create_baseline_experiment()
    # chunk_overlap (200) >= chunk_size (100) is invalid
    exp.parameters.chunking["chunk_size"] = [100, 500]
    exp.parameters.chunking["chunk_overlap"] = [200]

    with pytest.raises(ConfigurationExpansionError, match="must be strictly less than chunk_size"):
        CartesianExpander.expand(exp, skip_invalid=False)

    # When skip_invalid=True, the invalid combination (100, 200) is omitted
    valid_points = CartesianExpander.expand(exp, skip_invalid=True)
    assert len(valid_points) == 1
    assert valid_points[0].chunking.chunk_size == 500
    assert valid_points[0].chunking.chunk_overlap == 200


def test_pipeline_point_hashes() -> None:
    """Verify that all generated pipeline points have valid, distinct configuration hashes."""
    exp = create_baseline_experiment()
    exp.parameters.chunking["chunk_size"] = [256, 512]
    exp.parameters.retrieval["mode"] = ["dense", "bm25"]

    expanded = CartesianExpander.expand(exp)
    hashes = [p.compute_configuration_hash() for p in expanded]

    assert len(hashes) == 4
    # All 4 points are semantically distinct, so all 4 hashes must be unique
    assert len(set(hashes)) == 4
    for h in hashes:
        assert len(h) == 64
        int(h, 16)  # Must be valid hex


def test_dataset_version_aware_cache_identity() -> None:
    """Verify that cache identity strictly incorporates dataset id and version hash."""
    exp = create_baseline_experiment()
    baseline = CartesianExpander.get_baseline_pipeline(exp)
    config_hash = baseline.compute_configuration_hash()

    id_v1 = compute_cache_identity(
        dataset_id="squad_v2",
        dataset_version_hash="hash_version_1",
        pipeline_config_hash=config_hash,
    )
    id_v2 = compute_cache_identity(
        dataset_id="squad_v2",
        dataset_version_hash="hash_version_2",
        pipeline_config_hash=config_hash,
    )
    id_other_dataset = compute_cache_identity(
        dataset_id="ms_marco",
        dataset_version_hash="hash_version_1",
        pipeline_config_hash=config_hash,
    )

    assert id_v1.cache_key == f"squad_v2@hash_version_1:{config_hash}"
    assert id_v2.cache_key == f"squad_v2@hash_version_2:{config_hash}"

    # Critical invariant: Same pipeline configuration hash MUST yield different cache identities
    # when dataset version or dataset id changes.
    assert id_v1.cache_hash != id_v2.cache_hash
    assert id_v1.cache_hash != id_other_dataset.cache_hash

    # Empty inputs must be rejected
    with pytest.raises(ValueError, match="dataset_id cannot be empty"):
        compute_cache_identity("", "v1", config_hash)
    with pytest.raises(ValueError, match="dataset_version_hash cannot be empty"):
        compute_cache_identity("ds", "", config_hash)
    with pytest.raises(ValueError, match="pipeline_config_hash cannot be empty"):
        compute_cache_identity("ds", "v1", "")


def test_phase_c_retrieval_topology_mapping() -> None:
    """Verify explicit mapping of retrieval modes and rerankers onto Phase C topologies."""
    # 1. Dense (reranker disabled)
    topo, ret_cls = resolve_topology("dense", reranker_enabled=False)
    assert topo == RetrievalTopology.DENSE.value
    assert ret_cls == "DenseRetriever"

    # 2. BM25 / Sparse (reranker disabled) -> mapped to BM25Retriever without new abstraction
    topo, ret_cls = resolve_topology("sparse", reranker_enabled=False)
    assert topo == RetrievalTopology.BM25.value
    assert ret_cls == "BM25Retriever"

    topo, ret_cls = resolve_topology("bm25", reranker_enabled=False)
    assert topo == RetrievalTopology.BM25.value
    assert ret_cls == "BM25Retriever"

    # 3. Hybrid (reranker disabled)
    topo, ret_cls = resolve_topology("hybrid", reranker_enabled=False)
    assert topo == RetrievalTopology.HYBRID.value
    assert ret_cls == "HybridRetriever"

    # 4. Dense + Reranker
    topo, ret_cls = resolve_topology("dense", reranker_enabled=True)
    assert topo == RetrievalTopology.DENSE_RERANKER.value
    assert ret_cls == "DenseRetriever"

    # 5. BM25 + Reranker
    topo, ret_cls = resolve_topology("sparse", reranker_enabled=True)
    assert topo == RetrievalTopology.BM25_RERANKER.value
    assert ret_cls == "BM25Retriever"

    # 6. Hybrid + Reranker
    topo, ret_cls = resolve_topology("hybrid", reranker_enabled=True)
    assert topo == RetrievalTopology.HYBRID_RERANKER.value
    assert ret_cls == "HybridRetriever"

    # Invalid mode
    with pytest.raises(ValueError, match="Unsupported retrieval mode"):
        resolve_topology("random_unsupported_mode", reranker_enabled=False)


def test_cross_component_compatibility_validation() -> None:
    """Verify detection of valid configurations and cross-component incompatibilities."""
    exp = create_baseline_experiment()
    baseline = CartesianExpander.get_baseline_pipeline(exp)
    result = validate_component_compatibility(baseline)
    assert result.is_valid is True
    assert result.topology == RetrievalTopology.HYBRID_RERANKER.value
    assert result.resolved_retriever_class == "HybridRetriever"
    assert result.resolved_reranker_class == "FlashRankReranker"
    assert result.resolved_chunker_class == "RecursiveCharacterChunker"
    assert result.resolved_transform_class == "IdentityTransformer"
    assert len(result.errors) == 0

    # Incompatibility: reranker top_n > retrieval top_k
    incompatible_pipe = baseline.model_copy(deep=True)
    incompatible_pipe.retrieval.top_k = 5
    incompatible_pipe.reranker.top_n = 10
    bad_result = validate_component_compatibility(incompatible_pipe)
    assert bad_result.is_valid is False
    assert any("cannot exceed retrieval.top_k" in err for err in bad_result.errors)

    # Incompatibility: unresolvable chunker strategy
    bad_chunker = baseline.model_copy(deep=True)
    bad_chunker.chunking.strategy = "non_existent_chunker"
    chunker_result = validate_component_compatibility(bad_chunker)
    assert chunker_result.is_valid is False
    assert any("Unregistered chunking strategy" in err for err in chunker_result.errors)


def test_dry_run_planning_without_side_effects() -> None:
    """Verify that DryRunPlanner produces a complete plan without executing workloads."""
    exp = create_baseline_experiment()
    exp.parameters.chunking["chunk_size"] = [256, 512]
    exp.parameters.retrieval["mode"] = ["dense", "hybrid"]

    plan = DryRunPlanner.plan(
        config=exp,
        dataset_version_hash="v1_sha256_mock_hash",
        query_count=50,
    )

    assert plan.experiment_name == "Baseline Benchmark"
    assert plan.dataset_id == "tech_docs_eval"
    assert plan.dataset_version_hash == "v1_sha256_mock_hash"
    assert plan.total_planned_pipelines == 4
    assert plan.total_compatible_pipelines == 4
    assert plan.is_fully_compatible is True
    # 4 pipelines * 50 queries = 200 evaluations
    assert plan.estimated_query_evaluations == 200

    assert len(plan.planned_runs) == 4
    for idx, run in enumerate(plan.planned_runs):
        assert run.index == idx
        assert run.is_compatible is True
        assert run.cache_key.startswith("tech_docs_eval@v1_sha256_mock_hash:")
        assert len(run.cache_hash) == 64
        assert run.topology in (
            RetrievalTopology.DENSE_RERANKER.value,
            RetrievalTopology.HYBRID_RERANKER.value,
        )

    # Run #0 is identical to the baseline point -> empty diff
    diff_0 = plan.planned_runs[0].parameter_diff
    assert diff_0 == {}

    # Run #3 diff against baseline
    diff_3 = plan.planned_runs[3].parameter_diff
    assert diff_3 == {"chunking.chunk_size": 512, "retrieval.mode": "hybrid"}


def test_parameter_diff_computation() -> None:
    """Verify parameter delta computation against baseline."""
    baseline = CartesianExpander.get_baseline_pipeline(create_baseline_experiment())
    target = baseline.model_copy(deep=True)
    target.chunking.chunk_size = 1024
    target.retrieval.mode = "bm25"

    diff = CartesianExpander.compute_diff(baseline, target)
    assert diff == {
        "chunking.chunk_size": 1024,
        "retrieval.mode": "bm25",
    }

    # Identical target produces empty diff
    assert CartesianExpander.compute_diff(baseline, baseline) == {}
