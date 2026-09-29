"""Unit tests for Phase F8: Empirical Evaluation Protocols & Compatibility Hardening.

Verifies:
1. Default protocol versions are populated automatically.
2. BenchmarkQuerySet.compute_benchmark_hash changes when protocol versions change.
3. IncompatibleRunsError is raised by AlignmentValidator on evaluation_protocol_version divergence.
4. IncompatibleRunsError is raised by AlignmentValidator on metric_definition_version divergence.
5. Identical protocol versions allow alignment and set protocol_match=True in ComparisonAlignment.
"""

from unittest.mock import MagicMock

import pytest

from app.core.protocols import DEFAULT_EVALUATION_PROTOCOL, DEFAULT_METRIC_PROTOCOL
from app.engine.analysis.alignment import AlignmentValidator, IncompatibleRunsError
from app.models.entities import ExperimentRun
from app.schemas.benchmark import BenchmarkQuery, BenchmarkQuerySet, GroundTruthPassage


def test_protocol_constants_defined() -> None:
    """Ensure canonical protocol constants are non-empty strings."""
    assert DEFAULT_EVALUATION_PROTOCOL == "ragbench-protocol-v1.0"
    assert DEFAULT_METRIC_PROTOCOL == "metrics-v1.0"


def test_benchmark_hash_includes_protocol_versions() -> None:
    """Changing protocol versions in BenchmarkQuerySet must change its computed hash."""
    q = BenchmarkQuery(
        query_id="q1",
        query="What is RAGBench?",
        ground_truth_answer="An evaluation framework.",
        ground_truth_passages=[
            GroundTruthPassage(
                doc_id="doc1",
                text_snippet="RAGBench is an evaluation framework.",
            )
        ],
    )
    b1 = BenchmarkQuerySet(
        name="test-bench",
        version=1,
        queries=[q],
        evaluation_protocol_version="ragbench-protocol-v1.0",
        metric_definition_version="metrics-v1.0",
    )
    b2 = BenchmarkQuerySet(
        name="test-bench",
        version=1,
        queries=[q],
        evaluation_protocol_version="ragbench-protocol-v2.0",
        metric_definition_version="metrics-v1.0",
    )
    b3 = BenchmarkQuerySet(
        name="test-bench",
        version=1,
        queries=[q],
        evaluation_protocol_version="ragbench-protocol-v1.0",
        metric_definition_version="metrics-v2.0",
    )

    hash1 = b1.compute_benchmark_hash()
    hash2 = b2.compute_benchmark_hash()
    hash3 = b3.compute_benchmark_hash()

    assert hash1 != hash2
    assert hash1 != hash3
    assert hash2 != hash3


def test_alignment_validator_rejects_divergent_eval_protocol() -> None:
    """AlignmentValidator raises IncompatibleRunsError if evaluation_protocol_version differs."""
    run_a = MagicMock(spec=ExperimentRun)
    run_a.evaluation_protocol_version = "ragbench-protocol-v1.0"
    run_a.metric_definition_version = "metrics-v1.0"
    run_a.benchmark_hash = "bench_hash_shared"
    run_a.experiment = MagicMock()
    run_a.experiment.dataset_version_id = "dv1"
    run_a.experiment.dataset_version = MagicMock(content_hash="d_hash_shared")

    run_b = MagicMock(spec=ExperimentRun)
    run_b.evaluation_protocol_version = "ragbench-protocol-v2.0"
    run_b.metric_definition_version = "metrics-v1.0"
    run_b.benchmark_hash = "bench_hash_shared"
    run_b.experiment = MagicMock()
    run_b.experiment.dataset_version_id = "dv1"
    run_b.experiment.dataset_version = MagicMock(content_hash="d_hash_shared")

    scores_a = {"q1": 1.0, "q2": 0.5}
    scores_b = {"q1": 0.8, "q2": 0.6}

    with pytest.raises(IncompatibleRunsError, match="evaluation_protocol_version differs"):
        AlignmentValidator.validate_and_align(
            run_a=run_a,
            run_b=run_b,
            scores_a_by_query=scores_a,
            scores_b_by_query=scores_b,
            metric_name="precision_at_k",
            k=5,
        )


def test_alignment_validator_rejects_divergent_metric_protocol() -> None:
    """AlignmentValidator must raise IncompatibleRunsError if metric_definition_version differs."""
    run_a = MagicMock(spec=ExperimentRun)
    run_a.evaluation_protocol_version = "ragbench-protocol-v1.0"
    run_a.metric_definition_version = "metrics-v1.0"
    run_a.benchmark_hash = "bench_hash_shared"
    run_a.experiment = MagicMock()
    run_a.experiment.dataset_version_id = "dv1"
    run_a.experiment.dataset_version = MagicMock(content_hash="d_hash_shared")

    run_b = MagicMock(spec=ExperimentRun)
    run_b.evaluation_protocol_version = "ragbench-protocol-v1.0"
    run_b.metric_definition_version = "metrics-custom-v1.0"
    run_b.benchmark_hash = "bench_hash_shared"
    run_b.experiment = MagicMock()
    run_b.experiment.dataset_version_id = "dv1"
    run_b.experiment.dataset_version = MagicMock(content_hash="d_hash_shared")

    scores_a = {"q1": 1.0, "q2": 0.5}
    scores_b = {"q1": 0.8, "q2": 0.6}

    with pytest.raises(IncompatibleRunsError, match="metric_definition_version differs"):
        AlignmentValidator.validate_and_align(
            run_a=run_a,
            run_b=run_b,
            scores_a_by_query=scores_a,
            scores_b_by_query=scores_b,
            metric_name="precision_at_k",
            k=5,
        )


def test_alignment_validator_accepts_matching_protocols() -> None:
    """Matching protocols succeed alignment with protocol_match=True."""
    run_a = MagicMock(spec=ExperimentRun)
    run_a.evaluation_protocol_version = "ragbench-protocol-v1.0"
    run_a.metric_definition_version = "metrics-v1.0"
    run_a.benchmark_hash = "bench_hash_shared"
    run_a.experiment = MagicMock()
    run_a.experiment.dataset_version_id = "dv1"
    run_a.experiment.dataset_version = MagicMock(content_hash="d_hash_shared")

    run_b = MagicMock(spec=ExperimentRun)
    run_b.evaluation_protocol_version = "ragbench-protocol-v1.0"
    run_b.metric_definition_version = "metrics-v1.0"
    run_b.benchmark_hash = "bench_hash_shared"
    run_b.experiment = MagicMock()
    run_b.experiment.dataset_version_id = "dv1"
    run_b.experiment.dataset_version = MagicMock(content_hash="d_hash_shared")

    scores_a = {"q1": 1.0, "q2": 0.5}
    scores_b = {"q1": 0.8, "q2": 0.6}

    alignment = AlignmentValidator.validate_and_align(
        run_a=run_a,
        run_b=run_b,
        scores_a_by_query=scores_a,
        scores_b_by_query=scores_b,
        metric_name="precision_at_k",
        k=5,
    )

    assert alignment.protocol_match is True
    assert alignment.evaluation_protocol_version_a == "ragbench-protocol-v1.0"
    assert alignment.evaluation_protocol_version_b == "ragbench-protocol-v1.0"
    assert alignment.metric_definition_version_a == "metrics-v1.0"
    assert alignment.metric_definition_version_b == "metrics-v1.0"
    assert alignment.n_paired == 2
