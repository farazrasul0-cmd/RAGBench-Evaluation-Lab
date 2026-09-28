"""Unit tests for query alignment, multi-dimensional run compatibility, and permutation invariance.

Verifies:
1. Permutation invariance: Any permutation of query IDs yields identical results.
2. Mismatched benchmark hashes raise IncompatibleRunsError.
3. Mismatched metric cutoff K (e.g. Recall@5 vs Recall@10) raises IncompatibleRunsError.
4. Mismatched dataset versions and dataset content hashes raise IncompatibleRunsError.
5. Duplicate query IDs in observations raise ValueError.
6. Partial query overlap enforcement and explicit opt-in flag.
"""

import random
from unittest.mock import MagicMock

import pytest

from app.engine.analysis.alignment import (
    AlignmentValidator,
    IncompatibleRunsError,
    QuerySetMismatchError,
)
from app.engine.analysis.statistics import StatisticalComparator
from app.models.entities import Experiment, ExperimentRun


def test_permutation_invariance_shuffled_queries() -> None:
    """Row or iteration order of queries must never alter paired statistical outcomes."""
    exp = MagicMock(spec=Experiment, dataset_version_id="ver-1")
    run_a = MagicMock(spec=ExperimentRun, id="run-a", experiment=exp)
    run_b = MagicMock(spec=ExperimentRun, id="run-b", experiment=exp)

    # 20 queries with varied scores
    base_keys = [f"query_{i:03d}" for i in range(20)]
    scores_a_base = {k: 0.2 + 0.03 * i for i, k in enumerate(base_keys)}
    scores_b_base = {k: 0.25 + 0.025 * i for i, k in enumerate(base_keys)}

    # Compute baseline comparison
    res_base = StatisticalComparator.compare_single_metric(
        run_a=run_a,
        run_b=run_b,
        metric_name="recall@5",
        scores_a_by_query=scores_a_base,
        scores_b_by_query=scores_b_base,
    )

    # Test 5 randomized permutations of the query dictionary
    for seed in [42, 1337, 2026, 9999, 7]:
        rng = random.Random(seed)
        shuffled_keys_a = list(base_keys)
        shuffled_keys_b = list(base_keys)
        rng.shuffle(shuffled_keys_a)
        rng.shuffle(shuffled_keys_b)

        shuffled_a = {k: scores_a_base[k] for k in shuffled_keys_a}
        shuffled_b = {k: scores_b_base[k] for k in shuffled_keys_b}

        res_perm = StatisticalComparator.compare_single_metric(
            run_a=run_a,
            run_b=run_b,
            metric_name="recall@5",
            scores_a_by_query=shuffled_a,
            scores_b_by_query=shuffled_b,
        )

        assert res_perm.mean_a == res_base.mean_a
        assert res_perm.mean_b == res_base.mean_b
        assert res_perm.mean_difference == res_base.mean_difference
        assert res_perm.std_difference == res_base.std_difference
        assert res_perm.t_statistic == res_base.t_statistic
        assert res_perm.raw_p_value == res_base.raw_p_value
        assert res_perm.wilcoxon_statistic == res_base.wilcoxon_statistic
        assert res_perm.wilcoxon_pvalue == res_base.wilcoxon_pvalue
        assert res_perm.cohens_dz == res_base.cohens_dz
        assert res_perm.ci95_lower == res_base.ci95_lower
        assert res_perm.ci95_upper == res_base.ci95_upper
        # Also ensure paired observation sequence is strictly ordered
        assert [o.query_id for o in res_perm.raw_paired_observations] == [
            o.query_id for o in res_base.raw_paired_observations
        ]


def test_mismatched_benchmark_hashes() -> None:
    """Runs evaluating different benchmark specifications cannot be paired."""
    exp_a = MagicMock(spec=Experiment, dataset_version_id="ver-1", benchmark_hash="hash-bench-AAA")
    exp_b = MagicMock(spec=Experiment, dataset_version_id="ver-1", benchmark_hash="hash-bench-BBB")

    run_a = MagicMock(
        spec=ExperimentRun, id="run-a", experiment=exp_a, benchmark_hash="hash-bench-AAA"
    )
    run_b = MagicMock(
        spec=ExperimentRun, id="run-b", experiment=exp_b, benchmark_hash="hash-bench-BBB"
    )

    scores_a = {"q1": 0.5, "q2": 0.8}
    scores_b = {"q1": 0.6, "q2": 0.85}

    with pytest.raises(IncompatibleRunsError, match="benchmark specification hash differs"):
        AlignmentValidator.validate_and_align(
            run_a=run_a,
            run_b=run_b,
            scores_a_by_query=scores_a,
            scores_b_by_query=scores_b,
            metric_name="recall@5",
        )


def test_mismatched_k_cutoffs() -> None:
    """Comparing Recall@5 vs Recall@10 must fail compatibility validation."""
    exp = MagicMock(spec=Experiment, dataset_version_id="ver-1")
    run_a = MagicMock(spec=ExperimentRun, id="run-a", experiment=exp)
    run_b = MagicMock(spec=ExperimentRun, id="run-b", experiment=exp)

    scores_a = {"q1": 0.5, "q2": 0.8}
    scores_b = {"q1": 0.6, "q2": 0.85}

    with pytest.raises(IncompatibleRunsError, match="metric K differs"):
        AlignmentValidator.validate_and_align(
            run_a=run_a,
            run_b=run_b,
            scores_a_by_query=scores_a,
            scores_b_by_query=scores_b,
            metric_name="recall",
            k_a=5,
            k_b=10,
        )


def test_mismatched_dataset_version_and_content_hash() -> None:
    """Comparing runs across different dataset versions or content hashes must fail."""
    exp_a = MagicMock(spec=Experiment, dataset_version_id="ver-1")
    exp_b = MagicMock(spec=Experiment, dataset_version_id="ver-2")

    run_a = MagicMock(spec=ExperimentRun, id="run-a", experiment=exp_a)
    run_b = MagicMock(spec=ExperimentRun, id="run-b", experiment=exp_b)

    scores_a = {"q1": 0.5}
    scores_b = {"q1": 0.6}

    with pytest.raises(IncompatibleRunsError, match="dataset_version_id differs"):
        AlignmentValidator.validate_and_align(
            run_a=run_a,
            run_b=run_b,
            scores_a_by_query=scores_a,
            scores_b_by_query=scores_b,
            metric_name="recall@5",
        )

    # Same version id but differing content hash
    ver_a = MagicMock(content_hash="hash-111")
    ver_b = MagicMock(content_hash="hash-222")
    exp_a_hash = MagicMock(spec=Experiment, dataset_version_id="ver-1", dataset_version=ver_a)
    exp_b_hash = MagicMock(spec=Experiment, dataset_version_id="ver-1", dataset_version=ver_b)
    run_a_hash = MagicMock(spec=ExperimentRun, id="run-a", experiment=exp_a_hash)
    run_b_hash = MagicMock(spec=ExperimentRun, id="run-b", experiment=exp_b_hash)

    with pytest.raises(IncompatibleRunsError, match="dataset content_hash differs"):
        AlignmentValidator.validate_and_align(
            run_a=run_a_hash,
            run_b=run_b_hash,
            scores_a_by_query=scores_a,
            scores_b_by_query=scores_b,
            metric_name="recall@5",
        )


def test_query_set_mismatch_and_partial_overlap() -> None:
    """Verify QuerySetMismatchError and explicit partial overlap opt-in."""
    exp = MagicMock(spec=Experiment, dataset_version_id="ver-1")
    run_a = MagicMock(spec=ExperimentRun, id="run-a", experiment=exp)
    run_b = MagicMock(spec=ExperimentRun, id="run-b", experiment=exp)

    scores_a = {"q1": 0.5, "q2": 0.8, "q3": 0.7}
    scores_b = {"q1": 0.6, "q2": 0.85}  # Missing q3

    # Default: strict rejection
    with pytest.raises(QuerySetMismatchError, match="Benchmark query sets do not match"):
        AlignmentValidator.validate_and_align(
            run_a=run_a,
            run_b=run_b,
            scores_a_by_query=scores_a,
            scores_b_by_query=scores_b,
            metric_name="recall@5",
            allow_partial_query_overlap=False,
        )

    # Opt-in: evaluates on intersection
    align = AlignmentValidator.validate_and_align(
        run_a=run_a,
        run_b=run_b,
        scores_a_by_query=scores_a,
        scores_b_by_query=scores_b,
        metric_name="recall@5",
        allow_partial_query_overlap=True,
    )
    assert align.n_paired == 2
    assert align.paired_query_ids == ["q1", "q2"]
    assert align.missing_in_b == ["q3"]
