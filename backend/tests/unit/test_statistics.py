"""Comprehensive unit and mathematical correctness tests for F7.1: Statistical Analysis Core.

Verifies:
1. Paired Student's t-test calculation against hand-computable reference values.
2. Degrees of freedom, two-tailed p-values, and edge cases (N=1, identical runs, constant delta).
3. Non-parametric Wilcoxon Signed-Rank test (exact distribution, ties, zeros, identical runs).
4. Cohen's dz paired effect size.
5. 95% Confidence Interval for mean paired difference.
6. Holm-Bonferroni and Benjamini-Hochberg multiple-comparison corrections.
7. Strict query alignment: permutation invariance, missing query detection, query set mismatch.
8. Dataset version compatibility enforcement.
9. Component-aware latency measurement and warm-up exclusion filtering.
"""

import math
from unittest.mock import MagicMock

import pytest

from app.engine.analysis.alignment import (
    AlignmentValidator,
    IncompatibleRunsError,
    QuerySetMismatchError,
)
from app.engine.analysis.statistics import (
    StatisticalComparator,
    benjamini_hochberg,
    holm_bonferroni,
    student_t_critical_value,
    student_t_two_tailed_p,
    wilcoxon_signed_rank_test,
)
from app.models.entities import Experiment, ExperimentRun


def test_student_t_mathematical_precision() -> None:
    """Verify Student's t CDF and two-tailed p-values match exact textbook values."""
    # df=10, t=2.228138837856 -> p = 0.05
    p10 = student_t_two_tailed_p(2.228138837856, 10)
    assert abs(p10 - 0.05) < 1e-5

    # df=30, t=2.042272456301 -> p = 0.05
    p30 = student_t_two_tailed_p(2.042272456301, 30)
    assert abs(p30 - 0.05) < 1e-5

    # t=0.0 -> p = 1.0 regardless of df
    assert student_t_two_tailed_p(0.0, 5) == 1.0
    assert student_t_two_tailed_p(0.0, 50) == 1.0

    # df <= 0 returns nan
    assert math.isnan(student_t_two_tailed_p(2.0, 0))


def test_student_t_critical_value() -> None:
    """Verify Student's t critical values for 95% confidence intervals."""
    t_crit_10 = student_t_critical_value(10, alpha=0.05)
    assert abs(t_crit_10 - 2.228139) < 1e-4

    t_crit_30 = student_t_critical_value(30, alpha=0.05)
    assert abs(t_crit_30 - 2.042272) < 1e-4

    t_crit_large = student_t_critical_value(1000, alpha=0.05)
    assert abs(t_crit_large - 1.962339) < 1e-3


def test_wilcoxon_exact_and_edge_cases() -> None:
    """Verify Wilcoxon signed-rank test exact distribution and degenerate edge cases."""
    # 1. Identical runs (all differences are 0): must be degenerate_identical, p=1.0
    stat_id, p_id, nz_id, z_id, method_id = wilcoxon_signed_rank_test([0.0, 0.0, 0.0, 0.0])
    assert stat_id == 0.0
    assert p_id == 1.0
    assert nz_id == 0
    assert z_id == 4
    assert method_id == "degenerate_identical"

    # 2. Perfect improvement: 5 strictly positive differences
    # Ranks: 1, 2, 3, 4, 5. W+ = 15, W- = 0. T = 0.
    # Total permutations: 2^5 = 32. Permutations with W <= 0: only 1 (empty set).
    # Two-sided p = 2 * (1 / 32) = 2 / 32 = 0.0625
    stat_5, p_5, nz_5, z_5, method_5 = wilcoxon_signed_rank_test([0.1, 0.2, 0.3, 0.4, 0.5])
    assert stat_5 == 0.0
    assert p_5 is not None
    assert abs(p_5 - 0.0625) < 1e-5
    assert nz_5 == 5
    assert z_5 == 0
    assert method_5 == "exact"

    # 3. Zeros in sample are dropped
    stat_z, p_z, nz_z, z_cnt, _ = wilcoxon_signed_rank_test([0.0, 0.1, 0.0, -0.2, 0.3])
    assert z_cnt == 2
    assert nz_z == 3

    # 4. Asymptotic approximation when N > 20
    large_diffs = [0.1 * i for i in range(1, 26)]
    stat_lg, p_lg, nz_lg, _, method_lg = wilcoxon_signed_rank_test(large_diffs)
    assert nz_lg == 25
    assert method_lg == "asymptotic"
    assert p_lg is not None and p_lg < 0.001


def test_multiple_comparison_corrections() -> None:
    """Verify Holm-Bonferroni and Benjamini-Hochberg correction logic."""
    raw_p = [0.01, 0.04, 0.03, 0.20]
    # Sorted: 0.01 (k=1), 0.03 (k=2), 0.04 (k=3), 0.20 (k=4)
    # Holm multipliers: 4, 3, 2, 1
    # Raw adj: 0.04, 0.09, 0.08 (enforced mono -> 0.09), 0.20
    # Expected Holm: [0.04, 0.09, 0.09, 0.20]
    holm_p = holm_bonferroni(raw_p)
    assert abs(holm_p[0] - 0.04) < 1e-5
    assert abs(holm_p[1] - 0.09) < 1e-5
    assert abs(holm_p[2] - 0.09) < 1e-5
    assert abs(holm_p[3] - 0.20) < 1e-5

    # BH FDR step-up
    # m = 4. Sorted: 0.01, 0.03, 0.04, 0.20
    # Multipliers m/k: 4/1=4, 4/2=2, 4/3=1.333, 4/4=1
    # Raw: 0.04, 0.06, 0.0533, 0.20
    # Step-up min backwards:
    # k=4: 0.20
    # k=3: min(0.20, 0.0533) = 0.0533
    # k=2: min(0.0533, 0.06) = 0.0533
    # k=1: min(0.0533, 0.04) = 0.04
    bh_p = benjamini_hochberg(raw_p)
    assert abs(bh_p[0] - 0.04) < 1e-4
    assert abs(bh_p[1] - 0.05333) < 1e-4
    assert abs(bh_p[2] - 0.05333) < 1e-4
    assert abs(bh_p[3] - 0.20) < 1e-4


def test_alignment_validation_and_rejections() -> None:
    """Verify strict query alignment and rejection of incompatible runs."""
    exp_a = MagicMock(spec=Experiment, dataset_version_id="ver-1")
    exp_b = MagicMock(spec=Experiment, dataset_version_id="ver-1")
    exp_incompat = MagicMock(spec=Experiment, dataset_version_id="ver-2")

    run_a = MagicMock(spec=ExperimentRun, id="run-a", experiment=exp_a)
    run_b = MagicMock(spec=ExperimentRun, id="run-b", experiment=exp_b)
    run_diff_ds = MagicMock(spec=ExperimentRun, id="run-diff", experiment=exp_incompat)

    scores_a = {"q1": 0.5, "q2": 0.8, "q3": 0.9}
    scores_b = {"q1": 0.6, "q2": 0.8, "q3": 0.95}

    # 1. Exact match passes
    align = AlignmentValidator.validate_and_align(
        run_a, run_b, scores_a, scores_b, metric_name="recall@5", k=5
    )
    assert align.n_paired == 3
    assert align.query_set_match is True
    assert align.dataset_version_match is True

    # 2. Permutation invariance: different dict insertion orders align deterministically
    scores_b_permuted = {"q3": 0.95, "q1": 0.6, "q2": 0.8}
    align_perm = AlignmentValidator.validate_and_align(
        run_a, run_b, scores_a, scores_b_permuted, metric_name="recall@5"
    )
    assert align_perm.paired_query_ids == ["q1", "q2", "q3"]

    # 3. Incompatible dataset version raises IncompatibleRunsError
    with pytest.raises(IncompatibleRunsError, match="dataset_version_id differs"):
        AlignmentValidator.validate_and_align(
            run_a, run_diff_ds, scores_a, scores_b, metric_name="recall@5"
        )

    # 4. Query set mismatch raises QuerySetMismatchError by default
    scores_b_missing = {"q1": 0.6, "q2": 0.8}
    with pytest.raises(QuerySetMismatchError, match="Benchmark query sets do not match"):
        AlignmentValidator.validate_and_align(
            run_a, run_b, scores_a, scores_b_missing, metric_name="recall@5"
        )

    # 5. Query set mismatch allowed with explicit flag
    align_partial = AlignmentValidator.validate_and_align(
        run_a,
        run_b,
        scores_a,
        scores_b_missing,
        metric_name="recall@5",
        allow_partial_query_overlap=True,
    )
    assert align_partial.n_paired == 2
    assert align_partial.missing_in_b == ["q3"]


def test_statistical_comparator_paired_t_and_cohens_dz() -> None:
    """Verify paired t-test, Cohen's dz, and 95% confidence intervals on concrete sample."""
    exp = MagicMock(spec=Experiment, dataset_version_id="ver-stats")
    run_a = MagicMock(spec=ExperimentRun, id="run-11111111", experiment=exp)
    run_b = MagicMock(spec=ExperimentRun, id="run-22222222", experiment=exp)

    # Hand-calculable sample:
    # A: [0.60, 0.70, 0.80, 0.65, 0.75]
    # B: [0.70, 0.80, 0.90, 0.75, 0.85]
    # Constant delta of +0.10 for all 5 queries
    scores_a = {f"q{i}": v for i, v in enumerate([0.60, 0.70, 0.80, 0.65, 0.75])}
    scores_b = {f"q{i}": v for i, v in enumerate([0.70, 0.80, 0.90, 0.75, 0.85])}

    res_const = StatisticalComparator.compare_single_metric(
        run_a, run_b, metric_name="recall@5", scores_a_by_query=scores_a, scores_b_by_query=scores_b
    )
    assert abs(res_const.mean_difference - 0.10) < 1e-6
    assert abs(res_const.std_difference - 0.0) < 1e-6
    assert res_const.ci95_lower == 0.10
    assert res_const.ci95_upper == 0.10

    # Variable differences:
    # D = [+0.10, +0.20, +0.15, -0.05, +0.10]
    # Mean D = (0.10 + 0.20 + 0.15 - 0.05 + 0.10) / 5 = 0.50 / 5 = 0.10
    # Differences from mean: 0.0, +0.10, +0.05, -0.15, 0.0
    # Sum sq: 0.0 + 0.01 + 0.0025 + 0.0225 + 0.0 = 0.035
    # Var(D) = 0.035 / 4 = 0.00875
    # s_D = sqrt(0.00875) ~ 0.0935414
    # SE = s_D / sqrt(5) ~ 0.041833
    # t = 0.10 / 0.041833 ~ 2.390457
    # Cohen's dz = 0.10 / 0.0935414 ~ 1.069045
    scores_b_var = {
        "q0": 0.70,  # diff +0.10
        "q1": 0.90,  # diff +0.20
        "q2": 0.95,  # diff +0.15
        "q3": 0.60,  # diff -0.05
        "q4": 0.85,  # diff +0.10
    }
    res_var = StatisticalComparator.compare_single_metric(
        run_a,
        run_b,
        metric_name="recall@5",
        scores_a_by_query=scores_a,
        scores_b_by_query=scores_b_var,
    )
    assert abs(res_var.mean_difference - 0.10) < 1e-6
    assert abs(res_var.std_difference - 0.0935414) < 1e-5
    assert res_var.t_statistic is not None and abs(res_var.t_statistic - 2.390457) < 1e-4
    assert res_var.cohens_dz is not None and abs(res_var.cohens_dz - 1.069045) < 1e-4
    assert res_var.t_df == 4
    assert res_var.raw_p_value is not None and res_var.raw_p_value < 0.10
    assert len(res_var.raw_paired_observations) == 5


def test_latency_warmup_exclusion() -> None:
    """Verify that warm-up queries are excluded strictly from latency comparisons."""
    exp = MagicMock(spec=Experiment, dataset_version_id="ver-warmup")
    run_a = MagicMock(spec=ExperimentRun, id="run-a", experiment=exp)
    run_b = MagicMock(spec=ExperimentRun, id="run-b", experiment=exp)

    # 5 queries: first 2 are warm-up spikes (e.g. 500ms, 400ms) followed by steady state (50ms)
    metrics_a = {
        f"q{i}": {"total_latency_ms": 500.0 if i < 2 else 50.0, "recall@5": 0.8} for i in range(5)
    }
    metrics_b = {
        f"q{i}": {"total_latency_ms": 600.0 if i < 2 else 45.0, "recall@5": 0.9} for i in range(5)
    }

    # Compare with warmup_queries = 2
    report = StatisticalComparator.compare_runs(
        run_a=run_a,
        run_b=run_b,
        metrics_by_query_a=metrics_a,
        metrics_by_query_b=metrics_b,
        warmup_queries=2,
    )
    assert report.warmup_queries_excluded == 2

    # In latency metric, only queries q2, q3, q4 (steady state) are included
    lat_res = report.metrics["total_latency_ms"]
    assert lat_res.alignment.n_paired == 3
    assert lat_res.mean_a == 50.0
    assert lat_res.mean_b == 45.0
    assert abs(lat_res.mean_difference - (-5.0)) < 1e-6

    # Non-latency metrics (recall@5) retain all 5 queries
    recall_res = report.metrics["recall@5"]
    assert recall_res.alignment.n_paired == 5


def test_scipy_ttest_rel_oracle() -> None:
    """Cross-validate paired t-test statistics and two-tailed p-values against SciPy oracle."""
    from scipy import stats

    exp = MagicMock(spec=Experiment, dataset_version_id="ver-scipy")
    run_a = MagicMock(spec=ExperimentRun, id="run-a", experiment=exp)
    run_b = MagicMock(spec=ExperimentRun, id="run-b", experiment=exp)

    # Test cases: varying sample sizes and distributions
    test_cases = [
        # N=5 small sample
        ([0.45, 0.72, 0.61, 0.88, 0.33], [0.55, 0.90, 0.65, 0.78, 0.52]),
        # N=12 medium sample with bidirectional differences
        (
            [0.10, 0.25, 0.35, 0.50, 0.65, 0.70, 0.80, 0.85, 0.90, 0.40, 0.55, 0.75],
            [0.15, 0.20, 0.40, 0.60, 0.60, 0.80, 0.75, 0.95, 0.92, 0.50, 0.60, 0.85],
        ),
        # N=30 large sample
        (
            [0.3 + 0.02 * i for i in range(30)],
            [0.35 + 0.018 * i + (0.05 if i % 2 == 0 else -0.02) for i in range(30)],
        ),
    ]

    for a_vals, b_vals in test_cases:
        n = len(a_vals)
        scores_a = {f"q{i:03d}": a_vals[i] for i in range(n)}
        scores_b = {f"q{i:03d}": b_vals[i] for i in range(n)}

        res = StatisticalComparator.compare_single_metric(
            run_a=run_a,
            run_b=run_b,
            metric_name="recall@5",
            scores_a_by_query=scores_a,
            scores_b_by_query=scores_b,
        )

        scipy_res = stats.ttest_rel(b_vals, a_vals)
        assert res.t_statistic is not None
        assert abs(res.t_statistic - scipy_res.statistic) < 1e-6
        assert res.raw_p_value is not None
        assert abs(res.raw_p_value - scipy_res.pvalue) < 1e-6


def test_scipy_wilcoxon_oracle() -> None:
    """Cross-validate Wilcoxon signed-rank test against SciPy oracle."""
    from scipy import stats

    # 1. Exact regime (N=10 <= 20) with no ties
    diffs_exact = [0.12, -0.05, 0.22, 0.31, -0.15, 0.08, 0.19, -0.27, 0.44, 0.03]
    stat_exact, p_exact, nz_exact, z_exact, method_exact = wilcoxon_signed_rank_test(diffs_exact)
    scipy_exact = stats.wilcoxon(diffs_exact, method="exact")
    assert method_exact == "exact"
    assert stat_exact == scipy_exact.statistic
    assert abs(p_exact - scipy_exact.pvalue) < 1e-6

    # 2. Asymptotic regime (N=25 > 20) with continuity correction
    diffs_asymp = [float(i) * 0.05 * (-1 if i % 3 == 0 else 1) for i in range(1, 26)]
    stat_asymp, p_asymp, nz_asymp, z_asymp, method_asymp = wilcoxon_signed_rank_test(diffs_asymp)
    scipy_asymp = stats.wilcoxon(diffs_asymp, method="approx", correction=True)
    assert method_asymp == "asymptotic"
    assert stat_asymp == scipy_asymp.statistic
    assert abs(p_asymp - scipy_asymp.pvalue) < 1e-6


def test_scipy_student_t_critical_value_oracle() -> None:
    """Cross-validate Student's t critical values against SciPy t.ppf oracle."""
    from scipy import stats

    for df in [1, 2, 5, 10, 20, 30, 50, 100, 500]:
        ours = student_t_critical_value(df, alpha=0.05)
        scipy_val = float(stats.t.ppf(0.975, df))
        assert abs(ours - scipy_val) < 1e-5


def test_statistical_edge_cases_single_observation() -> None:
    """N=1 query: cannot compute degrees of freedom, variance, or paired t-test."""
    exp = MagicMock(spec=Experiment, dataset_version_id="ver-edge")
    run_a = MagicMock(spec=ExperimentRun, id="run-a", experiment=exp)
    run_b = MagicMock(spec=ExperimentRun, id="run-b", experiment=exp)

    res = StatisticalComparator.compare_single_metric(
        run_a=run_a,
        run_b=run_b,
        metric_name="recall@5",
        scores_a_by_query={"q1": 0.50},
        scores_b_by_query={"q1": 0.70},
    )
    assert abs(res.mean_difference - 0.20) < 1e-7
    assert res.t_statistic is None
    assert res.t_df is None
    assert res.raw_p_value is None
    assert res.cohens_dz is None
    assert res.ci95_lower is None
    assert res.ci95_upper is None


def test_statistical_edge_cases_identical_runs() -> None:
    """Degenerate identical runs: all differences are 0.0."""
    exp = MagicMock(spec=Experiment, dataset_version_id="ver-edge")
    run_a = MagicMock(spec=ExperimentRun, id="run-a", experiment=exp)
    run_b = MagicMock(spec=ExperimentRun, id="run-b", experiment=exp)

    scores = {f"q{i}": 0.75 for i in range(10)}
    res = StatisticalComparator.compare_single_metric(
        run_a=run_a,
        run_b=run_b,
        metric_name="recall@5",
        scores_a_by_query=scores,
        scores_b_by_query=scores,
    )
    assert res.mean_difference == 0.0
    assert res.std_difference == 0.0
    assert res.t_statistic == 0.0
    assert res.raw_p_value == 1.0
    assert res.cohens_dz == 0.0
    assert res.ci95_lower == 0.0
    assert res.ci95_upper == 0.0
    assert res.wilcoxon_statistic == 0.0
    assert res.wilcoxon_pvalue == 1.0
    assert res.wilcoxon_method == "degenerate_identical"


def test_statistical_edge_cases_constant_delta() -> None:
    """Zero difference variance with non-zero delta: delta = +0.15 across all queries."""
    exp = MagicMock(spec=Experiment, dataset_version_id="ver-edge")
    run_a = MagicMock(spec=ExperimentRun, id="run-a", experiment=exp)
    run_b = MagicMock(spec=ExperimentRun, id="run-b", experiment=exp)

    scores_a = {f"q{i}": 0.50 for i in range(8)}
    scores_b = {f"q{i}": 0.65 for i in range(8)}
    res = StatisticalComparator.compare_single_metric(
        run_a=run_a,
        run_b=run_b,
        metric_name="recall@5",
        scores_a_by_query=scores_a,
        scores_b_by_query=scores_b,
    )
    assert abs(res.mean_difference - 0.15) < 1e-7
    assert res.std_difference == 0.0
    assert res.t_statistic == float("inf")
    assert res.raw_p_value == 0.0
    assert res.ci95_lower is not None and abs(res.ci95_lower - 0.15) < 1e-7
    assert res.ci95_upper is not None and abs(res.ci95_upper - 0.15) < 1e-7
