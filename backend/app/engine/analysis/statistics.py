"""Statistical hypothesis testing and empirical comparative analysis engine."""

import contextlib
import math

from app.engine.analysis.alignment import AlignmentValidator
from app.models.entities import ExperimentRun
from app.schemas.analysis import (
    ComparisonAlignment,
    MetricSignificanceReport,
    PairedObservation,
    StatisticalComparisonResult,
)


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Compute regularized incomplete beta function I_x(a, b) using continued fractions."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Symmetry transformation for rapid convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a)

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a

    tiny = 1e-30
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    max_iter = 200
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        num = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c

        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        del_h = d * c
        h *= del_h

        if abs(del_h - 1.0) < 1e-14:
            break

    return front * h


def student_t_two_tailed_p(t: float, df: int) -> float:
    """Compute two-tailed p-value for Student's t distribution with df degrees of freedom."""
    if df <= 0:
        return float("nan")
    t_sq = t * t
    x = df / (df + t_sq)
    return regularized_incomplete_beta(x, 0.5 * df, 0.5)


def student_t_critical_value(df: int, alpha: float = 0.05) -> float:
    """Compute two-tailed critical value t_{1 - alpha/2, df} using bisection."""
    if df <= 0:
        return float("nan")
    low = 0.0
    high = 100.0
    target = alpha

    for _ in range(60):
        mid = (low + high) / 2.0
        p = student_t_two_tailed_p(mid, df)
        if abs(p - target) < 1e-12:
            return mid
        if p > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def wilcoxon_signed_rank_test(
    diffs: list[float],
) -> tuple[float | None, float | None, int, int, str]:
    """Two-sided Wilcoxon signed-rank test.

    Returns:
        (statistic, p_value, n_nonzero, n_zero, method)
    """
    non_zero = [d for d in diffs if abs(d) > 1e-12]
    n_zero = len(diffs) - len(non_zero)
    n_nonzero = len(non_zero)

    if n_nonzero == 0:
        return 0.0, 1.0, 0, n_zero, "degenerate_identical"

    abs_diffs = [abs(d) for d in non_zero]
    sorted_indices = sorted(range(n_nonzero), key=lambda i: abs_diffs[i])

    ranks = [0.0] * n_nonzero
    i = 0
    tie_groups = []
    while i < n_nonzero:
        j = i
        while (
            j < n_nonzero
            and abs(abs_diffs[sorted_indices[j]] - abs_diffs[sorted_indices[i]]) < 1e-12
        ):
            j += 1
        tie_size = j - i
        if tie_size > 1:
            tie_groups.append(tie_size)
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorted_indices[k]] = avg_rank
        i = j

    w_plus = sum(ranks[i] for i in range(n_nonzero) if non_zero[i] > 0)
    w_minus = sum(ranks[i] for i in range(n_nonzero) if non_zero[i] < 0)
    t_stat = min(w_plus, w_minus)

    if n_nonzero <= 20 and not tie_groups:
        max_w = int(n_nonzero * (n_nonzero + 1) // 2)
        dp = [0.0] * (max_w + 1)
        dp[0] = 1.0
        for r in range(1, n_nonzero + 1):
            for k in range(max_w, r - 1, -1):
                dp[k] = dp[k] + dp[k - r]
        total_perms = 2.0**n_nonzero
        cdf = [c / total_perms for c in dp]
        cutoff = int(math.floor(t_stat))
        p_val = min(1.0, 2.0 * sum(cdf[: cutoff + 1]))
        method = "exact"
    else:
        mean_w = n_nonzero * (n_nonzero + 1) / 4.0
        tie_adj = sum(t * (t * t - 1) for t in tie_groups) / 48.0
        var_w = (n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0) - tie_adj
        if var_w <= 0:
            return t_stat, 1.0, n_nonzero, n_zero, "asymptotic"
        std_w = math.sqrt(var_w)
        z = (abs(t_stat - mean_w) - 0.5) / std_w
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
        p_val = max(0.0, min(1.0, p_val))
        method = "asymptotic"

    return t_stat, p_val, n_nonzero, n_zero, method


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Adjust p-values using the Holm-Bonferroni step-down method (controls FWER)."""
    m = len(p_values)
    if m == 0:
        return []
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    cum_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank
        raw_adj = multiplier * p_val
        cum_max = max(cum_max, raw_adj)
        adjusted[orig_idx] = min(1.0, cum_max)

    return adjusted


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Adjust p-values using the Benjamini-Hochberg step-up procedure (controls FDR)."""
    m = len(p_values)
    if m == 0:
        return []
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    cum_min = 1.0
    for rank in range(m - 1, -1, -1):
        orig_idx, p_val = indexed_p[rank]
        k = rank + 1
        raw_adj = (m / k) * p_val
        cum_min = min(cum_min, raw_adj)
        adjusted[orig_idx] = min(1.0, cum_min)

    return adjusted


def get_significance_stars(p_value: float | None) -> str:
    """Format standard scientific significance asterisks from p-value."""
    if p_value is None or math.isnan(p_value):
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


class StatisticalComparator:
    """Performs empirical statistical comparisons between two benchmark runs."""

    @classmethod
    def compare_single_metric(
        cls,
        run_a: ExperimentRun,
        run_b: ExperimentRun,
        metric_name: str,
        scores_a_by_query: dict[str, float],
        scores_b_by_query: dict[str, float],
        k: int | None = None,
        k_a: int | None = None,
        k_b: int | None = None,
        allow_partial_query_overlap: bool = False,
    ) -> StatisticalComparisonResult:
        """Perform paired t-test, Wilcoxon, Cohen's dz, and CI on a single metric."""
        eff_k_a = k_a if k_a is not None else k
        eff_k_b = k_b if k_b is not None else k
        alignment: ComparisonAlignment = AlignmentValidator.validate_and_align(
            run_a=run_a,
            run_b=run_b,
            scores_a_by_query=scores_a_by_query,
            scores_b_by_query=scores_b_by_query,
            metric_name=metric_name,
            k=k,
            k_a=eff_k_a,
            k_b=eff_k_b,
            allow_partial_query_overlap=allow_partial_query_overlap,
        )

        paired_obs: list[PairedObservation] = []
        for q_id in alignment.paired_query_ids:
            sa = scores_a_by_query[q_id]
            sb = scores_b_by_query[q_id]
            paired_obs.append(
                PairedObservation(
                    query_id=q_id,
                    score_a=sa,
                    score_b=sb,
                    difference=sb - sa,
                )
            )

        n = len(paired_obs)
        if n == 0:
            raise ValueError(f"Zero paired observations for metric '{metric_name}'.")

        scores_a = [o.score_a for o in paired_obs]
        scores_b = [o.score_b for o in paired_obs]
        diffs = [o.difference for o in paired_obs]

        mean_a = sum(scores_a) / n
        mean_b = sum(scores_b) / n
        mean_diff = sum(diffs) / n

        # Paired difference standard deviation
        if n > 1:
            var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
            std_diff = math.sqrt(var_diff)
        else:
            std_diff = 0.0

        # Paired Student's t-test
        t_stat: float | None = None
        t_df: int | None = None
        raw_p: float | None = None
        cohens_dz: float | None = None
        ci_lower: float | None = None
        ci_upper: float | None = None

        if n >= 2:
            t_df = n - 1
            if std_diff > 1e-12:
                se = std_diff / math.sqrt(n)
                t_stat = mean_diff / se
                raw_p = student_t_two_tailed_p(t_stat, t_df)
                cohens_dz = mean_diff / std_diff

                # 95% Confidence Interval for mean paired difference
                t_crit = student_t_critical_value(t_df, alpha=0.05)
                margin = t_crit * se
                ci_lower = mean_diff - margin
                ci_upper = mean_diff + margin
            else:
                # Zero variance in differences
                if abs(mean_diff) < 1e-12:
                    # Identical runs
                    t_stat = 0.0
                    raw_p = 1.0
                    cohens_dz = 0.0
                    ci_lower = 0.0
                    ci_upper = 0.0
                else:
                    # Constant non-zero delta
                    t_stat = float("inf") if mean_diff > 0 else float("-inf")
                    raw_p = 0.0
                    cohens_dz = float("inf") if mean_diff > 0 else float("-inf")
                    ci_lower = mean_diff
                    ci_upper = mean_diff

        # Wilcoxon Signed-Rank Test
        w_stat, w_p, w_nz, w_z, w_method = wilcoxon_signed_rank_test(diffs)

        ds_ver_id = run_a.experiment.dataset_version_id if run_a.experiment else "unknown"

        return StatisticalComparisonResult(
            comparison_id=f"comp-{run_a.id[:8]}-{run_b.id[:8]}-{metric_name}",
            run_a_id=run_a.id,
            run_b_id=run_b.id,
            dataset_version_id=ds_ver_id,
            metric_name=metric_name,
            k=k,
            alignment=alignment,
            mean_a=mean_a,
            mean_b=mean_b,
            mean_difference=mean_diff,
            std_difference=std_diff,
            t_statistic=t_stat,
            t_df=t_df,
            raw_p_value=raw_p,
            wilcoxon_statistic=w_stat,
            wilcoxon_pvalue=w_p,
            wilcoxon_n_nonzero=w_nz,
            wilcoxon_zero_count=w_z,
            wilcoxon_method=w_method,
            cohens_dz=cohens_dz,
            ci95_lower=ci_lower,
            ci95_upper=ci_upper,
            adjustment_method="none",
            adjusted_p_value=raw_p,
            significance_level=get_significance_stars(raw_p),
            raw_paired_observations=paired_obs,
        )

    @classmethod
    def compare_runs(
        cls,
        run_a: ExperimentRun,
        run_b: ExperimentRun,
        metrics_by_query_a: dict[str, dict[str, float]],
        metrics_by_query_b: dict[str, dict[str, float]],
        correction_method: str = "holm",
        warmup_queries: int = 0,
        allow_partial_query_overlap: bool = False,
    ) -> MetricSignificanceReport:
        """Compare all mutual metrics between two runs with multiple-comparison correction."""
        # Detect all shared metric names
        sample_q_a = next(iter(metrics_by_query_a.values()), {})
        sample_q_b = next(iter(metrics_by_query_b.values()), {})
        all_metrics = sorted(set(sample_q_a.keys()) & set(sample_q_b.keys()))

        results: dict[str, StatisticalComparisonResult] = {}
        raw_p_values: list[float] = []
        metric_order: list[str] = []

        for m_name in all_metrics:
            # Extract scores map {query_id: score}
            # Latency special handling: if warm-up is requested
            scores_a: dict[str, float] = {}
            scores_b: dict[str, float] = {}

            # Sort query IDs to ensure deterministic warm-up exclusion
            ordered_qids = sorted(set(metrics_by_query_a.keys()) & set(metrics_by_query_b.keys()))
            if warmup_queries > 0 and "latency" in m_name.lower():
                usable_qids = set(ordered_qids[warmup_queries:])
            else:
                usable_qids = set(ordered_qids)

            for qid in usable_qids:
                if m_name in metrics_by_query_a.get(qid, {}):
                    scores_a[qid] = metrics_by_query_a[qid][m_name]
                if m_name in metrics_by_query_b.get(qid, {}):
                    scores_b[qid] = metrics_by_query_b[qid][m_name]

            if not scores_a or not scores_b:
                continue

            # Parse K if present in metric_name (e.g. recall@5)
            k_val: int | None = None
            if "@" in m_name:
                with contextlib.suppress(Exception):
                    k_val = int(m_name.split("@")[1])

            res = cls.compare_single_metric(
                run_a=run_a,
                run_b=run_b,
                metric_name=m_name,
                scores_a_by_query=scores_a,
                scores_b_by_query=scores_b,
                k=k_val,
                k_a=k_val,
                k_b=k_val,
                allow_partial_query_overlap=allow_partial_query_overlap,
            )
            results[m_name] = res
            metric_order.append(m_name)
            raw_p_values.append(res.raw_p_value if res.raw_p_value is not None else 1.0)

        # Apply multiple-comparison correction across all evaluated metrics
        if correction_method == "holm":
            adj_p_values = holm_bonferroni(raw_p_values)
        elif correction_method == "bh_fdr":
            adj_p_values = benjamini_hochberg(raw_p_values)
        else:
            adj_p_values = list(raw_p_values)

        for m_name, adj_p in zip(metric_order, adj_p_values, strict=False):
            res = results[m_name]
            res.adjustment_method = correction_method
            res.adjusted_p_value = adj_p
            res.significance_level = get_significance_stars(adj_p)

        ds_ver_id = run_a.experiment.dataset_version_id if run_a.experiment else "unknown"

        return MetricSignificanceReport(
            comparison_id=f"comp-{run_a.id[:8]}-{run_b.id[:8]}",
            run_a_id=run_a.id,
            run_b_id=run_b.id,
            dataset_version_id=ds_ver_id,
            adjustment_method=correction_method,
            metrics=results,
            warmup_queries_excluded=warmup_queries,
        )
