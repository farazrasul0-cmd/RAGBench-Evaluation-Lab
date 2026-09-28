"""Schemas for empirical statistical analysis, hypothesis testing, and query alignment."""

from typing import Any

from pydantic import BaseModel, Field


class ComparisonAlignment(BaseModel):
    """Encapsulates alignment integrity checks between two compared runs."""

    dataset_version_match: bool = Field(
        description="Whether runs evaluate the exact same dataset version"
    )
    dataset_version_id_a: str = Field(
        default="", description="Dataset version ID evaluated by Run A"
    )
    dataset_version_id_b: str = Field(
        default="", description="Dataset version ID evaluated by Run B"
    )
    dataset_version_hash_a: str | None = Field(
        default=None, description="Content hash of dataset version A"
    )
    dataset_version_hash_b: str | None = Field(
        default=None, description="Content hash of dataset version B"
    )

    benchmark_match: bool = Field(
        default=True, description="Whether runs evaluate the identical benchmark specification"
    )
    benchmark_hash_a: str | None = Field(default=None, description="Benchmark hash for Run A")
    benchmark_hash_b: str | None = Field(default=None, description="Benchmark hash for Run B")

    query_set_match: bool = Field(
        description="Whether runs evaluate the identical set of benchmark queries"
    )
    query_set_hash_a: str = Field(description="Deterministic hash of Run A query ID set")
    query_set_hash_b: str = Field(description="Deterministic hash of Run B query ID set")

    metric_name: str = Field(description="Standardized metric name being compared")
    k_a: int | None = Field(default=None, description="Cutoff K parameter for Run A")
    k_b: int | None = Field(default=None, description="Cutoff K parameter for Run B")
    metric_match: bool = Field(
        default=True, description="Whether compared metric definitions are identical"
    )
    k_match: bool = Field(default=True, description="Whether metric K parameter is identical")

    n_total_a: int = Field(description="Total queries executed in Run A")
    n_total_b: int = Field(description="Total queries executed in Run B")
    n_paired: int = Field(description="Number of strictly paired query observations")
    missing_in_a: list[str] = Field(
        default_factory=list, description="Query IDs present in B but missing in A"
    )
    missing_in_b: list[str] = Field(
        default_factory=list, description="Query IDs present in A but missing in B"
    )
    paired_query_ids: list[str] = Field(
        default_factory=list, description="Sorted aligned query IDs"
    )


class PairedObservation(BaseModel):
    """An individual paired query score observation."""

    query_id: str
    score_a: float
    score_b: float
    difference: float  # score_b - score_a


class StatisticalComparisonResult(BaseModel):
    """Comprehensive statistical hypothesis testing result for a single metric between two runs."""

    comparison_id: str
    run_a_id: str
    run_b_id: str
    dataset_version_id: str
    metric_name: str
    k: int | None = None

    alignment: ComparisonAlignment

    mean_a: float
    mean_b: float
    mean_difference: float  # d_bar = mean(score_b - score_a)
    std_difference: float  # s_D

    # Paired Student's t-test
    t_statistic: float | None = None
    t_df: int | None = None
    raw_p_value: float | None = None

    # Non-parametric Wilcoxon Signed-Rank Test
    wilcoxon_statistic: float | None = None
    wilcoxon_pvalue: float | None = None
    wilcoxon_n_nonzero: int = 0
    wilcoxon_zero_count: int = 0
    wilcoxon_method: str = "exact"  # "exact" | "asymptotic" | "degenerate_identical"

    # Effect size and bounds
    cohens_dz: float | None = None  # d_bar / s_D
    ci95_lower: float | None = None  # 95% CI lower bound for mean paired difference
    ci95_upper: float | None = None  # 95% CI upper bound for mean paired difference

    # Multiple-comparison correction
    adjustment_method: str = "none"  # "none" | "holm" | "bh_fdr"
    adjusted_p_value: float | None = None
    significance_level: str = "ns"  # "ns", "*", "**", "***"

    # Auditable raw observations
    raw_paired_observations: list[PairedObservation] = Field(default_factory=list)


class MetricSignificanceReport(BaseModel):
    """Aggregate multi-metric hypothesis testing report across an entire experiment comparison."""

    comparison_id: str
    run_a_id: str
    run_b_id: str
    dataset_version_id: str
    adjustment_method: str
    metrics: dict[str, StatisticalComparisonResult] = Field(default_factory=dict)
    warmup_queries_excluded: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
