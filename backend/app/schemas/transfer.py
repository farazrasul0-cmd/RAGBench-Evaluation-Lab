"""Language transfer penalty schema for Phase G multilingual analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LanguageTransferReport(BaseModel):
    """Result of a cross-lingual transfer penalty analysis for a single metric.

    Captures the matched-pair penalty (Claim A) and, optionally, the hybrid
    attenuation (Claim B) for one retrieval strategy comparison.

    Statistical N is always the number of matched information units (not raw
    query observations), preventing pseudoreplication.
    """

    modality_mono: str = Field(
        description="Monolingual retrieval modality (e.g. 'EN-EN' or 'BN-BN')"
    )
    modality_cross: str = Field(
        description="Cross-lingual retrieval modality (e.g. 'EN-BN' or 'BN-EN')"
    )
    retrieval_strategy: str = Field(
        description="Retrieval strategy (e.g. 'dense', 'bm25', 'hybrid')"
    )
    metric_name: str = Field(description="Metric being analysed (e.g. 'recall@5', 'ndcg@5')")
    n_pairs: int = Field(description="Number of matched information-unit pairs (statistical N)")
    information_unit_ids: list[str] = Field(
        default_factory=list,
        description="Ordered list of matched information_unit_id values",
    )
    per_unit_penalty: list[float] = Field(
        default_factory=list,
        description="Per-information-unit penalty: mono_score - cross_score",
    )
    mean_penalty: float = Field(
        description="Mean transfer penalty across all matched information units"
    )
    std_penalty: float = Field(
        default=0.0,
        description="Standard deviation of per-unit penalty",
    )
    ci_95_lower: float = Field(description="95% CI lower bound for mean paired penalty")
    ci_95_upper: float = Field(description="95% CI upper bound for mean paired penalty")
    cohens_dz: float | None = Field(
        default=None, description="Cohen's d_z effect size (mean_penalty / std_penalty)"
    )
    t_stat: float | None = Field(default=None, description="Paired Student's t statistic")
    wilcoxon_w: float | None = Field(default=None, description="Wilcoxon signed-rank statistic W")
    p_value: float | None = Field(default=None, description="Raw (unadjusted) p-value")
    p_adj: float | None = Field(
        default=None,
        description=("Holm-Bonferroni adjusted p-value over the pre-registered H1-H4 family (m=4)"),
    )
    significance_stars: str = Field(
        default="ns",
        description="Significance indicator: 'ns', '*', '**', or '***'",
    )
    # Claim B — hybrid attenuation (optional, populated when strategy-level comparison is done)
    attenuation: float | None = Field(
        default=None,
        description=(
            "Claim B: Penalty_Dense - Penalty_Hybrid.  Positive value indicates hybrid "
            "reduces the cross-lingual penalty more than dense retrieval."
        ),
    )
