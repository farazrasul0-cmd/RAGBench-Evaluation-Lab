"""Language transfer penalty and hybrid attenuation analysis engine (Phase G).

Implements:
  Claim A — Transfer Penalty:
      penalty_q = mono_score_q - cross_score_q
      mean_penalty = mean over N matched information units

  Claim B — Hybrid Attenuation:
      attenuation = mean_penalty_dense - mean_penalty_hybrid

Pre-registered hypothesis family (H1-H4, Holm m=4):
  H1: Dense EN-BN penalty > 0
  H2: Dense BN-EN penalty > 0
  H3: Hybrid attenuates EN-BN penalty (Attenuation_EN_BN > 0)
  H4: Hybrid attenuates BN-EN penalty (Attenuation_BN_EN > 0)

Statistical computation DELEGATES entirely to the existing F7/F8
``StatisticalComparator`` (statistics.py).  No new statistical math is
introduced here.
"""

from __future__ import annotations

import math

from app.engine.analysis.statistics import (
    get_significance_stars,
    student_t_critical_value,
    student_t_two_tailed_p,
    wilcoxon_signed_rank_test,
)
from app.schemas.transfer import LanguageTransferReport

__all__ = [
    "LanguageTransferAnalyzer",
    "student_t_two_tailed_p",
    "wilcoxon_signed_rank_test",
]


class LanguageTransferAnalyzer:
    """Compute cross-lingual transfer penalties and hybrid attenuation.

    All statistical computation delegates to the existing F7/F8 functions in
    ``app.engine.analysis.statistics``.  This class only handles grouping of
    observations by ``information_unit_id`` and computing the penalty / attenuation
    formulae.

    N for statistical tests = number of matched information units (never raw
    query count), preventing pseudoreplication across modalities.
    """

    # Pre-registered hypothesis family size for Holm correction (H1-H4).
    HYPOTHESIS_FAMILY_SIZE: int = 4

    def compute_penalty(
        self,
        mono_scores: dict[str, float],
        cross_scores: dict[str, float],
        information_unit_ids: list[str],
        modality_mono: str,
        modality_cross: str,
        retrieval_strategy: str,
        metric_name: str,
        pre_adjusted_p: float | None = None,
    ) -> LanguageTransferReport:
        """Compute Claim A: transfer penalty for one metric and one strategy.

        Args:
            mono_scores: Map of ``information_unit_id -> monolingual score``.
            cross_scores: Map of ``information_unit_id -> cross-lingual score``.
            information_unit_ids: Ordered list of information unit IDs (the
                statistical unit; order determines permutation invariance).
            modality_mono: E.g. ``"EN-EN"`` or ``"BN-BN"``.
            modality_cross: E.g. ``"EN-BN"`` or ``"BN-EN"``.
            retrieval_strategy: E.g. ``"dense"``, ``"bm25"``, ``"hybrid"``.
            metric_name: E.g. ``"recall@5"``, ``"ndcg@5"``.
            pre_adjusted_p: Holm-adjusted p-value from family correction (if
                already computed externally); otherwise raw p is stored and
                ``p_adj`` is ``None``.

        Returns:
            :class:`LanguageTransferReport` with per-unit penalties, aggregate
            statistics, and test results.
        """
        # Sort IDs for permutation invariance (Amendment 1 invariant)
        sorted_ids = sorted(information_unit_ids)
        shared_ids = [uid for uid in sorted_ids if uid in mono_scores and uid in cross_scores]

        if not shared_ids:
            return LanguageTransferReport(
                modality_mono=modality_mono,
                modality_cross=modality_cross,
                retrieval_strategy=retrieval_strategy,
                metric_name=metric_name,
                n_pairs=0,
                information_unit_ids=[],
                per_unit_penalty=[],
                mean_penalty=0.0,
                std_penalty=0.0,
                ci_95_lower=0.0,
                ci_95_upper=0.0,
            )

        n = len(shared_ids)
        diffs = [mono_scores[uid] - cross_scores[uid] for uid in shared_ids]

        mean_d = sum(diffs) / n
        var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
        std_d = math.sqrt(var_d)

        # 95% CI using t-distribution
        if n > 1:
            t_crit = student_t_critical_value(df=n - 1, alpha=0.05)
            se = std_d / math.sqrt(n)
            ci_lo = mean_d - t_crit * se
            ci_hi = mean_d + t_crit * se
        else:
            ci_lo = ci_hi = mean_d

        # Cohen's d_z
        cohens_dz = mean_d / std_d if std_d > 1e-12 else None

        # Paired t-test (via existing engine)
        t_stat: float | None = None
        raw_p: float | None = None
        if n > 1 and std_d > 1e-12:
            se_t = std_d / math.sqrt(n)
            t_stat = mean_d / se_t
            raw_p = student_t_two_tailed_p(abs(t_stat), df=n - 1)

        # Wilcoxon signed-rank (via existing engine)
        w_stat, w_p, _, _, _ = wilcoxon_signed_rank_test(diffs)

        # Use Wilcoxon p as primary non-parametric; store raw_p for t also
        primary_p = w_p if w_p is not None else raw_p

        p_adj_final = pre_adjusted_p if pre_adjusted_p is not None else primary_p
        stars = get_significance_stars(p_adj_final)

        return LanguageTransferReport(
            modality_mono=modality_mono,
            modality_cross=modality_cross,
            retrieval_strategy=retrieval_strategy,
            metric_name=metric_name,
            n_pairs=n,
            information_unit_ids=shared_ids,
            per_unit_penalty=diffs,
            mean_penalty=round(mean_d, 6),
            std_penalty=round(std_d, 6),
            ci_95_lower=round(ci_lo, 6),
            ci_95_upper=round(ci_hi, 6),
            cohens_dz=round(cohens_dz, 6) if cohens_dz is not None else None,
            t_stat=round(t_stat, 6) if t_stat is not None else None,
            wilcoxon_w=w_stat,
            p_value=round(primary_p, 6) if primary_p is not None else None,
            p_adj=round(p_adj_final, 6) if p_adj_final is not None else None,
            significance_stars=stars,
        )

    def compute_attenuation(
        self,
        penalty_dense: LanguageTransferReport,
        penalty_hybrid: LanguageTransferReport,
    ) -> float:
        """Compute Claim B: attenuation = Penalty_Dense - Penalty_Hybrid.

        A positive value indicates that hybrid retrieval reduces the cross-lingual
        transfer penalty more than dense retrieval alone.

        Args:
            penalty_dense: Transfer penalty report for the dense retrieval strategy.
            penalty_hybrid: Transfer penalty report for the hybrid retrieval strategy.

        Returns:
            Absolute attenuation value (float).
        """
        return penalty_dense.mean_penalty - penalty_hybrid.mean_penalty

    def attach_attenuation(
        self,
        penalty_dense: LanguageTransferReport,
        penalty_hybrid: LanguageTransferReport,
    ) -> LanguageTransferReport:
        """Return *penalty_hybrid* with its ``attenuation`` field populated."""
        attenuation = self.compute_attenuation(penalty_dense, penalty_hybrid)
        return penalty_hybrid.model_copy(update={"attenuation": round(attenuation, 6)})
