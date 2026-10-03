"""Unit tests for LanguageTransferAnalyzer (Phase G Claim A and Claim B).

Pre-registered hypothesis family (Holm m=4):
  H1: Dense EN-BN penalty > 0
  H2: Dense BN-EN penalty > 0
  H3: Hybrid attenuates EN-BN penalty
  H4: Hybrid attenuates BN-EN penalty
"""

from app.engine.analysis.transfer import LanguageTransferAnalyzer

_UNIT_IDS = [f"unit_{i:02d}" for i in range(1, 11)]  # 10 matched units for unit tests


def _make_scores(ids: list[str], base: float, noise: float = 0.0) -> dict[str, float]:
    """Create score map with slight per-unit variation."""
    return {uid: min(1.0, base + (hash(uid) % 7) * noise) for uid in ids}


class TestPenaltyFormula:
    """Claim A: penalty_q = mono_score_q - cross_score_q."""

    def test_penalty_computed_correctly(self) -> None:
        analyzer = LanguageTransferAnalyzer()
        ids = ["u01", "u02", "u03"]
        mono = {"u01": 0.90, "u02": 0.85, "u03": 0.80}
        cross = {"u01": 0.70, "u02": 0.65, "u03": 0.60}

        report = analyzer.compute_penalty(
            mono_scores=mono,
            cross_scores=cross,
            information_unit_ids=ids,
            modality_mono="EN-EN",
            modality_cross="EN-BN",
            retrieval_strategy="dense",
            metric_name="recall@5",
        )

        assert report.n_pairs == 3
        expected_diffs = [0.20, 0.20, 0.20]
        diff_pairs = zip(report.per_unit_penalty, expected_diffs, strict=True)
        assert all(abs(a - b) < 1e-9 for a, b in diff_pairs)
        assert abs(report.mean_penalty - 0.20) < 1e-9

    def test_zero_penalty_for_equal_performance(self) -> None:
        analyzer = LanguageTransferAnalyzer()
        ids = ["u01", "u02"]
        scores = {"u01": 0.80, "u02": 0.75}

        report = analyzer.compute_penalty(
            mono_scores=scores,
            cross_scores=scores,
            information_unit_ids=ids,
            modality_mono="BN-BN",
            modality_cross="BN-EN",
            retrieval_strategy="bm25",
            metric_name="ndcg@5",
        )

        assert abs(report.mean_penalty) < 1e-9
        assert report.n_pairs == 2

    def test_negative_penalty_possible(self) -> None:
        """Cross-lingual can outperform monolingual — penalty may be negative."""
        analyzer = LanguageTransferAnalyzer()
        ids = ["u01"]
        mono = {"u01": 0.60}
        cross = {"u01": 0.80}

        report = analyzer.compute_penalty(
            mono_scores=mono,
            cross_scores=cross,
            information_unit_ids=ids,
            modality_mono="EN-EN",
            modality_cross="EN-BN",
            retrieval_strategy="hybrid",
            metric_name="mrr@5",
        )

        assert report.mean_penalty < 0.0

    def test_empty_units_returns_zero_report(self) -> None:
        analyzer = LanguageTransferAnalyzer()
        report = analyzer.compute_penalty(
            mono_scores={},
            cross_scores={},
            information_unit_ids=[],
            modality_mono="EN-EN",
            modality_cross="EN-BN",
            retrieval_strategy="dense",
            metric_name="recall@5",
        )
        assert report.n_pairs == 0
        assert report.mean_penalty == 0.0

    def test_report_includes_all_fields(self) -> None:
        analyzer = LanguageTransferAnalyzer()
        ids = ["u01", "u02", "u03", "u04"]
        mono = dict.fromkeys(ids, 0.85)
        cross = dict.fromkeys(ids, 0.65)

        report = analyzer.compute_penalty(
            mono_scores=mono,
            cross_scores=cross,
            information_unit_ids=ids,
            modality_mono="EN-EN",
            modality_cross="EN-BN",
            retrieval_strategy="dense",
            metric_name="recall@5",
        )

        assert report.modality_mono == "EN-EN"
        assert report.modality_cross == "EN-BN"
        assert report.retrieval_strategy == "dense"
        assert report.metric_name == "recall@5"
        assert report.n_pairs == 4
        assert report.ci_95_lower <= report.mean_penalty <= report.ci_95_upper


class TestAttenuationFormula:
    """Claim B: attenuation = Penalty_Dense - Penalty_Hybrid."""

    def test_attenuation_positive_when_hybrid_reduces_penalty(self) -> None:
        analyzer = LanguageTransferAnalyzer()
        ids = ["u01", "u02", "u03"]

        # Dense: large penalty (mono-cross = 0.20)
        dense_mono = dict.fromkeys(ids, 0.9)
        dense_cross = dict.fromkeys(ids, 0.7)

        # Hybrid: smaller penalty (mono-cross = 0.10)
        hybrid_mono = dict.fromkeys(ids, 0.9)
        hybrid_cross = dict.fromkeys(ids, 0.8)

        report_dense = analyzer.compute_penalty(
            mono_scores=dense_mono,
            cross_scores=dense_cross,
            information_unit_ids=ids,
            modality_mono="EN-EN",
            modality_cross="EN-BN",
            retrieval_strategy="dense",
            metric_name="recall@5",
        )
        report_hybrid = analyzer.compute_penalty(
            mono_scores=hybrid_mono,
            cross_scores=hybrid_cross,
            information_unit_ids=ids,
            modality_mono="EN-EN",
            modality_cross="EN-BN",
            retrieval_strategy="hybrid",
            metric_name="recall@5",
        )

        attenuation = analyzer.compute_attenuation(report_dense, report_hybrid)
        # Dense penalty (0.20) - Hybrid penalty (0.10) = 0.10
        assert abs(attenuation - 0.10) < 1e-6

    def test_zero_attenuation_when_penalties_equal(self) -> None:
        analyzer = LanguageTransferAnalyzer()
        ids = ["u01"]
        mono = {"u01": 0.80}
        cross = {"u01": 0.70}

        r1 = analyzer.compute_penalty(
            mono_scores=mono,
            cross_scores=cross,
            information_unit_ids=ids,
            modality_mono="EN-EN",
            modality_cross="EN-BN",
            retrieval_strategy="dense",
            metric_name="recall@5",
        )
        r2 = analyzer.compute_penalty(
            mono_scores=mono,
            cross_scores=cross,
            information_unit_ids=ids,
            modality_mono="EN-EN",
            modality_cross="EN-BN",
            retrieval_strategy="hybrid",
            metric_name="recall@5",
        )

        attenuation = analyzer.compute_attenuation(r1, r2)
        assert abs(attenuation) < 1e-9

    def test_attach_attenuation_populates_field(self) -> None:
        analyzer = LanguageTransferAnalyzer()
        ids = ["u01", "u02"]
        dense_mono = {"u01": 0.90, "u02": 0.85}
        dense_cross = {"u01": 0.70, "u02": 0.65}
        hybrid_mono = {"u01": 0.90, "u02": 0.85}
        hybrid_cross = {"u01": 0.80, "u02": 0.78}

        r_dense = analyzer.compute_penalty(
            mono_scores=dense_mono,
            cross_scores=dense_cross,
            information_unit_ids=ids,
            modality_mono="EN-EN",
            modality_cross="EN-BN",
            retrieval_strategy="dense",
            metric_name="recall@5",
        )
        r_hybrid = analyzer.compute_penalty(
            mono_scores=hybrid_mono,
            cross_scores=hybrid_cross,
            information_unit_ids=ids,
            modality_mono="EN-EN",
            modality_cross="EN-BN",
            retrieval_strategy="hybrid",
            metric_name="recall@5",
        )

        r_with_attenuation = analyzer.attach_attenuation(r_dense, r_hybrid)
        assert r_with_attenuation.attenuation is not None
        assert r_with_attenuation.attenuation > 0


class TestMatchedPairOrderingInvariant:
    """Shuffling information_unit_ids must yield identical mean_penalty."""

    def test_permutation_invariant(self) -> None:
        import random

        analyzer = LanguageTransferAnalyzer()
        ids = [f"unit_{i}" for i in range(8)]
        mono = {uid: 0.80 + i * 0.01 for i, uid in enumerate(ids)}
        cross = {uid: 0.60 + i * 0.01 for i, uid in enumerate(ids)}

        report_ordered = analyzer.compute_penalty(
            mono_scores=mono,
            cross_scores=cross,
            information_unit_ids=ids,
            modality_mono="BN-BN",
            modality_cross="BN-EN",
            retrieval_strategy="hybrid",
            metric_name="ndcg@5",
        )

        shuffled = ids.copy()
        random.seed(42)
        random.shuffle(shuffled)

        report_shuffled = analyzer.compute_penalty(
            mono_scores=mono,
            cross_scores=cross,
            information_unit_ids=shuffled,
            modality_mono="BN-BN",
            modality_cross="BN-EN",
            retrieval_strategy="hybrid",
            metric_name="ndcg@5",
        )

        assert abs(report_ordered.mean_penalty - report_shuffled.mean_penalty) < 1e-12
        assert report_ordered.n_pairs == report_shuffled.n_pairs


class TestDelegatesToStatisticsEngine:
    """Transfer analyzer reuses F7/F8 statistical functions (no new math)."""

    def test_uses_wilcoxon_signed_rank(self) -> None:
        from unittest.mock import patch

        from app.engine.analysis import transfer as transfer_module

        analyzer = LanguageTransferAnalyzer()
        ids = ["u01", "u02", "u03", "u04", "u05"]
        mono = dict.fromkeys(ids, 0.9)
        cross = dict.fromkeys(ids, 0.7)

        target_fn = transfer_module.wilcoxon_signed_rank_test
        with patch.object(transfer_module, "wilcoxon_signed_rank_test", wraps=target_fn) as mock_w:
            analyzer.compute_penalty(
                mono_scores=mono,
                cross_scores=cross,
                information_unit_ids=ids,
                modality_mono="EN-EN",
                modality_cross="EN-BN",
                retrieval_strategy="dense",
                metric_name="recall@5",
            )
            assert mock_w.called

    def test_uses_student_t_p_function(self) -> None:
        from unittest.mock import patch

        from app.engine.analysis import transfer as transfer_module

        analyzer = LanguageTransferAnalyzer()
        ids = ["u01", "u02", "u03", "u04"]
        mono = {"u01": 0.90, "u02": 0.85, "u03": 0.80, "u04": 0.75}
        cross = {"u01": 0.60, "u02": 0.50, "u03": 0.70, "u04": 0.65}

        target_fn = transfer_module.student_t_two_tailed_p
        p_target = "app.engine.analysis.transfer.student_t_two_tailed_p"
        with patch(p_target, wraps=target_fn) as mock_t:
            analyzer.compute_penalty(
                mono_scores=mono,
                cross_scores=cross,
                information_unit_ids=ids,
                modality_mono="BN-BN",
                modality_cross="BN-EN",
                retrieval_strategy="bm25",
                metric_name="mrr@5",
            )
            assert mock_t.called
