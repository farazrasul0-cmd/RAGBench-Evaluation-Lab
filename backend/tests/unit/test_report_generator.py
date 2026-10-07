"""Comprehensive forensic unit and regression tests for AcademicReportGenerator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.services.report_generator import (
    AcademicReportGenerator,
    ReportDataResolutionError,
)


def _create_distinctive_fixture(fixture_path: Path) -> dict[str, object]:
    """Helper creating a valid fixture with distinctive values for Test A."""
    fixture_data: dict[str, object] = {
        "metadata": {
            "study": "Mock Controlled Study for Provenance Testing",
            "evaluation_protocol_version": "ragbench-mock-v9.9",
            "metric_definition_version": "metrics-mock-v9.9",
            "embedding_configuration": {
                "model_identifier": "MockModel/distinctive-777",
                "embedding_dimension": 768,
            },
        },
        "hypotheses": {
            "H1_dense_EN_to_BN": {
                "statement": "H1 statement with custom text",
                "metric": "recall@5",
                "mean_penalty": 0.7777,
                "cohens_dz": 9.99,
                "primary_decision": "SUPPORTED",
                "wilcoxon_signed_rank": {"adj_p_holm": 0.00001, "stars": "***"},
            },
            "H2_dense_BN_to_EN": {
                "statement": "H2 statement with custom text",
                "metric": "recall@5",
                "mean_penalty": -0.6666,
                "cohens_dz": -8.88,
                "primary_decision": "NOT_SUPPORTED",
                "wilcoxon_signed_rank": {"adj_p_holm": 0.00002, "stars": "***"},
            },
            "H3_hybrid_attenuation_EN_to_BN": {
                "statement": "H3 statement with custom text",
                "metric": "recall@5",
                "mean_attenuation": -0.5555,
                "cohens_dz": -4.44,
                "primary_decision": "NOT_SUPPORTED",
                "wilcoxon_signed_rank": {"adj_p_holm": 0.00003, "stars": "***"},
            },
            "H4_hybrid_attenuation_BN_to_EN": {
                "statement": "H4 statement with custom text",
                "metric": "recall@5",
                "mean_attenuation": -0.3333,
                "cohens_dz": -3.33,
                "primary_decision": "NOT_SUPPORTED",
                "wilcoxon_signed_rank": {"adj_p_holm": 0.00004, "stars": "***"},
            },
        },
        "all_comparisons": {
            "dense_EN->BN": {
                "recall@5": {
                    "mean_mono": 0.888,
                    "mean_cross": 0.111,
                    "mean_penalty": 0.777,
                    "ci_95": [0.70, 0.85],
                    "t_stat": 33.33,
                    "t_p_value": 0.00001,
                    "wilcoxon_w": 0.0,
                    "wilcoxon_p_value": 0.00001,
                    "cohens_dz": 9.99,
                    "significance_stars": "***",
                },
                "mrr@5": {
                    "mean_mono": 0.888,
                    "mean_cross": 0.444,
                    "mean_penalty": 0.444,
                    "ci_95": [0.35, 0.53],
                    "t_stat": 12.34,
                    "t_p_value": 0.00001,
                    "wilcoxon_w": 5.0,
                    "wilcoxon_p_value": 0.00001,
                    "cohens_dz": 3.45,
                    "significance_stars": "***",
                },
            },
            "bm25_EN->BN": {
                "recall@5": {
                    "mean_mono": 0.888,
                    "mean_cross": 0.001,
                    "mean_penalty": 0.887,
                    "ci_95": [0.80, 0.95],
                    "t_stat": 40.0,
                    "t_p_value": 0.00001,
                    "wilcoxon_w": 0.0,
                    "wilcoxon_p_value": 0.00001,
                    "cohens_dz": 8.0,
                    "significance_stars": "***",
                },
                "mrr@5": {
                    "mean_mono": 1.0,
                    "mean_cross": 0.02,
                    "mean_penalty": 0.98,
                    "ci_95": [0.90, 1.05],
                    "t_stat": 50.0,
                    "t_p_value": 0.00001,
                    "wilcoxon_w": 0.0,
                    "wilcoxon_p_value": 0.00001,
                    "cohens_dz": 10.0,
                    "significance_stars": "***",
                },
            },
            "hybrid_EN->BN": {
                "recall@5": {
                    "mean_mono": 0.888,
                    "mean_cross": 0.015,
                    "mean_penalty": 0.873,
                    "ci_95": [0.80, 0.94],
                    "t_stat": 38.0,
                    "t_p_value": 0.00001,
                    "wilcoxon_w": 0.0,
                    "wilcoxon_p_value": 0.00001,
                    "cohens_dz": 7.5,
                    "significance_stars": "***",
                },
                "mrr@5": {
                    "mean_mono": 1.0,
                    "mean_cross": 0.08,
                    "mean_penalty": 0.92,
                    "ci_95": [0.85, 0.99],
                    "t_stat": 24.0,
                    "t_p_value": 0.00001,
                    "wilcoxon_w": 0.0,
                    "wilcoxon_p_value": 0.00001,
                    "cohens_dz": 4.8,
                    "significance_stars": "***",
                },
            },
            "dense_BN->EN": {
                "recall@5": {
                    "mean_mono": 0.150,
                    "mean_cross": 0.888,
                    "mean_penalty": -0.738,
                    "ci_95": [-0.80, -0.67],
                    "t_stat": -30.0,
                    "t_p_value": 0.00001,
                    "wilcoxon_w": 0.0,
                    "wilcoxon_p_value": 0.00001,
                    "cohens_dz": -6.5,
                    "significance_stars": "***",
                },
                "mrr@5": {
                    "mean_mono": 0.850,
                    "mean_cross": 0.500,
                    "mean_penalty": 0.350,
                    "ci_95": [0.15, 0.55],
                    "t_stat": 3.5,
                    "t_p_value": 0.001,
                    "wilcoxon_w": 50.0,
                    "wilcoxon_p_value": 0.002,
                    "cohens_dz": 0.70,
                    "significance_stars": "**",
                },
            },
            "bm25_BN->EN": {
                "recall@5": {
                    "mean_mono": 0.180,
                    "mean_cross": 0.035,
                    "mean_penalty": 0.145,
                    "ci_95": [0.04, 0.25],
                    "t_stat": 2.8,
                    "t_p_value": 0.008,
                    "wilcoxon_w": 22.0,
                    "wilcoxon_p_value": 0.0002,
                    "cohens_dz": 0.55,
                    "significance_stars": "***",
                },
                "mrr@5": {
                    "mean_mono": 1.0,
                    "mean_cross": 0.01,
                    "mean_penalty": 0.99,
                    "ci_95": [0.96, 1.02],
                    "t_stat": 90.0,
                    "t_p_value": 0.00001,
                    "wilcoxon_w": 0.0,
                    "wilcoxon_p_value": 0.00001,
                    "cohens_dz": 18.0,
                    "significance_stars": "***",
                },
            },
            "hybrid_BN->EN": {
                "recall@5": {
                    "mean_mono": 0.190,
                    "mean_cross": 0.230,
                    "mean_penalty": -0.040,
                    "ci_95": [-0.22, 0.14],
                    "t_stat": -0.41,
                    "t_p_value": 0.68,
                    "wilcoxon_w": 130.0,
                    "wilcoxon_p_value": 0.45,
                    "cohens_dz": -0.08,
                    "significance_stars": "ns",
                },
                "mrr@5": {
                    "mean_mono": 0.98,
                    "mean_cross": 0.065,
                    "mean_penalty": 0.915,
                    "ci_95": [0.85, 0.98],
                    "t_stat": 28.0,
                    "t_p_value": 0.00001,
                    "wilcoxon_w": 0.0,
                    "wilcoxon_p_value": 0.00001,
                    "cohens_dz": 5.7,
                    "significance_stars": "***",
                },
            },
        },
    }
    fixture_path.write_text(json.dumps(fixture_data), encoding="utf-8")
    return fixture_data


def test_authoritative_value_extraction_test_a() -> None:
    """Test A: Verify reports consume dynamic fixture values rather than hardcoded dicts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fixture_file = Path(tmp_dir) / "custom_summary.json"
        _create_distinctive_fixture(fixture_file)

        generator = AcademicReportGenerator(data_source_path=fixture_file)
        md_content = generator.generate_benchmark_markdown_report()
        tex_table = generator.generate_rq3_latex_table()

        # Assert distinctive values appear in generated outputs
        assert "MockModel/distinctive-777" in md_content
        assert "+0.7777" in md_content
        assert "+9.99" in md_content
        assert "-0.5555" in md_content
        assert "-4.44" in md_content

        assert "MockModel/distinctive-777" in tex_table
        assert "+0.777***" in tex_table
        assert "+9.99" in tex_table


def test_h3_statistical_regression_test_b() -> None:
    """Test B: Verify H3 derives authoritative attenuation (-0.0989, -2.01, significant).

    Must NOT contain erroneous single-condition values (-0.09, 0.4431, ns).
    """
    generator = AcademicReportGenerator()
    md_content = generator.generate_benchmark_markdown_report()

    # Ground truth values for H3 in statistical_summary.json
    assert "-0.0989" in md_content
    assert "-2.01" in md_content
    assert "NOT SUPPORTED (Worse) ($p < 0.001$)" in md_content

    # Erroneous copied values must NOT appear in H3 line
    for line in md_content.splitlines():
        if "| **H3** |" in line:
            assert "-0.09 " not in line and "-0.0900" not in line
            assert "0.4431" not in line
            assert "ns" not in line


def test_missing_artifact_error_test_c() -> None:
    """Test C: Pointing resolver at nonexistent summary fails with clear domain error."""
    nonexistent = Path("nonexistent/path/to/missing_summary.json")
    generator = AcademicReportGenerator(data_source_path=nonexistent)

    with pytest.raises(ReportDataResolutionError) as exc_info:
        generator.generate_benchmark_markdown_report()
    assert "not found" in str(exc_info.value).lower()


def test_malformed_artifact_error_test_d() -> None:
    """Test D: Malformed or structurally incomplete JSON fails with clear domain error."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Invalid JSON
        bad_json = Path(tmp_dir) / "invalid.json"
        bad_json.write_text("{ incomplete json ...", encoding="utf-8")
        gen_bad = AcademicReportGenerator(data_source_path=bad_json)
        with pytest.raises(ReportDataResolutionError) as exc_info1:
            gen_bad.generate_benchmark_markdown_report()
        assert "malformed json" in str(exc_info1.value).lower()

        # Structurally incomplete JSON (missing hypotheses)
        incomplete_json = Path(tmp_dir) / "incomplete.json"
        incomplete_json.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
        gen_inc = AcademicReportGenerator(data_source_path=incomplete_json)
        with pytest.raises(ReportDataResolutionError) as exc_info2:
            gen_inc.generate_benchmark_markdown_report()
        assert "missing structural keys" in str(exc_info2.value).lower()


def test_all_four_hypotheses_resolution_test_e() -> None:
    """Test E: Verify H1-H4 each resolve to their respective pre-registered hypotheses."""
    generator = AcademicReportGenerator()
    summary = generator._resolve_rq3_data()

    assert set(summary.hypotheses.keys()) == {
        "H1_dense_EN_to_BN",
        "H2_dense_BN_to_EN",
        "H3_hybrid_attenuation_EN_to_BN",
        "H4_hybrid_attenuation_BN_to_EN",
    }

    h1 = summary.hypotheses["H1_dense_EN_to_BN"]
    assert h1.decision == "SUPPORTED"
    assert round(h1.delta, 4) == 0.8613
    assert round(h1.cohens_dz, 2) == 7.45
    assert h1.stars == "***"

    h2 = summary.hypotheses["H2_dense_BN_to_EN"]
    assert h2.decision == "NOT_SUPPORTED"
    assert round(h2.delta, 4) == -0.8382
    assert round(h2.cohens_dz, 2) == -7.11
    assert h2.stars == "***"

    h3 = summary.hypotheses["H3_hybrid_attenuation_EN_to_BN"]
    assert h3.decision == "NOT_SUPPORTED"
    assert round(h3.delta, 4) == -0.0989
    assert round(h3.cohens_dz, 2) == -2.01
    assert h3.stars == "***"

    h4 = summary.hypotheses["H4_hybrid_attenuation_BN_to_EN"]
    assert h4.decision == "NOT_SUPPORTED"
    assert round(h4.delta, 4) == -0.7981
    assert round(h4.cohens_dz, 2) == -1.75
    assert h4.stars == "***"


def test_deterministic_generation_test_f() -> None:
    """Test F: Generating reports twice from same evidence produces byte-identical output."""
    generator = AcademicReportGenerator()

    out1 = generator.generate_benchmark_markdown_report()
    out2 = generator.generate_benchmark_markdown_report()
    assert out1.encode("utf-8") == out2.encode("utf-8")

    tex1 = generator.generate_rq3_latex_table()
    tex2 = generator.generate_rq3_latex_table()
    assert tex1.encode("utf-8") == tex2.encode("utf-8")

    wp1 = generator.generate_whitepaper_latex()
    wp2 = generator.generate_whitepaper_latex()
    assert wp1.encode("utf-8") == wp2.encode("utf-8")


def test_generate_full_report_package() -> None:
    """Verify package generation produces all required artifacts and bundles IEEEtran.cls."""
    generator = AcademicReportGenerator()
    with tempfile.TemporaryDirectory() as tmp_dir:
        package = generator.generate_full_report_package(Path(tmp_dir))

        assert "rq3_table" in package
        assert "benchmark_report" in package
        assert "whitepaper" in package
        assert "ieeetran_class" in package

        for path in package.values():
            assert path.exists()
            assert path.stat().st_size > 0
            # Confirm files are readable as UTF-8 (or binary for cls)
            if path.suffix in {".tex", ".md"}:
                text = path.read_text(encoding="utf-8")
                assert len(text) > 0
