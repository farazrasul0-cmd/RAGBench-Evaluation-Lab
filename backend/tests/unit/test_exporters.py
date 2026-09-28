import hashlib
import json
import shutil
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.engine.analysis.export import (
    AcademicLatexExporter,
    CSVExporter,
    ReplicationArchiveExporter,
    escape_latex,
)
from app.engine.analysis.statistics import StatisticalComparator
from app.models.entities import Experiment, ExperimentRun
from app.schemas.analysis import MetricSignificanceReport


@pytest.fixture
def exporter_workspace() -> Generator[Path, None, None]:
    """Create isolated local workspace directory for exporter tests."""
    ws = Path("test_exporter_workspace")
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True, exist_ok=True)
    try:
        yield ws
    finally:
        shutil.rmtree(ws, ignore_errors=True)


@pytest.fixture
def sample_report() -> MetricSignificanceReport:
    exp = MagicMock(spec=Experiment, dataset_version_id="ver-export-1")
    run_a = MagicMock(spec=ExperimentRun, id="run-alpha-1234", experiment=exp)
    run_b = MagicMock(spec=ExperimentRun, id="run-beta-5678", experiment=exp)

    metrics_a = {
        "q1": {"recall@5": 0.60, "ndcg@5": 0.50},
        "q2": {"recall@5": 0.70, "ndcg@5": 0.55},
        "q3": {"recall@5": 0.65, "ndcg@5": 0.52},
    }
    metrics_b = {
        "q1": {"recall@5": 0.80, "ndcg@5": 0.65},
        "q2": {"recall@5": 0.85, "ndcg@5": 0.70},
        "q3": {"recall@5": 0.75, "ndcg@5": 0.60},
    }

    report = StatisticalComparator.compare_runs(
        run_a=run_a,
        run_b=run_b,
        metrics_by_query_a=metrics_a,
        metrics_by_query_b=metrics_b,
        correction_method="holm",
    )
    return report


def test_latex_escape_special_characters() -> None:
    """Verify special LaTeX characters are properly escaped."""
    raw = "recall@5_score & 100% #1 {result}"
    escaped = escape_latex(raw)
    assert "\\_" in escaped
    assert "\\&" in escaped
    assert "\\%" in escaped
    assert "\\#" in escaped
    assert "\\{" in escaped
    assert "\\}" in escaped


def test_academic_latex_table_structure(sample_report: MetricSignificanceReport) -> None:
    """Verify LaTeX exporter emits booktabs syntax, bold maximums, and CI bounds."""
    latex = AcademicLatexExporter.export_comparison_table(sample_report)

    assert "\\begin{table*}" in latex
    assert "\\toprule" in latex
    assert "\\midrule" in latex
    assert "\\bottomrule" in latex
    assert "\\end{table*}" in latex
    assert "Cohen's $d_z$" in latex
    assert "Adj. $p$ (Holm)" in latex
    assert "\\textbf{" in latex


def test_csv_exporter_summary_and_raw(sample_report: MetricSignificanceReport) -> None:
    """Verify CSV summary and raw observation exports match expected schema."""
    summary_csv = CSVExporter.export_summary_csv(sample_report)
    assert "metric,mean_run_a,mean_run_b,mean_difference" in summary_csv
    assert "cohens_dz" in summary_csv
    assert "wilcoxon_pvalue" in summary_csv
    assert "recall@5" in summary_csv
    assert "ndcg@5" in summary_csv

    raw_csv = CSVExporter.export_raw_paired_csv(sample_report)
    lines = raw_csv.strip().splitlines()
    assert lines[0] == "query_id,metric,score_run_a,score_run_b,difference"
    # 2 metrics * 3 queries = 6 data rows + 1 header = 7 rows
    assert len(lines) == 7
    assert "q1,recall@5,0.600000,0.800000,0.200000" in lines


def test_replication_archive_package(
    sample_report: MetricSignificanceReport, exporter_workspace: Path
) -> None:
    """Verify replication exporter generates valid manifest and verified SHA-256 hashes."""
    target_dir = exporter_workspace / "replication_pkg"
    config_diff = {"retrieval": {"mode": ["bm25", "dense"]}}

    ReplicationArchiveExporter.export_package(
        report=sample_report,
        target_dir=target_dir,
        config_diff=config_diff,
    )

    # Check files exist
    manifest_file = target_dir / "manifest.json"
    hashes_file = target_dir / "hashes.json"
    summary_json = target_dir / "statistical_summary.json"
    tex_file = target_dir / "comparison_table.tex"
    csv_file = target_dir / "statistical_summary.csv"
    raw_file = target_dir / "raw_paired_observations.csv"
    diff_file = target_dir / "config_diff.json"

    assert manifest_file.exists()
    assert hashes_file.exists()
    assert summary_json.exists()
    assert tex_file.exists()
    assert csv_file.exists()
    assert raw_file.exists()
    assert diff_file.exists()

    # Validate manifest content
    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["comparison_id"] == sample_report.comparison_id
    assert "system_environment" in manifest
    assert "os" in manifest["system_environment"]

    # Validate hashes
    with open(hashes_file, encoding="utf-8") as f:
        hashes = json.load(f)

    for rel_path, expected_hash in hashes.items():
        actual_hash = hashlib.sha256((target_dir / rel_path).read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}!"
