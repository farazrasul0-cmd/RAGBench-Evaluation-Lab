"""Unit tests for AcademicReportGenerator publication artifact generator."""

import tempfile
from pathlib import Path

from app.services.report_generator import AcademicReportGenerator


def test_generate_rq3_latex_table() -> None:
    """Verify RQ3 LaTeX table generation contains booktabs and accurate findings."""
    generator = AcademicReportGenerator()
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "rq3_table.tex"
        content = generator.generate_rq3_latex_table(out_file)

        assert out_file.exists()
        assert r"\begin{table*}" in content
        assert r"\toprule" in content
        assert r"\bottomrule" in content
        assert "BAAI/bge-m3" in content
        assert "+0.861***" in content
        assert "-0.838***" in content


def test_generate_benchmark_markdown_report() -> None:
    """Verify executive markdown report contains pre-registered hypothesis status."""
    generator = AcademicReportGenerator()
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "BENCHMARK_REPORT.md"
        content = generator.generate_benchmark_markdown_report(out_file)

        assert out_file.exists()
        assert "eb3bb4d" in content
        assert "c889df1" in content
        assert "+0.8613" in content
        assert "SUPPORTED" in content
        assert "NOT SUPPORTED" in content


def test_generate_whitepaper_latex() -> None:
    """Verify compilable whitepaper LaTeX document structure."""
    generator = AcademicReportGenerator()
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "whitepaper.tex"
        content = generator.generate_whitepaper_latex(out_file)

        assert out_file.exists()
        assert r"\documentclass" in content
        assert r"\begin{abstract}" in content
        assert r"\maketitle" in content
        assert r"\end{document}" in content


def test_generate_full_report_package() -> None:
    """Verify complete report package generation produces all required artifacts."""
    generator = AcademicReportGenerator()
    with tempfile.TemporaryDirectory() as tmp_dir:
        package = generator.generate_full_report_package(Path(tmp_dir))

        assert "rq3_table" in package
        assert "benchmark_report" in package
        assert "whitepaper" in package
        for path in package.values():
            assert path.exists()
            assert path.stat().st_size > 0
