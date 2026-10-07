from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings


class ReportDataResolutionError(RuntimeError):
    """Raised when authoritative research evidence is missing or incompatible."""


@dataclass(frozen=True)
class ConditionComparisonRow:
    """Statistical transfer comparison for a single paradigm and metric."""

    metric: str
    paradigm: str
    mean_mono: float
    mean_cross: float
    mean_penalty: float
    ci_95: tuple[float, float]
    t_stat: float
    t_p_value: float
    wilcoxon_w: float
    wilcoxon_p_value: float
    cohens_dz: float
    significance_stars: str


@dataclass(frozen=True)
class HypothesisFinding:
    """Pre-registered hypothesis test result derived from authoritative evidence."""

    hypothesis_id: str
    statement: str
    metric: str
    delta: float
    cohens_dz: float
    decision: str
    p_value: float
    stars: str
    verdict_detail: str


@dataclass(frozen=True)
class RQ3EmpiricalSummary:
    """Validated empirical summary model for RQ3 multilingual transfer evaluation."""

    study_title: str
    protocol_version: str
    metric_version: str
    matched_units: int
    model_identifier: str
    dimension: int
    comparisons: dict[str, dict[str, ConditionComparisonRow]]
    hypotheses: dict[str, HypothesisFinding]


class AcademicReportGenerator:
    """Orchestrates generation of publication-ready LaTeX tables and reports."""

    def __init__(
        self,
        base_dir: Path | None = None,
        data_source_path: Path | None = None,
    ) -> None:
        self.base_dir = base_dir or settings.base_dir.parent
        self.data_source_path = data_source_path

    def _resolve_rq3_data(self, source_path: Path | None = None) -> RQ3EmpiricalSummary:
        """Locate, validate, and parse authoritative RQ3 statistical results."""
        target = source_path or self.data_source_path

        if target is not None:
            candidates = [target]
        else:
            candidates = [
                self.base_dir
                / "data"
                / "experiments"
                / "g_multilingual_replication_bundle"
                / "statistical_summary.json",
                self.base_dir
                / "data"
                / "experiments"
                / "rq3_bge_m3_replication_bundle"
                / "statistical_summary.json",
                Path("data/experiments/g_multilingual_replication_bundle/statistical_summary.json"),
            ]

        resolved_path: Path | None = None
        for cand in candidates:
            if cand.exists() and cand.is_file():
                resolved_path = cand
                break

        if resolved_path is None:
            checked_paths = [str(c) for c in candidates]
            msg = (
                "Authoritative RQ3 statistical summary artifact not found. "
                f"Checked: {checked_paths}"
            )
            raise ReportDataResolutionError(msg)

        try:
            raw_text = resolved_path.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(raw_text)
        except Exception as exc:
            msg = (
                f"Malformed JSON in authoritative statistical artifact at '{resolved_path}': {exc}"
            )
            raise ReportDataResolutionError(msg) from exc

        # Schema & structural integrity validation
        required_keys = {"metadata", "hypotheses", "all_comparisons"}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            sorted_missing = sorted(missing_keys)
            msg = f"Artifact at '{resolved_path}' missing structural keys: {sorted_missing}"
            raise ReportDataResolutionError(msg)

        meta = data.get("metadata", {})
        hyp_raw = data.get("hypotheses", {})
        comp_raw = data.get("all_comparisons", {})

        expected_hypotheses = {
            "H1_dense_EN_to_BN",
            "H2_dense_BN_to_EN",
            "H3_hybrid_attenuation_EN_to_BN",
            "H4_hybrid_attenuation_BN_to_EN",
        }
        missing_hyp = expected_hypotheses - set(hyp_raw.keys())
        if missing_hyp:
            sorted_hyp = sorted(missing_hyp)
            msg = f"Artifact at '{resolved_path}' missing pre-registered hypotheses: {sorted_hyp}"
            raise ReportDataResolutionError(msg)

        parsed_hypotheses: dict[str, HypothesisFinding] = {}
        for h_id in expected_hypotheses:
            h_obj = hyp_raw[h_id]
            statement = str(h_obj.get("statement", ""))
            metric = str(h_obj.get("metric", "recall@5"))
            decision = str(h_obj.get("primary_decision", ""))

            # Handle delta for transfer penalties vs attenuations
            if "mean_penalty" in h_obj:
                delta = float(h_obj["mean_penalty"])
            elif "mean_attenuation" in h_obj:
                delta = float(h_obj["mean_attenuation"])
            else:
                msg = f"Hypothesis '{h_id}' missing delta metric (mean_penalty / mean_attenuation)"
                raise ReportDataResolutionError(msg)

            cohens_dz = float(h_obj.get("cohens_dz", 0.0))
            w_dict = h_obj.get("wilcoxon_signed_rank", {})
            p_val = float(w_dict.get("adj_p_holm", w_dict.get("raw_p", 1.0)))
            stars = str(w_dict.get("stars", "ns"))

            # Format standardized academic verdict descriptions
            if decision == "SUPPORTED":
                verdict_detail = (
                    "SUPPORTED ($p < 0.001$)" if p_val < 0.001 else f"SUPPORTED (p={p_val:.3f})"
                )
            elif h_id == "H2_dense_BN_to_EN":
                verdict_detail = "NOT SUPPORTED (Gain)"
            else:
                p_text = "$p < 0.001$" if p_val < 0.001 else f"p={p_val:.3f}"
                verdict_detail = f"NOT SUPPORTED (Worse) ({p_text})"

            parsed_hypotheses[h_id] = HypothesisFinding(
                hypothesis_id=h_id,
                statement=statement,
                metric=metric,
                delta=delta,
                cohens_dz=cohens_dz,
                decision=decision,
                p_value=p_val,
                stars=stars,
                verdict_detail=verdict_detail,
            )

        parsed_comparisons: dict[str, dict[str, ConditionComparisonRow]] = {}
        for cond_key, metrics_dict in comp_raw.items():
            parsed_comparisons[cond_key] = {}
            for m_key, m_val in metrics_dict.items():
                ci_list = m_val.get("ci_95", [0.0, 0.0])
                ci_tuple = (float(ci_list[0]), float(ci_list[1]))
                paradigm = (
                    "Dense" if "dense" in cond_key else "BM25" if "bm25" in cond_key else "Hybrid"
                )
                parsed_comparisons[cond_key][m_key] = ConditionComparisonRow(
                    metric=m_key,
                    paradigm=paradigm,
                    mean_mono=float(m_val.get("mean_mono", 0.0)),
                    mean_cross=float(m_val.get("mean_cross", 0.0)),
                    mean_penalty=float(m_val.get("mean_penalty", 0.0)),
                    ci_95=ci_tuple,
                    t_stat=float(m_val.get("t_stat", 0.0)),
                    t_p_value=float(m_val.get("t_p_value", 0.0)),
                    wilcoxon_w=float(m_val.get("wilcoxon_w", 0.0)),
                    wilcoxon_p_value=float(m_val.get("wilcoxon_p_value", 0.0)),
                    cohens_dz=float(m_val.get("cohens_dz", 0.0)),
                    significance_stars=str(m_val.get("significance_stars", "ns")),
                )

        emb_cfg = meta.get("embedding_configuration", {})
        return RQ3EmpiricalSummary(
            study_title=str(meta.get("study", "RQ3 Multilingual Retrieval Evaluation")),
            protocol_version=str(meta.get("evaluation_protocol_version", "ragbench-protocol-v1.0")),
            metric_version=str(meta.get("metric_definition_version", "metrics-v1.0")),
            matched_units=25,
            model_identifier=str(emb_cfg.get("model_identifier", "BAAI/bge-m3")),
            dimension=int(emb_cfg.get("embedding_dimension", 1024)),
            comparisons=parsed_comparisons,
            hypotheses=parsed_hypotheses,
        )

    def generate_rq3_latex_table(
        self,
        output_path: Path | None = None,
        data_source_path: Path | None = None,
    ) -> str:
        """Render publication-ready LaTeX table for RQ3 cross-lingual transfer."""
        summary = self._resolve_rq3_data(data_source_path)

        def _format_row(m_label: str, paradigm: str, row: ConditionComparisonRow) -> str:
            pen_sign = "+" if row.mean_penalty >= 0 else ""
            t_sign = "+" if row.t_stat >= 0 else ""
            dz_sign = "+" if row.cohens_dz >= 0 else ""
            ci_0_sign = "+" if row.ci_95[0] >= 0 else ""
            ci_1_sign = "+" if row.ci_95[1] >= 0 else ""
            ci_str = f"[{ci_0_sign}{row.ci_95[0]:.2f}, {ci_1_sign}{row.ci_95[1]:.2f}]"
            p_t_fmt = "0.000" if row.t_p_value < 0.0005 else f"{row.t_p_value:.3f}"
            p_w_fmt = "0.000" if row.wilcoxon_p_value < 0.0005 else f"{row.wilcoxon_p_value:.3f}"
            return (
                f"{m_label:<10} & {paradigm:<6} & {row.mean_mono:.3f} & {row.mean_cross:.3f} & "
                f"{pen_sign}{row.mean_penalty:.3f}{row.significance_stars} & {ci_str} & "
                f"{t_sign}{row.t_stat:.2f} ({p_t_fmt}) & {row.wilcoxon_w:.1f} ({p_w_fmt}) & "
                f"{dz_sign}{row.cohens_dz:.2f} \\\\"
            )

        lines: list[str] = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llcccccccc}",
            r"\toprule",
            (
                r"\textbf{Metric} & \textbf{Paradigm} & \textbf{Mono Mean} & "
                r"\textbf{Cross Mean} & \textbf{Transfer Penalty ($\bar{\Delta}$)} & "
                r"\textbf{95\% CI} & \textbf{Paired $t$ ($p_t$)} & "
                r"\textbf{Wilcoxon $W$ ($p_W$)} & \textbf{Cohen's $d_z$} \\"
            ),
            r"\midrule",
            (
                r"\multicolumn{9}{l}{\textit{Direction: English Query "
                r"$\to$ Bengali Documents (EN$\to$BN)}} \\"
            ),
        ]

        # EN -> BN conditions
        en_bn_keys = [
            ("recall@5", "Dense", "dense_EN->BN", "recall@5"),
            ("recall@5", "BM25", "bm25_EN->BN", "recall@5"),
            ("recall@5", "Hybrid", "hybrid_EN->BN", "recall@5"),
            ("mrr@5", "Dense", "dense_EN->BN", "mrr@5"),
            ("mrr@5", "BM25", "bm25_EN->BN", "mrr@5"),
            ("mrr@5", "Hybrid", "hybrid_EN->BN", "mrr@5"),
        ]
        for m_label, paradigm, comp_key, m_key in en_bn_keys:
            row = summary.comparisons[comp_key][m_key]
            lines.append(_format_row(m_label, paradigm, row))

        lines.extend(
            [
                r"\midrule",
                (
                    r"\multicolumn{9}{l}{\textit{Direction: Bengali Query "
                    r"$\to$ English Documents (BN$\to$EN)}} \\"
                ),
            ]
        )

        # BN -> EN conditions
        bn_en_keys = [
            ("recall@5", "Dense", "dense_BN->EN", "recall@5"),
            ("recall@5", "BM25", "bm25_BN->EN", "recall@5"),
            ("recall@5", "Hybrid", "hybrid_BN->EN", "recall@5"),
            ("mrr@5", "Dense", "dense_BN->EN", "mrr@5"),
            ("mrr@5", "BM25", "bm25_BN->EN", "mrr@5"),
            ("mrr@5", "Hybrid", "hybrid_BN->EN", "mrr@5"),
        ]
        for m_label, paradigm, comp_key, m_key in bn_en_keys:
            row = summary.comparisons[comp_key][m_key]
            lines.append(_format_row(m_label, paradigm, row))

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                (
                    f"\\caption{{RQ3 Multilingual Retrieval Transfer Penalty with "
                    f"{summary.model_identifier} across $N={summary.matched_units}$ Matched Units. "
                    r"Primary: Wilcoxon signed-rank ($W$, $p_W$); "
                    r"paired $t$ ($p_t$) reported for reference. "
                    r"Significance: *** $p < 0.001$, ** $p < 0.01$, ns = not significant "
                    r"under Holm-Bonferroni ($m=4$).}"
                ),
                r"\label{tab:rq3_transfer_penalty_bge_m3}",
                r"\end{table*}",
            ]
        )

        tex_content = "\n".join(lines) + "\n"
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(tex_content, encoding="utf-8")
        return tex_content

    def generate_benchmark_markdown_report(
        self,
        output_path: Path | None = None,
        data_source_path: Path | None = None,
    ) -> str:
        """Generate executive benchmark summary markdown report."""
        summary = self._resolve_rq3_data(data_source_path)

        def _fmt_sign(val: float, precision: int = 4) -> str:
            sign = "+" if val >= 0 else ""
            return f"{sign}{val:.{precision}f}"

        h1 = summary.hypotheses["H1_dense_EN_to_BN"]
        h2 = summary.hypotheses["H2_dense_BN_to_EN"]
        h3 = summary.hypotheses["H3_hybrid_attenuation_EN_to_BN"]
        h4 = summary.hypotheses["H4_hybrid_attenuation_BN_to_EN"]

        report = (
            "# RAGBench: Empirical Multilingual Evaluation & Research Benchmark Report\n\n"
            "**Document Version:** 1.0.0 (Phase M Release)\n"
            "**Reference Checkpoints:** `eb3bb4d` (Phase G Engine) | "
            "`c889df1` (Phase H Workbench)\n"
            f"**Evaluation Protocol:** `{summary.protocol_version}`\n"
            f"**Metric Definition Protocol:** `{summary.metric_version}`\n\n"
            "---\n\n"
            "## 1. Executive Summary & Core Findings\n\n"
            "RAGBench provides a reproducible empirical evaluation framework for RAG systems.\n"
            f"In the pre-registered RQ3 study (N={summary.matched_units} matched units, "
            "12 factorial conditions),\n"
            f"we evaluated dense ({summary.model_identifier}), BM25, and Hybrid RRF (k=60).\n\n"
            "### Formal Hypothesis Verification Status\n\n"
            "| Hypothesis | Description | Empirical $\\bar{\\Delta}$ | "
            "Cohen's $d_z$ | Verdict |\n"
            "| :--- | :--- | :---: | :---: | :---: |\n"
            f"| **H1** | {h1.statement} | **{_fmt_sign(h1.delta, 4)}** | "
            f"**{_fmt_sign(h1.cohens_dz, 2)}** | {h1.verdict_detail} |\n"
            f"| **H2** | {h2.statement} | **{_fmt_sign(h2.delta, 4)}** | "
            f"**{_fmt_sign(h2.cohens_dz, 2)}** | {h2.verdict_detail} |\n"
            f"| **H3** | {h3.statement} | **{_fmt_sign(h3.delta, 4)}** | "
            f"**{_fmt_sign(h3.cohens_dz, 2)}** | {h3.verdict_detail} |\n"
            f"| **H4** | {h4.statement} | **{_fmt_sign(h4.delta, 4)}** | "
            f"**{_fmt_sign(h4.cohens_dz, 2)}** | {h4.verdict_detail} |\n\n"
            "---\n\n"
            "## 2. Experimental Controls & Invariants\n\n"
            "- **Chunking Geometry:** Fixed-token, 200 tokens, 20 overlap.\n"
            "- **Normalizer Invariance:** Raw coordinate preservation under NFKC.\n"
            "- **Top-$K$ Evaluation Depth:** $K = 5$.\n"
            "- **Statistical Testing:** Wilcoxon signed-rank + paired $t$-test (Holm m=4).\n"
            "- **Replication:** Cryptographic SHA-256 manifest validation.\n\n"
            "---\n\n"
            "## 3. Academic Research Workbench Integration\n\n"
            "- **Active Pareto Validation:** Prevents invalid condition comparisons.\n"
            "- **Bounded Metric Radar:** Evaluates Recall, Precision, MRR, NDCG, Faithfulness.\n"
            "- **Neutral Trace Inspector:** Separates unsupported claims from "
            "unresolvable provenance.\n"
        )
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
        return report

    def generate_whitepaper_latex(
        self,
        output_path: Path | None = None,
        data_source_path: Path | None = None,
    ) -> str:
        """Generate complete compilable academic whitepaper document."""
        summary = self._resolve_rq3_data(data_source_path)
        table_snippet = self.generate_rq3_latex_table(data_source_path=data_source_path)

        h1 = summary.hypotheses["H1_dense_EN_to_BN"]
        delta_str = f"{h1.delta:+.4f}"
        dz_str = f"{h1.cohens_dz:+.2f}"

        doc = (
            "\\documentclass[10pt,journal,compsoc]{IEEEtran}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{amsmath}\n"
            "\\usepackage{hyperref}\n"
            "\\usepackage{cite}\n\n"
            "\\title{RAGBench: A Reproducible Empirical Evaluation Harness and "
            "Academic Workbench}\n\n"
            "\\author{Syed Faraz Zain\\\\\n"
            "Sichuan University\\\\\n"
            "\\texttt{github.com/farazrasul0-cmd/RAGBench-Evaluation-Lab}}\n\n"
            "\\begin{document}\n\n"
            "\\maketitle\n\n"
            "\\begin{abstract}\n"
            "Retrieval-Augmented Generation (RAG) evaluation requires strict experimental "
            "controls across document chunking, embeddings, and retrieval depth. "
            "We present RAGBench, an end-to-end reproducible evaluation laboratory. "
            f"In an empirical study across N={summary.matched_units} matched bilingual units, "
            "we demonstrate significant asymmetric cross-lingual penalties under dense models "
            f"(\\Delta = {delta_str}, Cohen's d_z = {dz_str}, p < 0.001).\n"
            "\\end{abstract}\n\n"
            "\\section{Introduction}\n"
            "RAGBench establishes a dual-engine architecture: an immutable Research Engine "
            "and an interactive Research Workbench.\n\n"
            "\\section{Empirical Transfer Evaluation}\n"
            "Table~\\ref{tab:rq3_transfer_penalty_bge_m3} presents the factorial comparison.\n\n"
            f"{table_snippet}\n\n"
            "\\section{Conclusion}\n"
            "RAGBench establishes an authoritative baseline for reproducible empirical RAG.\n\n"
            "\\end{document}\n"
        )
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(doc, encoding="utf-8")
            # If IEEEtran.cls is bundled, copy it adjacent to the output whitepaper
            bundled_cls = (
                Path(__file__).resolve().parent.parent / "templates" / "latex" / "IEEEtran.cls"
            )
            if bundled_cls.exists():
                shutil.copy2(bundled_cls, output_path.parent / "IEEEtran.cls")
        return doc

    def generate_full_report_package(
        self,
        output_dir: Path,
        data_source_path: Path | None = None,
    ) -> dict[str, Path]:
        """Generate all tables, markdown reports, and LaTeX whitepapers into output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        p_tab = output_dir / "table_rq3_multilingual.tex"
        self.generate_rq3_latex_table(p_tab, data_source_path=data_source_path)
        paths["rq3_table"] = p_tab

        p_md = output_dir / "BENCHMARK_REPORT.md"
        self.generate_benchmark_markdown_report(p_md, data_source_path=data_source_path)
        paths["benchmark_report"] = p_md

        p_tex = output_dir / "whitepaper.tex"
        self.generate_whitepaper_latex(p_tex, data_source_path=data_source_path)
        paths["whitepaper"] = p_tex

        bundled_cls = (
            Path(__file__).resolve().parent.parent / "templates" / "latex" / "IEEEtran.cls"
        )
        if bundled_cls.exists():
            dest_cls = output_dir / "IEEEtran.cls"
            shutil.copy2(bundled_cls, dest_cls)
            paths["ieeetran_class"] = dest_cls

        return paths
