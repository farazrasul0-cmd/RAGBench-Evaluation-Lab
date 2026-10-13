# RAGBench-Evaluation-Lab

**A reproducible framework for evaluating Retrieval-Augmented Generation pipelines through controlled retrieval experiments.**

[![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg)](https://github.com/farazrasul0-cmd/RAGBench-Evaluation-Lab)
[![Python](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

Retrieval-Augmented Generation (RAG) systems are frequently evaluated using end-to-end question answering benchmarks that obscure intermediate retrieval failures. When a generation pipeline produces an incorrect response, diagnosing whether the failure stemmed from document chunking boundaries, embedding representation collapse, lexical mismatches, or generative hallucination remains difficult without controlled isolation.

**RAGBench-Evaluation-Lab** is an empirical research platform designed to evaluate RAG pipelines under strictly controlled experimental conditions. Rather than treating RAG as an opaque black box, RAGBench parameterizes the pipeline into factorial variations across document chunking geometry, dense versus lexical representations, rank fusion algorithms, and retrieval depth ($K$), backed by paired non-parametric statistical hypothesis testing.

---

## Research Contributions

1. **Controlled RAG Evaluation Framework:** Isolates retrieval variables under identical chunking, tokenization, and query sets to eliminate confounding factors across pipeline comparisons.
2. **Systematic Retrieval Strategy Comparisons:** Benchmarks dense semantic retrieval, lexical BM25 Okapi, and Hybrid Reciprocal Rank Fusion (RRF, $k=60$) across identical document corpora.
3. **Multilingual Retrieval Evaluation:** Measures cross-lingual language transfer penalties and asymmetric degradation across English and Bengali information units.
4. **Reproducible Experiment Orchestration:** Enforces immutable configuration hashes, deterministic seeds, and cryptographic SHA-256 replication manifests for all stored evidence.
5. **Interactive Research Workbench:** Provides a React-based visual exploration environment featuring active Pareto frontier validation, multi-metric radar charts, and claim-level citation attribution inspection.

---

## System Architecture

RAGBench operates as a dual-engine architecture:

```
                    RAGBench Platform Topology
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────────────────┐       ┌───────────────────────────────┐
│        Research Engine        │       │      Research Workbench       │
│        (Python / CLI)         │       │        (React 19 / UI)        │
├───────────────────────────────┤       ├───────────────────────────────┤
│ • Deterministic Chunker       │       │ • Factorial Matrix Sweeps     │
│ • Dense, BM25 & Hybrid RRF    │       │ • Active Pareto Validation    │
│ • Paired Statistical Engine   │       │ • Bounded Metric Radar        │
│ • Automated LaTeX Generator   │       │ • Claim Trace Inspector       │
└───────────────┬───────────────┘       └───────────────▲───────────────┘
                │                                       │
                │        REST API / OpenAPI Contract    │
                └───────────────────────────────────────┘
```

Detailed technical contracts, metric formulas, and persistence schemas are documented in [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md).

---

## Experimental Baseline & Pre-Registered Findings

In the pre-registered Phase G reference study, RAGBench evaluated cross-lingual transfer between English and Bengali across $N=25$ matched, non-pseudoreplicated bilingual information units (12 factorial conditions) using `BAAI/bge-m3` (1024-d, cosine similarity, fixed 200/20 chunking, $K=5$).

Hypothesis testing applied two-sided Wilcoxon signed-rank tests ($W, p_W$) as the primary non-parametric criterion, two-sided paired Student's $t$-tests ($t, p_t$), and Step-Down Holm-Bonferroni correction with family size $m=4$:

| Hypothesis | Description | Empirical $\bar{\Delta}$ | Cohen's $d_z$ | Approved Verdict |
| :--- | :--- | :---: | :---: | :--- |
| **H1** | Dense cross-lingual penalty for EN $\to$ BN | **+0.8613** | **+7.45** | **Supported** ($p < 0.001$, large penalty) |
| **H2** | Dense cross-lingual penalty for BN $\to$ EN | **-0.8382** | **-7.11** | **Not Supported** (Retrieval gain observed) |
| **H3** | Hybrid RRF attenuates EN $\to$ BN penalty | **-0.0989** | **-2.01** | **Not Supported** (Hybrid exacerbated penalty, $p < 0.001$) |
| **H4** | Hybrid RRF attenuates BN $\to$ EN penalty | **-0.7981** | **-1.75** | **Not Supported** (Hybrid attenuated less than dense, $p < 0.001$) |

Cryptographic digests for all raw observations and statistical files are listed in [`docs/ARTIFACT_HASHES.md`](./docs/ARTIFACT_HASHES.md).

---

## Quickstart & Reproduction

### Installation

```bash
# 1. Clone repository
git clone https://github.com/farazrasul0-cmd/RAGBench-Evaluation-Lab.git
cd RAGBench-Evaluation-Lab

# 2. Setup backend environment
python3.12 -m venv backend/.venv
source backend/.venv/bin/activate       # On Windows: .\backend\.venv\Scripts\Activate.ps1
pip install -e "backend[dev]"

# 3. Setup frontend dependencies
cd frontend && npm install && cd ..
```

### Running with Docker Compose

```bash
docker compose up -d
```
* Research Workbench UI: [http://localhost:3000](http://localhost:3000)
* REST API Documentation: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)

### Generating Academic Reports

Generate publication-ready LaTeX tables, whitepaper manuscripts, and Markdown summaries directly from empirical artifacts:

```bash
ragbench report generate --output-dir reports/
```

For detailed replication steps and environment requirements, see [`docs/REPRODUCIBILITY.md`](./docs/REPRODUCIBILITY.md).

---

## Documentation

* **[`docs/REPRODUCIBILITY.md`](./docs/REPRODUCIBILITY.md):** Complete guide to reproducing benchmarks and verification checks.
* **[`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md):** Formal metric formulations, component contracts, and Pareto control signatures.
* **[`docs/ARTIFACT_HASHES.md`](./docs/ARTIFACT_HASHES.md):** SHA-256 cryptographic manifest for empirical datasets and reference databases.
* **[`CHANGELOG.md`](./CHANGELOG.md):** Version history and release notes.

---

## Citation

If you use RAGBench in your research, please cite:

```bibtex
@software{zain2026ragbench,
  author = {Zain, Syed Faraz},
  title = {{RAGBench-Evaluation-Lab: A Parameterizable Scientific Benchmarking Platform for RAG Pipelines}},
  url = {https://github.com/farazrasul0-cmd/RAGBench-Evaluation-Lab},
  version = {1.0.0},
  year = {2026}
}
```

See [`CITATION.cff`](./CITATION.cff) for machine-readable citation metadata.

---

## License

This project is licensed under the MIT License — see the [`LICENSE`](./LICENSE) file for details.
