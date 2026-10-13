# RAGBench-Evaluation-Lab

**A reproducible framework for evaluating Retrieval-Augmented Generation pipelines through controlled retrieval experiments.**

[![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg)](https://github.com/farazrasul0-cmd/RAGBench-Evaluation-Lab)
[![Python](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What is RAGBench?

**RAGBench-Evaluation-Lab** is an experimental evaluation framework designed to benchmark Retrieval-Augmented Generation (RAG) pipelines under strictly controlled, parameterizable conditions.

When a RAG pipeline fails, diagnosing whether the failure was caused by chunking boundaries, embedding degradation, lexical mismatch, or LLM hallucination requires isolating intermediate retrieval stages. RAGBench isolates these variables through factorial permutations across chunking geometry, retrieval modalities (dense, sparse, hybrid), and retrieval depth ($K$), backed by paired non-parametric statistical hypothesis testing.

### What Research Questions Does It Study?

This repository investigates retrieval behavior across multilingual and technical domains:
1. **Retrieval Paradigm Trade-offs:** How dense bi-encoder embeddings (`BAAI/bge-m3`), lexical search (BM25 Okapi), and Hybrid Reciprocal Rank Fusion (RRF, $k=60$) trade off retrieval recall against latency.
2. **Cross-Lingual Language Transfer:** Measuring empirical degradation and asymmetric transfer penalties when queries and documents cross language boundaries (English $\leftrightarrow$ Bengali).
3. **Retrieval Sensitivity:** How chunk size, token overlap, and retrieval depth affect downstream answer faithfulness and citation attribution.

### What Did We Build?

```
RAGBench-Evaluation-Lab
├── Research Engine (Backend CLI & Core)
│   ├── ingestion       Deterministic parsing & NFKC normalization
│   ├── chunking        Fixed, sentence, and character-level geometry
│   ├── embeddings      Dense vector providers & Qdrant/SQLite storage
│   ├── retrieval       Dense cosine, BM25 Okapi, and Hybrid RRF (k=60)
│   ├── metrics         Recall@K, MRR@K, NDCG@K, Faithfulness, Citations
│   └── experiments     Factorial sweep runner & paired statistical testing
│
└── Research Workbench (React 19 Interactive UI)
    ├── experiment matrix   Combinatorial sweep configurator & scale estimation
    ├── Pareto analysis     Active control-signature validation for fair comparison
    ├── radar charting      Multi-axial trade-off plots across IR & generation metrics
    └── trace inspection    Claim-level context attribution & citation verification
```

---

## Research Contributions

1. **Controlled Evaluation Invariants:** Eliminates confounding factors by enforcing byte-exact coordinate preservation under Unicode NFKC and immutable configuration hashes.
2. **Systematic Paradigm Comparison:** Evaluates dense semantic retrieval, BM25 Okapi, and Hybrid RRF under identical evaluation corpora.
3. **Multilingual Transfer Evaluation:** Quantifies transfer penalties across $N=25$ matched, non-pseudoreplicated bilingual information units.
4. **Reproducible Experiment Orchestration:** Backed by cryptographic SHA-256 manifests, paired Wilcoxon signed-rank tests, and Holm-Bonferroni correction ($m=4$).
5. **Interactive Exploration Workbench:** Provides active Pareto frontier validation and granular claim attribution inspection.

---

## Experimental Baseline & Reference Findings

In the pre-registered reference study, RAGBench evaluated cross-lingual transfer between English and Bengali across $N=25$ matched bilingual information units (12 factorial conditions) using `BAAI/bge-m3` (1024-d, cosine similarity, fixed 200/20 chunking, $K=5$).

Hypothesis testing applied two-sided Wilcoxon signed-rank tests ($W, p_W$) as the primary non-parametric criterion, two-sided paired Student's $t$-tests ($t, p_t$), and Step-Down Holm-Bonferroni correction ($m=4$):

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
