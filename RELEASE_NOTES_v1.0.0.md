# RAGBench v1.0.0 Release Notes

**Release Version:** 1.0.0  
**Release Date:** 2026-10-13  
**Target Milestone:** Public Research Software Release Candidate  

RAGBench-Evaluation-Lab v1.0.0 establishes an end-to-end, scientifically controlled evaluation framework for Retrieval-Augmented Generation (RAG) pipelines.

---

## Highlights

### 1. Research Engine (CLI & Core)
* **Factorial Experiment Sweeps:** Evaluates combinatorial permutations across chunking geometry (fixed, sentence, character), dense embeddings (`BAAI/bge-m3`), lexical search (BM25 Okapi), neural rerankers, and retrieval depth ($K$).
* **Cross-Lingual Transfer Evaluation:** Standardized benchmark evaluating language transfer penalties across $N=25$ matched bilingual information units (English and Bengali).
* **Controlled Statistical Math:** Paired Wilcoxon signed-rank tests, paired Student's $t$-tests, Cohen's $d_z$, and Step-Down Holm-Bonferroni correction ($m=4$) to prevent pseudoreplication.
* **Automated Publication Reporting:** Dynamic generation of publication-ready LaTeX tables (`table_rq3_multilingual.tex`), Markdown summaries, and academic whitepapers with self-contained `IEEEtran.cls` support.

### 2. Academic Research Workbench (Frontend)
* **Active Pareto Frontier Validation:** Prevents invalid comparisons by validating that evaluated runs share identical canonical control signatures (chunk size, overlap, embedding model, population).
* **Bounded Radar Metric Charting:** Visualizes trade-offs between Recall@K, Precision@K, MRR@K, NDCG@K, and Grounded Faithfulness.
* **Neutral Trace Inspector:** Distinguishes between unsupported claims and unresolvable citation provenance without subjective bias.

### 3. Production Infrastructure & Quality
* **Docker Multi-Stage Deployment:** Hardened non-root containers with Nginx reverse proxy and native healthcheck endpoints.
* **CI/CD Quality Pipeline:** Automated GitHub Actions matrix testing Python 3.12, Node 22, Ruff, Mypy strict, Pytest, Oxlint, and Vitest.
* **Cryptographic Immutability:** All empirical reference studies and replication bundles verified via SHA-256 manifests (`docs/ARTIFACT_HASHES.md`).

---

## Verification Summary

* **Backend Tests:** 268/268 passing (`pytest -q -m "not slow"`)
* **Frontend Tests:** 23/23 passing (`vitest run`)
* **Type Safety:** 100% strict Mypy and TypeScript check passing
* **Code Formatting:** 100% clean under Ruff and Oxlint
