# RAGBench — Retrieval-Augmented Generation Evaluation Laboratory

**A Parameterizable Scientific Benchmarking Platform for RAG Pipelines**

[![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg)](https://github.com/farazrasul0-cmd/RAGBench-Evaluation-Lab)
[![Tests](https://img.shields.io/badge/backend%20tests-263%20passed-success.svg)](https://github.com/farazrasul0-cmd/RAGBench-Evaluation-Lab)
[![Frontend Tests](https://img.shields.io/badge/frontend%20tests-23%20passed-success.svg)](https://github.com/farazrasul0-cmd/RAGBench-Evaluation-Lab)
[![Python](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/react-19-61dafb.svg)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/typescript-strict-3178c6.svg)](https://www.typescriptlang.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

RAGBench is an enterprise and academic-grade evaluation laboratory that treats Retrieval-Augmented Generation (RAG) pipelines as parameterizable scientific experiments. Rather than serving as an ungrounded question-answering wrapper, RAGBench systematically benchmarks factorial variations across chunking geometry, embedding representations, retrieval topologies, neural rerankers, query transformations, and generation models.

---

## Architecture: Dual-Engine Design

```
                    RAGBench Platform Architecture
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        ▼                                                   ▼
   Research Engine                                   Research Workbench
(Phases A–G · eb3bb4d)                               (Phase H · c889df1)
        │                                                   │
  • Ingestion & Normalization                         • Combinatorial Matrix Builder
  • Dual Embedding Inference                          • Active Pareto Validation
  • Factorial Sweep Execution                         • Forensic Trace Inspector
  • Wilcoxon / Holm-Bonferroni FWER                   • Unresolved Provenance Semantics
  • SHA-256 Replication Bundles                       • 5-Axis Bounded IR Radar
        │                                                   │
        └─────────────────────────┬─────────────────────────┘
                                  ▼
             Fully Reproducible Empirical RAG Platform
```

---

## Locked Research Baselines & Empirical Evidence

- **Phase G Baseline ([`eb3bb4d`](https://github.com/farazrasul0-cmd/RAGBench-Evaluation-Lab/commit/eb3bb4d)):**
  Pre-registered RQ3 Multilingual Cross-Lingual Transfer Study ($N=25$ matched bilingual information units, 12 factorial conditions) comparing `BAAI/bge-m3` (1024-dim, real PyTorch inference), Okapi BM25, and Hybrid RRF ($k=60$):
  - **H1 (Supported):** English $->$ Bengali dense cross-lingual penalty ($\Delta = +0.8613$, Cohen's $d_z = +7.45$, $p < 0.001$).
  - **H2 (Not Supported):** Bengali $->$ English dense cross-lingual transfer exhibits a significant retrieval gain ($\Delta = -0.8382$, Cohen's $d_z = -7.11$).
  - **H3 (Not Supported):** Hybrid RRF does not attenuate the English $->$ Bengali dense penalty ($\Delta = -0.0989$, ns).
  - **H4 (Not Supported):** Hybrid RRF does not attenuate the Bengali $->$ English penalty ($\Delta = -0.7981$, $p < 0.001$).
- **Phase H Baseline ([`c889df1`](https://github.com/farazrasul0-cmd/RAGBench-Evaluation-Lab/commit/c889df1)):**
  Full-stack Academic Research Workbench frontend enforcing 13-field backend Canonical Control Signatures (`POST /api/v1/experiments/pareto/validate`), strictly suppressing Pareto frontier derivations across divergent experimental controls.

---

## Governance & Architecture Documentation

| Specification | Focus |
| :--- | :--- |
| **[`SYSTEM_ARCHITECTURE.md`](./SYSTEM_ARCHITECTURE.md)** | Mathematical formulation of IR/generation metrics, system architecture contracts, and persistence layer |
| **[`PHASED_IMPLEMENTATION_ROADMAP.md`](./PHASED_IMPLEMENTATION_ROADMAP.md)** | Step-by-step master lifecycle roadmap from Phase 0 to Phase M |
| **[`EVALUATION_METHODOLOGY_RQ.md`](./EVALUATION_METHODOLOGY_RQ.md)** | Formal experimental design, RQ1–RQ6 hypothesis pre-registration, and evaluation protocols |
| **[`AGENT_EXECUTION_PROTOCOL.md`](./AGENT_EXECUTION_PROTOCOL.md)** | Quality gates, linting/typing standards, and supervisory review protocols |

---

## Quickstart: Docker Compose Orchestration

Spin up the containerized Research Engine and Research Workbench:

```bash
docker compose up -d --build
```

- **Research Workbench UI:** `http://localhost:3000`
- **FastAPI REST API & Docs:** `http://localhost:8000/docs`
- **Health Check Endpoint:** `http://localhost:8000/api/v1/health`

---

## Quickstart: Local Developer Environment

### 1. Backend Setup & Test Suite

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Or: .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# Run full regression suite (260+ tests)
pytest -q -m "not slow"

# Static quality gates
ruff check app
mypy app
```

### 2. Frontend Research Workbench

```bash
cd frontend
npm install

# Run behavioral vitest suite (23 tests)
npm run test:run

# Lint & strict TypeScript check
npm run lint
npm run typecheck

# Production build
npm run build
```

---

## CLI Usage: Automated Academic Report & LaTeX Export

Generate peer-reviewed `booktabs` LaTeX tables, markdown reports, and complete whitepaper documents:

```bash
# Generate complete academic publication package into reports/
python -m app.cli.main report generate --output-dir reports/
```

Generated publication artifacts:
- `reports/table_rq3_multilingual.tex`: Publication-ready LaTeX table with Wilcoxon $W$, paired $t$, 95% CIs, and Cohen's $d_z$.
- `reports/BENCHMARK_REPORT.md`: Comprehensive executive markdown report.
- `reports/whitepaper.tex`: Standalone compilable academic whitepaper document.

---

## Phase Lifecycle Matrix

- 🟢 **Phase 0:** Foundations, Toolchain & Directory Scaffold
- 🟢 **Phase A:** Document Ingestion, Parsing & Deterministic Chunking Engine
- 🟢 **Phase B:** Dual-Mode Embedding & Vector Store Storage
- 🟢 **Phase C:** Retrieval Engine Matrix (Dense, BM25 & Hybrid Fusion)
- 🟢 **Phase D:** Query Transformations & Context Formulation
- 🟢 **Phase E:** Automated Metric Calculation Engine (IR & Generation)
- 🟢 **Phase F:** Experiment Orchestrator & CLI Runner
- 🟢 **Phase G:** Multilingual Evaluation Extension (`eb3bb4d` locked)
- 🟢 **Phase H:** Academic Research Workbench Frontend (`c889df1` locked)
- 🟢 **Phase M:** Production Hardening, Release Sign-Off & Whitepaper Export (v1.0.0)
