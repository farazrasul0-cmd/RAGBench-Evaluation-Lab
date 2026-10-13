# RAGBench Reproducibility & Replication Guide

This guide provides end-to-end instructions for external researchers wishing to reproduce the experimental benchmarks, verify statistical calculations, or deploy the interactive research workbench in **RAGBench-Evaluation-Lab**.

---

## 1. System Requirements

* **Operating System:** Linux (Ubuntu 22.04+ recommended), macOS (13+), or Windows 11 with PowerShell.
* **Python:** Version 3.12 (strict typing enforced via `mypy`).
* **Node.js:** Version 22 LTS (with npm).
* **Docker:** Docker Engine 24+ and Docker Compose v2+ (optional, for containerized execution).

---

## 2. Environment Setup

### 2.1 Backend Installation
Clone the repository and install backend dependencies with development packages:

```bash
git checkout main
python3.12 -m venv backend/.venv
source backend/.venv/bin/activate   # On Windows: .\backend\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e "backend[dev]"
```

Verify backend installation:
```bash
ragbench --version
# Output: ragbench, version 1.0.0
```

### 2.2 Frontend Installation
```bash
cd frontend
npm ci || npm install
cd ..
```

---

## 3. Data Integrity Verification

Before running evaluations, verify the cryptographic integrity of the empirical baseline datasets:

```bash
# On Linux / macOS:
sha256sum data/experiments/g_multilingual_reference_studies.db
sha256sum data/experiments/g_multilingual_replication_bundle/statistical_summary.json

# On Windows (PowerShell):
Get-FileHash data/experiments/g_multilingual_reference_studies.db -Algorithm SHA256
Get-FileHash data/experiments/g_multilingual_replication_bundle/statistical_summary.json -Algorithm SHA256
```

Compare the printed digests with the reference values listed in [`docs/ARTIFACT_HASHES.md`](./ARTIFACT_HASHES.md).

---

## 4. Reproducing Empirical Research Reports

The RAGBench reporting engine extracts findings directly from the authoritative replication bundle and renders publication-ready LaTeX tables, whitepaper manuscripts, and Markdown summaries.

Generate all publication artifacts:
```bash
ragbench report generate --output-dir reports/
```

### Expected Generated Artifacts:
* `reports/table_rq3_multilingual.tex`: Publication-grade LaTeX table with `booktabs` formatting covering all 12 factorial transfer conditions.
* `reports/BENCHMARK_REPORT.md`: Markdown summary detailing empirical $\bar{\Delta}$, Cohen's $d_z$, and Holm-corrected significance values for hypotheses H1–H4.
* `reports/whitepaper.tex`: Standalone compilable academic paper manuscript.
* `reports/IEEEtran.cls`: Bundled IEEE Transactions LaTeX class file enabling immediate local compilation.

---

## 5. Running the Academic Research Workbench

### Option A: Docker Deployment (Recommended)
Launch both backend API and frontend workbench simultaneously:
```bash
docker compose up -d
```
* **Frontend UI:** Open [http://localhost:3000](http://localhost:3000)
* **Backend API Documentation:** Open [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)

To stop services:
```bash
docker compose down
```

### Option B: Local Development Services
In terminal 1 (Backend):
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

In terminal 2 (Frontend):
```bash
cd frontend
npm run dev
# Workbench available at http://localhost:5173
```

---

## 6. Running Quality Gates & Test Suites

To verify that all evaluation modules, statistical math, and frontend interfaces pass quality gates:

### Backend Tests & Static Analysis:
```bash
cd backend
pytest -q -m "not slow"
ruff check app tests
ruff format --check app tests
mypy app
```

### Frontend Tests & Type Checking:
```bash
cd frontend
npm run lint          # Oxlint static analysis
npm run typecheck     # Strict TypeScript compiler check
npm run test:run      # Vitest component and behavior suite
npm run build         # Vite production asset compilation
```
