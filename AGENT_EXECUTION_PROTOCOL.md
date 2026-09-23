# Agent Execution Protocol & Engineering Guardrails

**Governing Entity:** Autonomous Coding Agent (Antigravity)  
**Project:** RAGBench Evaluation Laboratory  
**Enforcement:** Mandatory Pre-Commit & Pre-Review Checks  

---

## 1. Absolute Golden Rules

1. **No Mock Data in Production Paths**: Evaluation metrics must be mathematically calculated from actual executions. If an external model is unavailable, run local offline models (FastEmbed, Ollama, or PyTorch CPU)—never generate random or fabricated metric numbers.
2. **Preserve Academic Rigor**: Every metric must conform to the mathematical definitions in `SYSTEM_ARCHITECTURE.md`.
3. **Zero Regressions**: No existing unit test may be broken or skipped to make a new test pass.
4. **Strict Typing & Linting**:
   - Python: `ruff check` and `ruff format` must pass without warnings. `mypy --strict` must report 0 errors.
   - Frontend: `tsc --noEmit` must pass with 0 errors.

---

## 2. Command-Line Toolchain Verification

Before and after every milestone, execute these commands sequentially:

```powershell
# 1. Backend Code Quality
cd backend
python -m ruff check app tests
python -m ruff format --check app tests
python -m mypy app

# 2. Backend Automated Test Suite
python -m pytest tests -v --durations=10

# 3. Frontend Static Typecheck & Unit Tests
cd ../frontend
npm run typecheck
npm run test:run
npm run build
```

---

## 3. Architecture Layer Boundaries

- `app/engine/`: Pure functional RAG logic (chunkers, tokenizers, mathematical metric calculators). Must have zero dependencies on FastAPI HTTP request/response objects.
- `app/services/`: Stateful orchestrators (managing database sessions, Celery dispatch, disk/Qdrant storage).
- `app/api/`: REST HTTP routing, request validation, authentication, and SSE streaming. Must not contain direct retrieval or embedding logic.
- `frontend/src/`: Research dashboard interface. Must communicate strictly via typed API clients matching backend OpenAPI schemas.

---

## 4. Standard Escalation Protocol

If an error occurs during execution:
1. **Root Cause Identification**: Locate the exact line, type mismatch, or import failure.
2. **Deterministic Fix**: Apply minimal targeted adjustments that resolve the root cause without altering surrounding architecture.
3. **Regression Confirmation**: Re-run the full test suite (`pytest -v` and `npm run test:run`) to prove no regressions were introduced.
