# Phased Implementation Roadmap — RAGBench

**Execution Model:** Phased Milestone Delivery  
**Strict Rule:** No phase shall begin until all verification gates of the preceding phase pass with zero errors, 100% type coverage, and passing tests.

---

## Phase 0: Foundations, Toolchain & Directory Scaffold
- [x] Initialize repository structure:
  - `backend/app/{api, core, engine, models, schemas, services, db}`
  - `backend/tests/{unit, integration, fixtures}`
  - `frontend/{src/{components, views, hooks, api, types}}`
  - `datasets/{software_eng, academic_research, general, multilingual}`
- [x] Configure Python environment: Python 3.12+, Poetry/pip requirements (`fastapi`, `pydantic-settings`, `qdrant-client`, `rank-bm25`, `fastembed`, `sentence-transformers`, `litellm`, `celery`, `redis`, `pytest`).
- [x] Establish pre-commit linting & static analysis: `ruff` (linting & formatting), `mypy --strict` (0 type errors allowed).
- [x] Set up frontend toolchain: Node 22+, Vite 6+, React 19, TypeScript, Tailwind CSS, Lucide icons, Vitest.
- [x] **Verification Gate 0**:
  - `ruff check .` passes with 0 errors.
  - `mypy` passes across 100% of scaffolded files.
  - `npm run typecheck` and `npm run build` pass cleanly.

---

## Phase A: Document Ingestion, Parsing & Deterministic Chunking Engine
- [ ] Implement `DocumentParser` supporting Markdown, PDF (`pypdf`), and Plain Text with front-matter extraction.
- [ ] Implement extensible `ChunkerStrategy` interface:
  - `FixedTokenChunker`: Slotted token windows with configurable overlap.
  - `RecursiveCharacterChunker`: Hierarchical separators (`\n\n`, `\n`, `. `, ` `).
  - `SentenceBoundaryChunker`: Sentence-aware boundary extraction using SpaCy/Regex.
  - `SemanticSimilarityChunker`: Adjacent embedding distance threshold chunking.
- [ ] Compute deterministic hashes for every chunk: `chunk_id = sha256(doc_id + chunk_index + content)`.
- [ ] **Verification Gate A**:
  - Unit tests verifying zero chunk loss, accurate boundary splits, and overlap preservation.
  - Benchmark test verifying $\ge 1,000$ pages processed in $< 5$ seconds locally.

---

## Phase B: Dual-Mode Embedding & Vector Store Storage
- [ ] Implement unified `EmbeddingProvider` interface with offline fallback:
  - `FastEmbedProvider` (CPU-optimized ONNX local embeddings, e.g., `bge-small-en-v1.5`).
  - `SentenceTransformersProvider` (PyTorch CUDA/MPS/CPU).
  - `CloudEmbeddingProvider` (LiteLLM-backed for OpenAI/Voyage/Cohere).
- [ ] Implement `VectorStoreAdapter` for Qdrant:
  - Dynamic collection creation based on schema hash.
  - Dense payload indexing with metadata filtering (`doc_id`, `chunk_size`, `tags`).
- [ ] Local in-memory vector store fallback for rapid offline testing.
- [ ] **Verification Gate B**:
  - Embedding dimensionality validation tests.
  - Cosine distance ranking unit tests asserting near-zero error on deterministic synthetic vectors.

---

## Phase C: Retrieval Engine Matrix (Dense, BM25 & Hybrid Fusion)
- [ ] Implement `BM25Retriever` using Rank-BM25 with language-aware tokenization and stop-word filtering.
- [ ] Implement `DenseRetriever` using Qdrant vector similarity search.
- [ ] Implement Score Fusion Layer:
  - **Reciprocal Rank Fusion (RRF)**:
    $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  - **Relative Score Normalization (RSN)**: Min-max normalization prior to weighted linear combination.
- [ ] Implement `CrossEncoderReranker`:
  - Neural reranking stage using `ms-marco-MiniLM-L-6-v2` or `FlashRank`.
  - Configurable `top_n` candidate filtering.
- [ ] **Verification Gate C**:
  - Unit tests verifying RRF scoring equations with known rank order fixtures.
  - Integration test evaluating retrieval latency and candidate deduplication.

---

## Phase D: Query Transformations & Context Formulation
- [ ] Implement Query Reformulation Pipeline:
  - `HyDE` (Hypothetical Document Embeddings): Generate hypothetical answer, embed it, and retrieve.
  - `MultiQueryExpander`: Generate 3 alternative sub-queries and perform reciprocal rank merging.
  - `StepBackPrompting`: Extract high-level conceptual questions.
- [ ] Implement Context Builder:
  - Hard token limit enforcement (sliding budget).
  - Document re-ordering strategies (e.g., "Lost-in-the-Middle" mitigation: placing highest-scoring chunks at the beginning and end of the context window).
- [ ] **Verification Gate D**:
  - Golden evaluation asserting that query transformation outputs match required Pydantic output schemas.
  - Token budget guard tests asserting no generation prompt ever exceeds the model's context ceiling.

---

## Phase E: Automated Metric Calculation Engine (IR & Generation)
- [ ] Implement Information Retrieval Metric Evaluator:
  - `Recall@K`, `Precision@K`, `MRR@K`, `NDCG@K`, `Hit@K`.
- [ ] Implement Generation & Faithfulness Evaluator:
  - Proposition extraction engine (deconstructing generated text into atomic statements).
  - NLI Entailment Classifier / LLM-as-a-Judge prompt chain verifying each statement against context.
  - Hallucination score calculator ($1.0 - \text{Faithfulness}$).
- [ ] Implement Citation Verifier:
  - Regex extraction of `[Source N]` citations.
  - Ground-truth passage mapping verifying that the cited chunk actually entails the preceding sentence.
- [ ] **Verification Gate E**:
  - Unit tests running on synthetic, pre-labeled claim-evidence pairs (asserting exact recall on known hallucinated answers).

---

## Phase F: Experiment Orchestrator & CLI Runner
- [ ] Build YAML-based experiment specification parser with Pydantic validation.
- [ ] Build `ExperimentRunner` capable of Cartesian-product matrix sweeps:
  - Example: 3 chunk sizes $\times$ 2 embedding models $\times$ 3 retrieval modes = 18 configurations.
- [ ] Integrate Celery task distribution for parallel query execution.
- [ ] Build CLI tool: `ragbench run config.yaml --parallel 4`.
- [ ] SQLite / PostgreSQL result persistence layer with run comparison queries.
- [ ] **Verification Gate F**:
  - End-to-end execution of a 4-configuration matrix sweep on a 50-question synthetic benchmark in $< 60$ seconds.

---

## Phase G: Multilingual Evaluation Extension (Bengali & English)
- [ ] Ingest parallel English–Bengali bilingual evaluation dataset.
- [ ] Integrate multilingual embedding models (`multilingual-e5-small`, `bge-m3`).
- [ ] Implement Cross-Lingual Evaluation Modes:
  - English Query $\rightarrow$ Bengali Chunks $\rightarrow$ English Answer.
  - Bengali Query $\rightarrow$ Bengali Chunks $\rightarrow$ Bengali Answer.
- [ ] Measure language transfer penalty:
  $$\Delta_{\text{lang}} = \text{Metric}_{\text{EN-EN}} - \text{Metric}_{\text{BN-BN}}$$
- [ ] **Verification Gate G**:
  - Bilingual test suite asserting that tokenization, BM25 stop words, and vector retrieval execute without encoding corruptions.

---

## Phase H: Academic Research Workbench Frontend (React / Vite)
- [ ] Implement Interactive Configuration Matrix Builder.
- [ ] Implement Multi-Experiment Comparative Dashboard:
  - **Pareto Frontier Plot**: Recall vs. Latency vs. Cost.
  - **Radar Chart**: 5-axis metric visualizer (Recall, Precision, Faithfulness, Citation Acc, Token Efficiency).
  - **Per-Question Inspector**: Side-by-side view showing retrieved chunks, highlighted entailing spans, and flagged hallucinations.
- [ ] **Verification Gate H**:
  - All Vitest frontend tests pass. Production build completes cleanly with zero TypeScript errors.

---

## Phase M: Production Hardening, Release Sign-Off & Whitepaper Export
- [ ] Automated Report Generator: One-click export of experimental findings into publication-ready LaTeX tables and Markdown summaries.
- [ ] Full end-to-end regression test suite execution (`pytest -v --cov`).
- [ ] Docker Compose orchestration (`ragbench-backend`, `ragbench-worker`, `ragbench-qdrant`, `ragbench-frontend`, `redis`).
- [ ] Final Sign-Off and Master Evaluation Run across all 6 Research Questions (RQ1–RQ6).
