# Phased Implementation Roadmap — RAGBench (Scratch to Production)

**Document Version:** 2.0.0-PROD  
**Target Repository:** `RAGBench-Evaluation-Lab`  
**Execution Model:** Phased Milestone Delivery  
**Strict Quality Rule:** No phase shall begin until all verification gates of the preceding phase pass with zero errors, 100% type coverage under `mypy --strict`, passing lint checks, and passing unit/integration tests.

---

## Architectural File & Module Layout

Every implementation task maps to an exact location in this production hierarchy:

```
RAGBench-Evaluation-Lab/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py                 # Core routing registry
│   │   │   ├── v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── experiments.py        # Experiment matrix CRUD & dispatch
│   │   │   │   ├── datasets.py           # Dataset management & corpus ingest
│   │   │   │   ├── runs.py               # Evaluation runs, metrics, traces
│   │   │   │   └── health.py             # Liveness and readiness endpoints
│   │   │   └── dependencies.py           # DB session, Celery client, auth injection
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py                 # Pydantic-settings configuration
│   │   │   ├── logging.py                # Structured JSON logging & tracing
│   │   │   └── exceptions.py             # Unified domain exceptions & HTTP handlers
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── session.py                # SQLAlchemy async session factory
│   │   │   ├── base.py                   # Declarative base & metadata
│   │   │   └── repository.py             # CRUD repositories for experiments & traces
│   │   ├── engine/                       # Pure functional RAG & Metric logic
│   │   │   ├── __init__.py
│   │   │   ├── parsers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py               # BaseDocumentParser abstract interface
│   │   │   │   ├── markdown.py           # Markdown front-matter & section parser
│   │   │   │   ├── pdf.py                # PDF page & layout parser (pypdf)
│   │   │   │   └── text.py               # Plain text & code parser
│   │   │   ├── chunkers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py               # BaseChunker abstract interface
│   │   │   │   ├── fixed_token.py        # Slotted token windows with overlap
│   │   │   │   ├── recursive.py          # Hierarchical multi-separator chunker
│   │   │   │   ├── sentence.py           # Sentence boundary-aware chunker
│   │   │   │   └── semantic.py           # Adjacent embedding distance chunker
│   │   │   ├── embeddings/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py               # BaseEmbeddingProvider abstract interface
│   │   │   │   ├── fastembed_provider.py # ONNX CPU BAAI/bge embeddings
│   │   │   │   ├── sentence_trans.py     # PyTorch sentence-transformers
│   │   │   │   └── cloud_provider.py     # LiteLLM cloud embedding adapter
│   │   │   ├── retrievers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py               # BaseRetriever abstract interface
│   │   │   │   ├── dense.py              # Qdrant dense vector retriever
│   │   │   │   ├── bm25.py               # Language-aware Rank-BM25 retriever
│   │   │   │   └── hybrid.py             # RRF & Relative Score Fusion
│   │   │   ├── rerankers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py               # BaseReranker abstract interface
│   │   │   │   ├── cross_encoder.py      # MS-MARCO MiniLM cross-encoder
│   │   │   │   └── flashrank.py          # FlashRank ultra-lightweight reranker
│   │   │   ├── query_transforms/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── hyde.py               # Hypothetical Document Embeddings
│   │   │   │   ├── multi_query.py        # Sub-query expander & reciprocal merge
│   │   │   │   └── step_back.py          # High-level conceptual abstraction
│   │   │   ├── context/
│   │   │   │   ├── __init__.py
│   │   │   │   └── builder.py            # Sliding budget & Lost-in-the-Middle
│   │   │   └── metrics/
│   │   │       ├── __init__.py
│   │   │       ├── ir.py                 # Recall@K, Precision@K, MRR@K, NDCG@K
│   │   │       ├── generation.py         # Proposition decomposition, NLI Entailment
│   │   │       └── citation.py           # Citation precision, recall, source mapping
│   │   ├── models/                       # SQLAlchemy ORM models
│   │   │   ├── __init__.py
│   │   │   ├── experiment.py             # Experiment definitions & matrix states
│   │   │   └── run.py                    # Evaluation runs, query traces, metrics
│   │   ├── schemas/                      # Pydantic data contracts
│   │   │   ├── __init__.py
│   │   │   ├── document.py               # Document & Section schemas
│   │   │   ├── chunk.py                  # Chunk schema with sha256 hashes
│   │   │   ├── experiment.py             # YAML experiment matrix schema
│   │   │   ├── evaluation.py             # Ground truth eval_qa.jsonl schema
│   │   │   └── result.py                 # Run result data contracts (result.json)
│   │   ├── services/                     # Stateful orchestrators
│   │   │   ├── __init__.py
│   │   │   ├── vector_store.py           # Qdrant client adapter & collection manager
│   │   │   ├── orchestrator.py           # Experiment matrix Cartesian sweeper
│   │   │   ├── worker.py                 # Celery distributed query task workers
│   │   │   └── report_generator.py       # LaTeX & Markdown report exporter
│   │   └── main.py                       # FastAPI application entry point
│   ├── tests/
│   │   ├── unit/                         # Fast isolated functional tests
│   │   ├── integration/                  # End-to-end component tests
│   │   ├── fixtures/                     # Synthetic test corpora & golden answers
│   │   └── benchmarks/                   # Latency & throughput stress tests
│   └── pyproject.toml                    # Poetry/uv/pip, ruff, mypy, pytest config
├── frontend/
│   ├── src/
│   │   ├── api/                          # Typed OpenAPI client functions
│   │   ├── components/                   # UI components (Radars, Tables, Traces)
│   │   ├── hooks/                        # Custom React query & SSE hooks
│   │   ├── types/                        # TypeScript domain interfaces
│   │   ├── views/                        # Main dashboard views
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── datasets/                             # Benchmark evaluation corpora
│   ├── software_eng/
│   ├── academic_research/
│   ├── general/
│   └── multilingual/
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.worker
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── .gitignore
├── README.md
├── SYSTEM_ARCHITECTURE.md
├── PHASED_IMPLEMENTATION_ROADMAP.md
├── EVALUATION_METHODOLOGY_RQ.md
└── AGENT_EXECUTION_PROTOCOL.md
```

---

## Phase 0: Foundations, Toolchain & Directory Scaffold
- [x] **Initialize complete repository structure**:
  - `backend/app/{api, core, engine, models, schemas, services, db}`
  - `backend/tests/{unit, integration, fixtures}`
  - `frontend/{src/{components, views, hooks, api, types}}`
  - `datasets/{software_eng, academic_research, general, multilingual}`
- [x] **Configure Python environment**:
  - Python 3.12+ virtualenv established via `uv` in `backend/.venv`.
  - Installed core runtime & tooling: `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `httpx`, `pytest`, `ruff`, `mypy`.
- [x] **Establish pre-commit linting & static analysis**:
  - `ruff` configured in `pyproject.toml` (rules: `E`, `W`, `F`, `I`, `B`, `C4`, `UP`, `SIM`, line-length 100).
  - `mypy` configured with `strict = true`, `disallow_untyped_defs = true`, and Pydantic plugin.
- [x] **Set up frontend toolchain**:
  - Node 24+, Vite 8+, React 19, TypeScript 6+, Tailwind CSS (`@tailwindcss/vite`), Lucide icons, Vitest.
  - Scripts: `"typecheck": "tsc --noEmit"`, `"test:run": "vitest run"`, `"build": "tsc -b && vite build"`.
- [x] **Verification Gate 0 (Passed)**:
  - `ruff check app tests` (0 errors).
  - `ruff format --check app tests` (16 files clean).
  - `mypy app` (100% type coverage, 0 errors under `--strict`).
  - `pytest tests -v` (3/3 unit tests passing in 0.66s).
  - `npm run typecheck`, `npm run test:run`, and `npm run build` pass cleanly in 1.15s.

---

## Phase A: Document Ingestion, Parsing & Deterministic Chunking Engine

### Goal
Build an industrial-strength document ingestion engine capable of parsing multiple file formats, extracting rich front-matter metadata, and segmenting documents using 4 distinct, parameterizable chunking strategies with collision-free cryptographic hashing.

### Detailed Engineering Tasks
- [x] **Data Schema Definition (`backend/app/schemas/document.py`, `chunk.py`)**:
  - `RawDocument`: `doc_id`, `filename`, `content`, `metadata: dict[str, Any]`, `domain: str`, `file_type: str`, `checksum: str`.
  - `DocumentChunk`: `chunk_id`, `doc_id`, `chunk_index`, `content`, `token_count`, `char_count`, `start_char`, `end_char`, `strategy: str`, `metadata: dict[str, Any]`.
  - Enforce deterministic hashing:
    $$\text{chunk\_id} = \text{sha256}(\text{doc\_id} + \text{str}(\text{chunk\_index}) + \text{content\_normalized})$$
- [x] **Document Parsers (`backend/app/engine/parsers/`)**:
  - `BaseDocumentParser`: Abstract base class with async `parse(file_path: Path) -> RawDocument`.
  - `MarkdownParser`: Extracts YAML front-matter, header hierarchies (`#`, `##`, `###`), tables, and code fences. Preserves header path context for each sub-section.
  - `PDFParser`: Page-by-page extraction via `pypdf`, tracking page numbers and bounding offsets.
  - `PlainTextParser`: Clean Unicode normalization (NFKC), strip non-printable characters.
- [x] **Deterministic Chunkers (`backend/app/engine/chunkers/`)**:
  - `BaseChunker`: Abstract interface `chunk(document: RawDocument, **params) -> list[DocumentChunk]`.
  - `FixedTokenChunker`: Slotted token windows with configurable `chunk_size` (e.g., 256, 512, 1024) and `chunk_overlap` (e.g., 32, 64) using `tiktoken` (`cl100k_base` / `o200k_base`).
  - `RecursiveCharacterChunker`: Hierarchical splitting on separators `["\n\n", "\n", ". ", " ", ""]` ensuring chunks do not break mid-sentence unless forced by token limits.
  - `SentenceBoundaryChunker`: Sentence-aware boundary extraction using regex and language boundary rules (supporting English and Bengali full stops: `.` and `।`).
  - `SemanticSimilarityChunker`: Computes cosine distance between adjacent sliding sentences; inserts chunk boundary when semantic distance exceeds threshold $\tau$ (e.g., $\tau \ge 0.82$).
- [x] **Automated Tests (`backend/tests/unit/test_chunkers.py`, `test_parsers.py`)**:
  - Test zero data loss: concatenated text of non-overlapping chunks preserves 100% of non-whitespace characters.
  - Test boundary integrity: verify code blocks and Markdown tables are not split across chunks when within size budget.
  - Test deterministic reproducibility: chunk IDs and boundary indices match across repeated runs.
- [x] **Verification Gate A**:
  - `ruff check app tests` and `mypy app` pass with 0 errors.
  - Unit tests verify 0 chunk loss and boundary validity across synthetic Markdown, PDF, and code files.
  - Benchmark test verifying $\ge 1,000$ pages processed in $< 5$ seconds locally.

---

## Phase B: Dual-Mode Embedding & Vector Store Storage

### Goal
Implement a modular embedding abstraction supporting CPU-optimized local models (FastEmbed ONNX), GPU/PyTorch sentence-transformers, and cloud API embeddings (LiteLLM), backed by a production Qdrant vector store adapter with collection hashing and in-memory CI fallback.

### Detailed Engineering Tasks
- [x] **Embedding Provider Abstraction (`backend/app/engine/embeddings/`)**:
  - `BaseEmbeddingProvider`: Abstract interface with `embed_texts(list[str]) -> list[list[float]]` and `embed_query(str) -> list[float]`.
  - `FastEmbedProvider`: Local CPU-optimized ONNX runtime embeddings (default: `BAAI/bge-small-en-v1.5` [384d], `BAAI/bge-base-en-v1.5` [768d]). Fully functional offline with zero cloud cost.
  - `SentenceTransformersProvider`: PyTorch CUDA/MPS/CPU adapter supporting any HuggingFace dense embedding model.
  - `CloudEmbeddingProvider`: LiteLLM integration for OpenAI (`text-embedding-3-small`, `large`), Voyage AI, and Cohere Embed v3 with automatic retry and rate-limiting.
  - Dimensionality validation: Validate that vector output dimensions match model configuration contracts.
- [x] **Vector Store Adapter (`backend/app/services/vector_store.py`)**:
  - `VectorStoreAdapter`: Unified interface for vector operations.
  - Qdrant integration via `qdrant-client`:
    - Strict collection naming contract:
      $$\text{collection\_name} = \text{ragbench\_}\{\text{dataset\_id}\}\_\{\text{chunking\_hash}\}\_\{\text{embedding\_model\_hash}\}$$
    - Vector index parameters: HNSW cosine distance, $M=16$, $ef\_construct=100$.
    - Payload schema: `doc_id`, `chunk_id`, `chunk_index`, `content`, `strategy`, `token_count`, `domain`.
    - Payload index: Keyword indexes on `doc_id`, `domain`, and `strategy` for fast filtered retrieval.
  - `InMemoryVectorStore`: Numpy-based cosine similarity vector store for fast, standalone offline unit testing and CI without requiring a running Qdrant daemon.
- [x] **Automated Tests (`backend/tests/unit/test_embeddings.py`, `test_vector_store.py`)**:
  - Vector cosine ranking unit test: Assert distance calculation on synthetic orthogonal vectors is deterministic.
  - Collection creation and batch points upsertion test ($> 500$ points upserted in $< 1$s).
- [x] **Verification Gate B**:
  - `ruff check app tests` and `mypy app` pass with 0 errors.
  - Vector storage and retrieval tests pass on both `InMemoryVectorStore` and `QdrantClient`.

---

## Phase C: Retrieval Engine Matrix (Dense, BM25 & Hybrid Fusion)

### Goal
Build the comparative retrieval engine that implements Dense vector search, Lexical BM25 keyword search, reciprocal rank fusion (RRF), relative score normalization (RSN), and neural cross-encoder reranking.

### Detailed Engineering Tasks
- [ ] **Lexical Retriever (`backend/app/engine/retrievers/bm25.py`)**:
  - In-process `BM25Retriever` using `rank-bm25` (Okapi BM25 with $k_1=1.5, b=0.75$).
  - Tokenization pipeline: lowercasing, punctuation stripping, language-specific stopword removal (supporting English and Bengali).
  - Corpus indexing: Cache BM25 inverted index to disk (`.bm25.pkl`) alongside chunk metadata for fast reloading.
- [ ] **Dense Vector Retriever (`backend/app/engine/retrievers/dense.py`)**:
  - Query vector generation via configured `EmbeddingProvider`.
  - Qdrant ANN search with configurable `top_k` (e.g., $K=20$) and optional metadata filtering.
- [ ] **Score Fusion Layer (`backend/app/engine/retrievers/hybrid.py`)**:
  - **Reciprocal Rank Fusion (RRF)**:
    $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
    where $M = \{\text{dense}, \text{bm25}\}$, rank $r_m(d) \in [1, K]$, and default smoothing constant $k=60$.
  - **Relative Score Normalization (RSN)**:
    $$s_{\text{norm}}(d) = \frac{s(d) - \min_j s(d_j)}{\max_j s(d_j) - \min_j s(d_j) + \epsilon}$$
    Weighted linear combination: $S_{\text{final}}(d) = \alpha \cdot s_{\text{dense\_norm}}(d) + (1 - \alpha) \cdot s_{\text{bm25\_norm}}(d)$.
- [ ] **Neural Cross-Encoder Reranker (`backend/app/engine/rerankers/`)**:
  - `BaseReranker`: `rerank(query: str, candidates: list[DocumentChunk], top_n: int) -> list[RankedChunk]`.
  - `CrossEncoderReranker`: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to compute query-chunk cross-attention relevance logits.
  - `FlashRankReranker`: Lightweight CPU-based ONNX cross-encoder fallback (`ms-marco-TinyBERT-L-2-v2`).
- [ ] **Automated Tests (`backend/tests/unit/test_retrieval.py`, `test_fusion.py`)**:
  - Mathematical verification of RRF scores matching manual calculation on a fixed candidate ranking fixture.
  - Deduplication test: Assert no document chunk appears twice in fused results.
- [ ] **Verification Gate C**:
  - `ruff check app tests` and `mypy app` pass with 0 errors.
  - Unit tests prove RRF and RSN mathematically conform to `SYSTEM_ARCHITECTURE.md`.
  - Latency benchmarks prove hybrid fusion adds $< 10$ms overhead over single retrieval.

---

## Phase D: Query Transformations & Context Formulation

### Goal
Implement advanced query pre-processing strategies (HyDE, Multi-Query, Step-Back) and context window packing logic that mitigates the "Lost-in-the-Middle" phenomenon while strictly respecting token budgets.

### Detailed Engineering Tasks
- [ ] **Query Transformation Pipeline (`backend/app/engine/query_transforms/`)**:
  - `BaseQueryTransformer`: Abstract interface `transform(query: str) -> list[str]`.
  - `HyDETransformer` (Hypothetical Document Embeddings):
    - Prompt LLM: *"Generate a hypothetical technical passage that answers the question: {query}"*.
    - Embed generated hypothetical passage and retrieve based on hypothetical vector representation.
  - `MultiQueryExpander`:
    - Generate 3 distinct query formulations capturing synonyms, technical terms, and alternative perspectives.
    - Retrieve top-$K$ for each sub-query, then merge and rank via RRF.
  - `StepBackPrompting`:
    - Extract a high-level conceptual question from the user query to retrieve background concepts alongside specific facts.
- [ ] **Context Builder (`backend/app/engine/context/builder.py`)**:
  - Token Budget Enforcement: Enforce sliding token ceilings (e.g., 2,048 or 4,096 tokens) using `tiktoken`.
  - Re-ordering strategies:
    - Standard: Retain retriever rank order $1, 2, \dots, K$.
    - **Lost-in-the-Middle Mitigation**: Alternating placement so highest scoring chunks are at the beginning and end of the context window:
      $$[d_1, d_3, d_5, \dots, d_6, d_4, d_2]$$
  - Citation Header Injection: Format each context chunk with explicit identifiers:
    ```markdown
    [Source 1] (doc: paper_mamba.pdf, chunk: chunk_048)
    Content snippet...
    ```
- [ ] **Automated Tests (`backend/tests/unit/test_query_transforms.py`, `test_context_builder.py`)**:
  - Context budget guard: Assert formatted prompt never exceeds maximum allowed token limit by even 1 token.
  - Lost-in-the-Middle assertion: Verify chunk ordering indices match expected placement.
- [ ] **Verification Gate D**:
  - `ruff check app tests` and `mypy app` pass with 0 errors.
  - Golden query transformation schema tests pass.

---

## Phase E: Automated Metric Calculation Engine (IR & Generation)

### Goal
Implement the core mathematical evaluation engine calculating both Information Retrieval (IR) metrics and Generation/Factuality/Citation metrics in strict conformance with Section 2 of `SYSTEM_ARCHITECTURE.md`.

### Detailed Engineering Tasks
- [ ] **IR Metric Evaluator (`backend/app/engine/metrics/ir.py`)**:
  - Given query ground-truth passages $R_q$ and retrieved list $\hat{R}_{q, K}$:
    - **Recall@K**: $\frac{|\hat{R}_{q, K} \cap R_q|}{|R_q|}$
    - **Precision@K**: $\frac{|\hat{R}_{q, K} \cap R_q|}{K}$
    - **MRR@K** (Mean Reciprocal Rank): $\frac{1}{\text{rank}}$ of first hit $\le K$.
    - **NDCG@K** (Normalized Discounted Cumulative Gain): with binary and graded relevance $r_i \in \{0, 1\}$.
    - **Hit@K**: $1$ if $|\hat{R}_{q, K} \cap R_q| \ge 1$ else $0$.
- [ ] **Generation & Faithfulness Evaluator (`backend/app/engine/metrics/generation.py`)**:
  - **Proposition Decomposer**: Splits generated answer $A$ into atomic statements $S_A = \{s_1, \dots, s_m\}$ (each expressing a single factual claim).
  - **Entailment Classifier**: Evaluates whether context $C \vdash s_i$ using local NLI model (`cross-encoder/nli-deberta-v3-small`) or zero-shot LLM-as-a-Judge.
  - **Faithfulness Score**: $\frac{|\{s \in S_A \mid C \vdash s\}|}{|S_A|}$.
  - **Hallucination Rate**: $1.0 - \text{Faithfulness}$.
  - **Answer Relevance**: LLM generates synthetic queries from answer $A$; computes mean cosine similarity $\frac{1}{N} \sum \cos(\mathbf{e}(A), \mathbf{e}(q_{\text{gen}}^{(i)}))$.
- [ ] **Citation Precision & Recall Evaluator (`backend/app/engine/metrics/citation.py`)**:
  - Regex extraction of `[Source N]` or `[N]` citations from generated text.
  - Verify if sentence preceding `[Source N]` is entailed by chunk $N$.
  - Citation Precision = $\frac{\text{Entailing citations}}{\text{Total citations made}}$.
  - Citation Recall = $\frac{\text{Required ground truth facts cited}}{\text{Total factual claims in generated answer}}$.
- [ ] **Automated Tests (`backend/tests/unit/test_metrics.py`)**:
  - Synthetic claim-evidence test: 10 pre-labeled claim-evidence pairs asserting exact faithfulness calculation (0% on complete hallucination, 100% on perfect entailment).
  - Mathematical verification of NDCG@K against manual pen-and-paper calculation fixtures.
- [ ] **Verification Gate E**:
  - `ruff check app tests` and `mypy app` pass with 0 errors.
  - All metric functions pass validation with zero floating-point division by zero exceptions on edge cases (empty results, 0 citations).

---

## Phase F: Experiment Orchestrator & CLI Runner

### Goal
Build the experiment execution orchestrator capable of reading `experiment.yaml`, generating Cartesian parameter sweeps, dispatching parallel evaluation jobs via Celery/async workers, and persisting results in SQLite/PostgreSQL with a CLI runner.

### Detailed Engineering Tasks
- [ ] **Configuration Specification Parser (`backend/app/schemas/experiment.py`)**:
  - Pydantic models validating `experiment.yaml` schema:
    - Dataset configuration (id, corpus path, eval QA path).
    - Pipeline parameter sweeps: chunking strategies $\times$ embedding models $\times$ retrieval modes $\times$ rerankers $\times$ LLMs.
    - Evaluation metrics list.
- [ ] **Cartesian Matrix Sweeper (`backend/app/services/orchestrator.py`)**:
  - Computes Cartesian product of pipeline configurations.
  - Hashes each configuration to produce deterministic `configuration_hash`.
  - Skips already executed configurations if caching is enabled.
- [ ] **Task Distribution & Persistence (`backend/app/services/worker.py`, `db/repository.py`)**:
  - Asynchronous task distribution supporting in-process `asyncio` worker pool for local runs, with Celery queue bindings (`indexer_lane`, `retrieval_lane`, `eval_lane`) for distributed execution.
  - SQLite with WAL mode enabled (`sqlite+aiosqlite:///ragbench.db`) for zero-configuration local runs; PostgreSQL 16 compatible schema.
  - Tables: `experiments`, `runs`, `query_traces` (storing prompt, context, response, metrics, latency per query).
- [ ] **CLI Tool (`backend/app/cli.py`)**:
  - `ragbench run <config.yaml> [--parallel N] [--dry-run]`
  - `ragbench list-runs`
  - `ragbench compare-runs <run_id_1> <run_id_2>`
  - `ragbench export-run <run_id> --format [json|csv|latex]`
- [ ] **Automated Tests (`backend/tests/integration/test_orchestrator.py`)**:
  - End-to-end matrix sweep on a 5-question synthetic corpus executing 4 configurations in $< 15$ seconds.
- [ ] **Verification Gate F**:
  - `ruff check app tests` and `mypy app` pass with 0 errors.
  - Full end-to-end CLI execution executes cleanly and generates valid `result.json`.

---

## Phase G: Multilingual Evaluation Extension (Bengali & English)

### Goal
Extend the evaluation platform to benchmark cross-lingual RAG pipelines, quantifying the language transfer penalty between English and Bengali queries, chunks, and answers.

### Detailed Engineering Tasks
- [ ] **Bilingual Corpus & QA Loader (`datasets/multilingual/`)**:
  - Load parallel English–Bengali bilingual evaluation dataset (government & technical documentation).
  - Support Bengali character normalization and sentence tokenization (delimited by `।`, `?`, `!`).
- [ ] **Multilingual Embedding Integration**:
  - FastEmbed/Sentence-Transformers integration of `multilingual-e5-small` and `BAAI/bge-m3`.
  - Validate UTF-8 character preservation across chunking and vector storage without encoding corruption.
- [ ] **Cross-Lingual Evaluation Modes (`backend/app/engine/retrievers/cross_lingual.py`)**:
  - Mode 1: English Query $\rightarrow$ Bengali Chunks $\rightarrow$ English Answer.
  - Mode 2: Bengali Query $\rightarrow$ Bengali Chunks $\rightarrow$ Bengali Answer.
  - Mode 3: Bengali Query $\rightarrow$ English Chunks $\rightarrow$ Bengali Answer.
  - Compute Language Transfer Penalty:
    $$\Delta_{\text{lang}} = \text{Metric}_{\text{EN-EN}} - \text{Metric}_{\text{BN-BN}}$$
- [ ] **Automated Tests (`backend/tests/unit/test_multilingual.py`)**:
  - Bengali BM25 stopword filtering and tokenization tests.
  - Cross-lingual retrieval ranking test asserting non-zero Recall@5 on Bengali queries against Bengali documents.
- [ ] **Verification Gate G**:
  - `ruff check app tests` and `mypy app` pass with 0 errors.
  - Bilingual test suite passes with zero encoding errors.

---

## Phase H: Academic Research Workbench Frontend (React / Vite)

### Goal
Build the research-grade React 19 dashboard for visual comparison of parameter sweeps, interactive Pareto frontier exploration, radar charts, and per-query trace inspection.

### Detailed Engineering Tasks
- [ ] **Experiment Matrix Builder View (`frontend/src/views/MatrixBuilderView.tsx`)**:
  - Interactive multi-select UI for defining parameter sweeps (select chunk sizes, embeddings, retrievers, LLMs).
  - Real-time calculation of total configuration count ($N_1 \times N_2 \times \dots \times N_k$).
  - One-click YAML export and API dispatch (`POST /api/v1/experiments`).
- [ ] **Multi-Experiment Comparative Dashboard (`frontend/src/views/CompareDashboardView.tsx`)**:
  - **Pareto Frontier Plot**: Scatter visualizer plotting Recall@5 (X-axis) vs. Mean Latency ms (Y-axis) vs. Estimated Cost (bubble size). Highlights non-dominated optimal configurations.
  - **5-Axis Radar Chart**: Interactive polygon comparing 2–5 configurations across:
    1. Retrieval Recall@10
    2. Precision@5
    3. Generation Faithfulness
    4. Citation Accuracy
    5. Cost/Token Efficiency
- [ ] **Per-Question Trace Inspector (`frontend/src/views/TraceInspectorView.tsx`)**:
  - Side-by-side view for any selected query sample:
    - User query and ground truth answer.
    - Retrieved context chunks with relevance scores.
    - Generated answer with highlighted entailing spans (green) vs. flagged hallucinated propositions (red).
    - Inline citation click-through showing the exact source snippet.
- [ ] **Automated Frontend Tests (`frontend/src/**/*.test.tsx`)**:
  - Unit tests for Matrix Builder, Compare Dashboard, and Trace Inspector views.
- [ ] **Verification Gate H**:
  - `npm run typecheck` passes with 0 TypeScript errors.
  - `npm run test:run` passes with 100% passing tests.
  - `npm run build` completes production bundle in `dist/`.

---

## Phase M: Production Hardening, Release Sign-Off & Whitepaper Export

### Goal
Containerize the entire platform with Docker Compose, execute formal regression tests across all 6 Research Questions (RQ1–RQ6), generate academic publication-ready LaTeX tables, and release v1.0.0.

### Detailed Engineering Tasks
- [ ] **Automated Academic Report Generator (`backend/app/services/report_generator.py`)**:
  - Generates publication-ready LaTeX tables (`table_rq1_chunking.tex`, etc.) formatted with `booktabs`.
  - Statistical significance testing: Computes two-tailed paired Student's $t$-test $p$-values and 95% confidence intervals between configurations.
  - Exports executive summary markdown: `BENCHMARK_REPORT_RQ1_RQ6.md`.
- [ ] **Docker Orchestration (`docker/`)**:
  - Multi-stage `Dockerfile.backend` (distroless/slim Python 3.12).
  - Multi-stage `Dockerfile.frontend` (nginx serving Vite static assets).
  - `docker-compose.yml` orchestrating:
    - `ragbench-backend` (FastAPI)
    - `ragbench-worker` (Celery/Async)
    - `ragbench-qdrant` (Qdrant vector engine)
    - `ragbench-redis` (Redis task broker)
    - `ragbench-frontend` (React research UI)
- [ ] **CI/CD Pipeline (`.github/workflows/ci.yml`)**:
  - Runs sequentially: backend ruff, backend mypy, backend pytest, frontend typecheck, frontend vitest, frontend build.
- [ ] **Master Evaluation Run across RQ1–RQ6**:
  - Execute standard benchmark suites across all 4 datasets (`software_eng`, `academic_research`, `general`, `multilingual`).
  - Validate hypotheses for RQ1–RQ6 according to `EVALUATION_METHODOLOGY_RQ.md`.
- [ ] **Final Sign-Off**:
  - Zero open warnings or type errors.
  - Clean production release tagged as `v1.0.0`.
