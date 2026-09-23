# RAGBench-Evaluation-Lab — Technical Requirements Document (TRD)

**Version:** 1.0  
**Status:** Authoritative technical baseline  

---

## 1. System Architecture

```text
                                  React 19 + TypeScript (Frontend)
                                                 │
                                                 ▼
                                        FastAPI REST API
                                                 │
                   ┌─────────────────────────────┼─────────────────────────────┐
                   ▼                             ▼                             ▼
            Dataset Manager              Experiment Manager               Results API
                   │                             │
                   └─────────────────────────────┼─────────────────────────────┘
                                                 ▼
                                          Job / Task Queue (Async / Celery)
                                                 │
                                                 ▼
                                          Evaluation Engine
                                                 │
                                                 ▼
                                            RAG Pipeline
                                                 │
       ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
       ▼                                         ▼                                         ▼
   Chunking                                  Retrieval                                 Generation
       │                                  ┌──────┼──────┐                                  │
       │                                  ▼      ▼      ▼                                  │
       │                                Dense   BM25  Hybrid                               │
       │                                  │      │      │                                  │
       │                                  └──────┼──────┘                                  │
       │                                         ▼                                         │
       │                                 RerankedRetriever                                 │
       │                                         │                                         │
       │                                         ▼                                         │
       │                                    BaseReranker                                   │
       │                                         │                                         │
       └─────────────────────────────────────────┼─────────────────────────────────────────┘
                                                 ▼
                                        Query Transformation
                                                 │
                                                 ▼
                                        Context Formulation
                                                 │
                                                 ▼
                                             LLM Engine
                                                 │
                                                 ▼
                                      Metric Evaluation Engine
                                                 │
                                                 ▼
                                      PostgreSQL (Authoritative Store)
```

---

## 2. Technology Stack & Dependencies

### Backend Toolchain
- **Language Runtime:** Python 3.12+ (strictly managed via `uv`).
- **Web Framework:** FastAPI 0.115+ with ASGI server Uvicorn.
- **Data Validation & Settings:** Pydantic v2 (with `pydantic-settings` and `pydantic.mypy` plugin).
- **Relational Database:** PostgreSQL 16 (production) / SQLite WAL mode with `aiosqlite` (local test runner).
- **ORM & Migrations:** SQLAlchemy 2.0 (asyncio) + Alembic.
- **Vector Database:** Qdrant (client 1.12+, supporting in-memory `:memory:` and remote cluster).
- **Lexical Search:** `rank-bm25` (Okapi BM25 with multilingual tokenization).
- **Embeddings:** `fastembed` (CPU ONNX runtime) and `sentence-transformers`.
- **Rerankers:** `flashrank` (ONNX runtime) and `sentence-transformers` cross-encoders.
- **Context Tokenization:** `tiktoken` (`cl100k_base`).
- **Distributed Queuing:** Celery with Redis broker (for large Cartesian sweeps).
- **Linting & Typing:** `ruff` 0.9+ (line length 100) and `mypy` 1.14+ under `--strict`.

### Frontend Toolchain
- **Framework & Runtime:** React 19 + TypeScript.
- **Build Tool:** Vite 8.
- **Styling:** Tailwind CSS v4.
- **Icons:** Lucide React.
- **Testing:** Vitest 5.0 + `@testing-library/react`.

---

## 3. Layering Rules & Architectural Boundaries

The codebase strictly enforces the following downward dependency hierarchy:

```text
API Layer (app/api/)
   ↓
Service Layer (app/services/)
   ↓
Domain & Schemas (app/schemas/)
   ↓
Engine Matrix (app/engine/)
   ↓
Infrastructure & DB (app/db/, external drivers)
```

### Strict Architectural Invariants:
1. **The Engine must not depend on FastAPI request/response models.** Components in `app/engine/` must operate purely on domain schemas (`RawDocument`, `DocumentChunk`, `RankedChunk`, `PackedContext`) and primitive types.
2. **Retrievers must not depend on database models.** A retriever queries an index adapter, not the relational database.
3. **Retrievers and Rerankers are strictly decoupled.** A retriever produces candidates; a reranker re-scores candidates. Composition is handled via [`RerankedRetriever`](file:///D:/projects/RAGBench-Evaluation-Lab/backend/app/engine/retrievers/reranked.py).
4. **Query Transformers are decoupled from retrievers.** A transformer generates query variations; the execution orchestrator dispatches queries to retrievers.
5. **UI components must not contain experiment execution logic.** All sweeps, evaluations, and metrics calculations occur server-side.

---

## 4. Core Engine Interfaces

### 4.1 Document Ingestion & Chunking
```python
class BaseParser(ABC):
    @abstractmethod
    def parse(self, content: bytes, filename: str) -> RawDocument: ...

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: RawDocument) -> list[DocumentChunk]: ...
```
- **Deterministic Chunk ID Contract:**
  $$	ext{chunk\_id} = 	ext{SHA256}(f"\{	ext{doc\_id}\}:\{	ext{chunk\_index}\}:\{	ext{NFKC}(	ext{content})\}")$$

### 4.2 Embedding Providers
```python
class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...
```

### 4.3 Vector Store Adapter
```python
class VectorStoreAdapter(ABC):
    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine") -> None: ...

    @abstractmethod
    def upsert_chunks(self, collection_name: str, chunks: list[DocumentChunk], vectors: list[list[float]]) -> int: ...

    @abstractmethod
    def search(self, collection_name: str, query_vector: list[float], top_k: int = 10, filter_metadata: dict[str, Any] | None = None) -> list[tuple[DocumentChunk, float]]: ...
```

### 4.4 Retrieval Engine Matrix
```python
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10, filter_metadata: dict[str, Any] | None = None, **kwargs: Any) -> list[tuple[DocumentChunk, float]]: ...
```
Every retriever outputs a homogeneous `list[tuple[DocumentChunk, float]]` sorted descending by score, with ties resolved deterministically by ascending `chunk_id`.

### 4.5 Passage Rerankers
```python
class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[DocumentChunk], top_n: int = 5, **kwargs: Any) -> list[tuple[DocumentChunk, float]]: ...

    def rerank_ranked(self, query: str, candidates: list[DocumentChunk], top_n: int = 5, **kwargs: Any) -> list[RankedChunk]: ...
```

### 4.6 Query Transformation
```python
class BaseQueryTransformer(ABC):
    @property
    @abstractmethod
    def strategy_name(self) -> str: ...

    @abstractmethod
    def transform(self, query: str, **kwargs: Any) -> QueryTransformationResult: ...
```

### 4.7 Context Formulation
```python
class ContextBuilder:
    def build(self, candidates: list[tuple[DocumentChunk, float]]) -> PackedContext: ...
```

### 4.8 Evaluation Engine
```python
def evaluate_ir_metrics(retrieved: Sequence[str], ground_truth: Sequence[str] | set[str], k: int = 10) -> IRMetricsResult: ...
def evaluate_faithfulness(answer: str, context: str, classifier: BaseEntailmentClassifier) -> FaithfulnessResult: ...
def evaluate_citations(answer: str, context_sources: dict[int, str], classifier: BaseEntailmentClassifier) -> CitationMetricsResult: ...
```

---

## 5. Experiment Execution & Immutability

1. **Configuration Specification:** An experiment is defined by a validated `experiment.yaml` file parsed into strongly typed Pydantic models.
2. **Configuration Hash:** A canonical JSON representation of all hyperparameters is hashed (SHA-256) to produce `configuration_hash`.
3. **Immutability:** Once an experiment is executed (`RUNNING`), its configuration cannot be edited. Modifying hyperparameters generates a distinct experiment or a new run.
4. **Execution Tracing:** Every evaluation query executed records a complete trace in `QueryRun`:
   - `TransformedQuery`: exact queries produced by HyDE / Multi-Query.
   - `RetrievedChunk`: raw retrieval ranking and scores before reranking.
   - `RerankedChunk`: reranked positions and cross-encoder logits.
   - `PackedContext`: the exact context text formatted and fed to the LLM.
   - `GenerationResult`: answer text, latency, token consumption.
   - `MetricResult`: individual query-level scores.

---

## 6. Background Jobs & Observability

- Ingestion, embedding generation, parameter sweeps, and evaluations execute asynchronously.
- Distributed queuing is facilitated via Celery lanes:
  - `indexer_lane`: heavy embedding generation and vector upserts.
  - `retrieval_lane`: multi-channel retrieval and reranking execution.
  - `eval_lane`: LLM-as-a-judge and NLI faithfulness scoring.
- Correlation IDs: every log entry and database trace preserves `experiment_id`, `run_id`, and `query_id`.

---

## 7. Testing Strategy

1. **Unit Tests:** Verify discrete algorithms in isolation (parsers, chunk boundaries, BM25 scoring, mathematical formulas).
2. **Contract Tests:** Verify interchangeable components obey identical interface contracts (e.g. all 6 retrieval topologies output `list[tuple[DocumentChunk, float]]`).
3. **Integration Tests:** Verify multi-stage pipelines (ingest $	o$ chunk $	o$ embed $	o$ index $	o$ retrieve $	o$ pack).
4. **Reproducibility Tests:** Assert that running identical configurations repeatedly produces identical rankings and scores to $10^{-9}$ tolerance.
5. **Frontend Tests:** Verify UI component rendering, pipeline builder configuration graph, comparison views, and error states using Vitest.

---

## 8. Definition of Done

A development phase is complete only when:
- [x] Implementation satisfies functional and non-functional requirements.
- [x] All unit, contract, and integration tests pass (100%).
- [x] Code passes `ruff check app tests` with 0 errors.
- [x] Code passes `ruff format --check app tests` cleanly.
- [x] Code passes `mypy app tests --strict` with 0 errors.
- [x] Frontend passes `tsc --noEmit`, vitest unit tests, and production build.
- [x] Architectural boundaries and component decoupling are strictly preserved.
- [x] Working tree is clean with a descriptive Git commit.
