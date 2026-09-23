# RAGBench-Evaluation-Lab — Product Requirements Document (PRD)

**Version:** 1.0  
**Status:** Authoritative baseline  
**Project Type:** Research + Engineering Platform  
**Primary Goal:** Controlled evaluation and benchmarking of Retrieval-Augmented Generation (RAG) systems  

---

## 1. Product Overview

RAGBench-Evaluation-Lab is a research-grade platform for designing, executing, comparing, and analyzing Retrieval-Augmented Generation (RAG) experiments.

**The system is not primarily a chatbot.**

Its primary purpose is to answer fundamental empirical questions such as:
- Does hybrid retrieval outperform dense retrieval on technical documentation?
- How much does neural cross-encoder reranking improve retrieval quality, and at what latency cost?
- Which chunking strategy (fixed-token, recursive, sentence, semantic) produces the best retrieval results?
- How does retrieval depth ($K=3, 5, 10, 20$) affect answer quality, faithfulness, latency, and token cost?
- Does better retrieval necessarily produce better generated answers, or can generation fail despite high recall?
- Which query transformation technique (Identity, HyDE, Multi-Query, Step-Back) provides measurable improvement on specific query types?
- Can a compact local open-weight model achieve comparable faithfulness to frontier cloud models when retrieval context is optimized?
- How do RAG configurations behave across English, Bengali, code identifiers, and mixed-language technical datasets?

Every experiment must produce reproducible, inspectable, and statistically verifiable evidence.

---

## 2. Product Vision

The platform enables researchers and ML engineers to execute the following modular research workflow:

```text
Dataset
   ↓
Document Ingestion & Parsing (PDF, Markdown, TXT)
   ↓
Chunking (Fixed, Recursive, Sentence, Semantic)
   ↓
Embedding (FastEmbed, SentenceTransformers, Cloud, Mock)
   ↓
Vector & Lexical Indexing (Qdrant, BM25)
   ↓
Retrieval (Dense, BM25, Hybrid RRF / RSN)
   ↓
Reranking (FlashRank, Cross-Encoder, Mock)
   ↓
Query Transformation (Identity, HyDE, Multi-Query, Step-Back)
   ↓
Context Formulation (Token Budget, Lost-in-the-Middle Reordering, Citations)
   ↓
LLM Generation (LiteLLM, Local Ollama, Cloud Frontier)
   ↓
Evaluation (IR Metrics, Faithfulness, Hallucination, Citation Precision/Recall)
   ↓
Experiment Run Results
   ↓
Multi-Run Comparison Matrix
   ↓
Evidence-Backed Research Conclusions
```

Every stage of this pipeline is parameterized, observable, and independently switchable.

---

## 3. Primary Users

### 3.1 Researcher
Needs to:
- Create, curate, and version benchmark datasets with deterministic checksums.
- Configure Cartesian parameter sweeps across retrieval, reranking, and generation topologies.
- Execute controlled ablation studies isolating single experimental variables.
- Inspect individual query-level traces (retrieved chunks, scores, reranked positions, packed context, generated text).
- Compute standardized Information Retrieval (Recall, Precision, MRR, NDCG) and generation factuality metrics.
- Export raw query-level evidence and aggregate summaries in JSON, CSV, and LaTeX for academic publications.
- Reproduce past experimental runs with identical configuration hashes.

### 3.2 ML/AI Engineer
Needs to:
- Benchmark retrieval architectures for latency (p50, p95, p99), token consumption, and cost.
- Evaluate trade-offs between local CPU/GPU execution (FastEmbed, FlashRank, Ollama) and cloud frontier APIs.
- Detect pipeline regressions through rigorous automated quality gates.
- Verify component interchangeability through strict interface contracts.

### 3.3 Student / Portfolio Reviewer
Needs to:
- Understand the scientific rationale behind each RAG architecture decision.
- Trace how data flows through ingestion, chunking, retrieval, reranking, and evaluation.
- Inspect clear, un-obfuscated empirical data without arbitrary, ungrounded "AI quality scores."

---

## 4. Core Product Principles

### 4.1 Experiment-First
The platform treats an **Experiment** as a first-class citizen. Individual interactive queries exist purely for diagnostic and inspection purposes, not as the primary product unit.

### 4.2 Absolute Reproducibility
An experiment must capture and persist sufficient metadata to reproduce the exact run:
- Dataset version and document/chunk content checksums.
- Chunking strategy, parameters (chunk size, overlap), and tokenizer encoding.
- Embedding provider, model name, dimensionality, and distance metric.
- Retrieval topology, candidate depth, and score fusion parameters.
- Reranker model identity, candidate cutoffs, and device configuration.
- Query transformation strategy, prompt templates, and random seeds.
- Context packing budget, reordering strategy, and citation formatting.
- LLM model identifier, generation temperature, and system prompts.
- Evaluation metric formulas, classifier models, and software/Git commit hashes.

### 4.3 Component Independence & Decoupling
Every pipeline component must be independently replaceable. Retrievers must not depend on rerankers; query transformers must not depend on retrievers; evaluation metrics must not depend on specific LLMs.

### 4.4 Evidence Over Assumptions
The system must never hardcode assumptions such as "Hybrid retrieval is superior to dense retrieval" or "Reranking always improves MRR." The platform's duty is to measure and expose raw empirical evidence, enabling the researcher to draw valid conclusions.

### 4.5 Configuration Over Hardcoding
All experimental variables are represented as serializable, validated configuration schemas.

---

## 5. Functional Requirements (FR)

### FR-001: Dataset Management
- The system shall support creating, versioning, and managing datasets.
- Supported ingestion formats: Markdown (`.md`), PDF (`.pdf`), and Plain Text (`.txt`).
- The system shall compute deterministic SHA-256 checksums for raw documents and detect duplicates.
- Document metadata (filename, mime type, page counts, author, tags) shall be stored and preserved.
- Dataset versions shall be immutable: modifying dataset contents creates a new `DatasetVersion`.

### FR-002: Document Chunking
- The system shall implement four discrete chunking strategies:
  1. `FixedTokenChunker`: strict token-bounded windowing with sliding overlap.
  2. `RecursiveCharacterChunker`: hierarchical structure-aware splitting on paragraphs and newlines.
  3. `SentenceChunker`: sentence boundary segmentation supporting English and Bengali (`।`, `॥`).
  4. `SemanticChunker`: dynamic boundary detection based on consecutive sentence embedding cosine distance.
- Each chunk must receive a deterministic identifier computed as:
  $$	ext{chunk\_id} = 	ext{SHA256}(	ext{doc\_id} : 	ext{chunk\_index} : 	ext{NFKC}(	ext{content}))$$
- Chunk boundaries (`start_char`, `end_char`, `token_count`) must be preserved.

### FR-003: Embedding Representation
- The system shall support interchangeable embedding providers:
  - Local CPU ONNX: `FastEmbed` (`BAAI/bge-small-en-v1.5`, `multilingual-e5-small`).
  - PyTorch: `SentenceTransformers` (`BAAI/bge-base-en-v1.5`).
  - Cloud Providers: OpenAI / Cohere.
  - Deterministic Mock Provider: offline n-gram hash projection for unit testing.
- Provider identity, model name, and vector dimensionality must be recorded.

### FR-004: Vector & Inverted Index Storage
- The system shall support Qdrant (in-memory mode for testing and remote server for production).
- Vector collections must be deterministically named following:
  $$	ext{ragbench\_}\{	ext{dataset\_id}\}\_\{	ext{chunking\_hash}\}\_\{	ext{embedding\_model\_hash}\}$$
- The system shall support in-process BM25 inverted index serialization (`.json` / `.bm25.pkl`) with fast reloading.

### FR-005: Comparative Retrieval Engine
- The system shall support three primary retrieval topologies:
  1. `DenseRetriever`: Approximate Nearest Neighbor (ANN) search on dense vectors.
  2. `BM25Retriever`: Okapi BM25 lexical keyword search with multilingual Unicode tokenization.
  3. `HybridRetriever`: Dual-channel retrieval combining Dense and BM25.
- Supported score fusion algorithms:
  - **Reciprocal Rank Fusion (RRF):**
    $$RRF(d) = \sum_{m \in M} rac{1}{k + r_m(d)}$$
  - **Relative Score Normalization (RSN):**
    $$s_{	ext{norm}}(d) = rac{s(d) - \min s}{\max s - \min s + \epsilon}, \quad S_{	ext{final}}(d) = lpha \cdot s_{	ext{dense}} + (1 - lpha) \cdot s_{	ext{bm25}}$$
- All retrievers must respect `top_k`, support `filter_metadata`, and resolve score ties deterministically by ascending `chunk_id`.

### FR-006: Neural Passage Reranking
- Reranking shall be an independent, composable post-retrieval stage.
- Supported topologies:
  - `Dense` and `Dense + Reranker`
  - `BM25` and `BM25 + Reranker`
  - `Hybrid` and `Hybrid + Reranker`
- Supported rerankers:
  - `FlashRankReranker`: CPU-optimized ONNX cross-encoder (`ms-marco-TinyBERT-L-2-v2`).
  - `CrossEncoderReranker`: PyTorch cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
  - `DeterministicMockReranker`: Offline keyword-overlap reranker for unit tests.
- Rerankers must preserve candidate provenance and output structured `RankedChunk` records.

### FR-007: Query Transformation Pipeline
- The system shall support four independently switchable query transformation strategies:
  1. `IdentityTransformer` (`strategy="none"`): Pass-through baseline control.
  2. `HyDETransformer`: Generates a hypothetical technical passage answering the query to retrieve by semantic representation.
  3. `MultiQueryExpander`: Generates $N$ distinct query perspectives and technical formulations, retrieving candidates per query and merging via RRF.
  4. `StepBackTransformer`: Extracts high-level conceptual questions to retrieve fundamental principles alongside specific facts.
- Every transformation must log original query, strategy, transformed queries, and prompts in `QueryTransformationResult`.

### FR-008: Context Formulation & Lost-in-the-Middle Mitigation
- `ContextBuilder` shall pack retrieved passages into a single context window strictly bounded by `token_budget` using `tiktoken`.
- Supported passage ordering strategies:
  - `standard`: Preserves retrieval rank order $[d_1, d_2, \dots, d_K]$.
  - `lost_in_the_middle`: Alternating placement $[d_1, d_3, d_5, \dots, d_6, d_4, d_2]$ concentrating relevance at context window boundaries.
- Citation Header Injection: Each context snippet must be tagged with explicit provenance:
  `[Source {i}] (doc: {doc_id}, chunk: {chunk_id})
{content}`.

### FR-009: Generation Engine
- The system shall interface with LLM generation providers (local Ollama / Qwen-2.5-Coder and frontier cloud models).
- Every generated answer must record answer text, model identifier, system prompt, temperature, token counts, latency, and cited sources.

---

## 6. Evaluation Metrics Requirements

### 6.1 Information Retrieval (IR) Metrics
- **Recall@K:** $rac{|\hat{R}_{q, K} \cap R_q|}{|R_q|}$
- **Precision@K:** $rac{|\hat{R}_{q, K} \cap R_q|}{K}$
- **MRR@K (Mean Reciprocal Rank):** $rac{1}{	ext{rank}_q}$ of first relevant hit $\le K$.
- **NDCG@K:** Normalized Discounted Cumulative Gain supporting binary and graded relevance.
- **Hit@K:** Binary indicator ($1$ if hit $\ge 1$ else $0$).

### 6.2 Generation & Factuality Metrics
- **Proposition Decomposition:** Splits generated answer $A$ into atomic factual statements $S_A = \{s_1, \dots, s_m\}$.
- **Faithfulness Score:** $rac{|\{s \in S_A \mid C dash s\}|}{|S_A|}$ using NLI or zero-shot LLM-as-a-judge.
- **Hallucination Rate:** $1.0 - 	ext{Faithfulness}$.
- **Answer Relevance:** Cosine similarity between original query and synthetic questions generated from the answer.

### 6.3 Citation Accuracy Metrics
- **Citation Precision:** $rac{	ext{Entailing citations}}{	ext{Total citations made}}$.
- **Citation Recall:** $rac{	ext{Factual claims with valid citations}}{	ext{Total factual claims in generated answer}}$.

### 6.4 System Performance Metrics
- Latency per component and end-to-end (p50, p95, p99).
- Token usage (prompt tokens, completion tokens, context tokens).
- Estimated query cost based on model pricing tables.

---

## 7. Experiment Management & Lifecycle

- Each experiment possesses: `id`, `name`, `description`, `dataset_version_id`, `configuration`, `status`, `created_at`, `updated_at`.
- Lifecycle Statuses:
  - `DRAFT`: Configuration being authored.
  - `QUEUED`: Scheduled for execution.
  - `RUNNING`: Actively processing parameter sweeps.
  - `COMPLETED`: Finished with all metrics calculated and stored.
  - `FAILED`: Aborted due to an unrecoverable error.
  - `CANCELLED`: Manually halted by the user.
- **Immutability Invariant:** Once an experiment enters `RUNNING`, its configuration is locked and immutable. Re-running requires cloning into a new `ExperimentRun`.

---

## 8. Multi-Run Experiment Comparison

The platform must allow side-by-side comparison across arbitrary runs:
- Direct metric contrast (Recall@5, MRR@10, NDCG@10, Faithfulness, Answer Relevance, Latency, Cost).
- Inspectable per-query divergence: identify exactly which queries improved or regressed between configurations.
- Raw query evidence must never be hidden behind an unexplained aggregate score.

---

## 9. Formal Research Questions (RQ1–RQ6)

The platform is purpose-built to evaluate:
- **RQ1 (Chunking):** Impact of chunk size (256 vs 512 vs 1024) and boundary logic (fixed vs recursive vs semantic) on recall and downstream faithfulness.
- **RQ2 (Retrieval Topology):** Comparative performance of Dense vs BM25 vs Hybrid (RRF/RSN) on domain-specific technical documentation.
- **RQ3 (Reranking Utility):** Quantitative trade-off between retrieval recall/MRR gains and inference latency penalties using cross-encoders.
- **RQ4 (Context Window Depth):** Effect of context budget (Top-3 vs Top-5 vs Top-10 vs Top-20) and Lost-in-the-Middle mitigation on answer faithfulness.
- **RQ5 (Disentangling Quality):** Correlation between retrieval recall and answer factual accuracy.
- **RQ6 (Model Efficiency):** Feasibility of compact local models matching frontier cloud models when supplied with optimized reranked context.

---

## 10. Non-Functional Requirements (NFR)

- **NFR-001 (Reproducibility):** Identical configurations on identical datasets must produce bitwise identical retrieval ranks and metric scores under deterministic settings.
- **NFR-002 (Observability):** Every pipeline operation must record component name, experiment ID, run ID, query ID, execution latency, and error traces.
- **NFR-003 (Testability):** 100% type safety with `mypy --strict`, linting compliance with `ruff`, and comprehensive unit/contract test coverage.
- **NFR-004 (Extensibility):** Adding a new retriever, chunker, or metric must only require implementing the respective abstract base class without modifying existing components.
- **NFR-005 (Zero Offline Dependencies):** Standard unit tests must run offline without requiring external network connectivity or paid API credits.

---

## 11. Explicit Non-Goals

The platform is NOT:
- A generic customer-facing conversational chatbot.
- A social collaborative document management system.
- An autonomous multi-agent web scraping bot.
- A production enterprise CRM system.

Every feature implemented must directly serve RAG benchmarking, evaluation, reproducibility, or scientific analysis.

---

## 12. Quality Gate & Definition of Done

A development phase is complete only when:
1. All detailed engineering tasks are fully implemented.
2. Unit tests and behavioral contract tests pass (100%).
3. Code passes `ruff check app tests` with 0 errors.
4. Code passes `ruff format --check app tests` cleanly.
5. Code passes `mypy app tests --strict` with 0 errors.
6. Frontend passes `tsc --noEmit`, vitest unit tests, and production build.
7. Architectural boundaries and scope limits are strictly maintained.
8. Git working tree is clean with a descriptive conventional commit.
