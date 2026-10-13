# RAGBench System Architecture & Specifications

This document outlines the technical architecture, mathematical formulations, and engineering contracts governing the **RAGBench-Evaluation-Lab** platform.

---

## 1. Architectural Philosophy: The Dual-Engine Design

RAGBench is organized into two decoupled, complementary subsystems:

```
                    RAGBench Platform Topology
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────────────────┐       ┌───────────────────────────────┐
│        Research Engine        │       │      Research Workbench       │
│  (Phases A–G / Backend Core)  │       │   (Phase H / Interactive UI)  │
├───────────────────────────────┤       ├───────────────────────────────┤
│ • Deterministic Ingestion     │       │ • Factorial Matrix Builder    │
│ • Dense, BM25 & Hybrid Fusion │       │ • Active Pareto Frontier      │
│ • Paired Statistical Engine   │       │ • Bounded Metric Radar        │
│ • Cryptographic Reproducibility│      │ • Provenance Trace Inspector  │
└───────────────┬───────────────┘       └───────────────▲───────────────┘
                │                                       │
                │        REST API / OpenAPI Contract    │
                └───────────────────────────────────────┘
```

1. **Research Engine (Backend CLI & Python Core):**
   An immutable, deterministic evaluation pipeline that treats Retrieval-Augmented Generation (RAG) pipelines as parameterizable scientific experiments. It enforces strict experimental controls across chunking geometry, embedding models, retrieval paradigms, and paired statistical comparisons.
2. **Research Workbench (Frontend UI):**
   A React 19 application providing interactive multidimensional exploration, Pareto efficiency analysis, multi-metric radar comparisons, and granular claim-level citation attribution inspection.

---

## 2. Layered Pipeline Architecture

```
[Corpus Ingestion] ──► [Chunking & Normalization] ──► [Embedding & Indexing]
                                                             │
[Query Formulation] ──► [Retrieval Engine (Dense/BM25/RRF)] ◄┘
                               │
[Neural Reranker] ────────────►┘
       │
[Context Assembler] ──► [Generation & Synthesis] ──► [Metric Evaluation Engine]
                                                             │
                                                     [Statistical Comparator]
                                                             │
                                                     [Artifact Persistence]
```

### 2.1 Document Ingestion & Chunking Geometry
* **Text Normalization:** Bengali and English texts undergo Unicode NFKC normalization. The normalizer preserves byte-exact coordinates in raw source text to guarantee accurate citation resolution.
* **Chunking Strategies:**
  - `fixed`: Fixed token-length chunking (default: 200 tokens with 20 token overlap).
  - `sentence`: Punctuation- and delimiter-aware sentence chunking (configurable chunk size and overlap).
  - `character`: Character-level offset chunking for raw text evaluation.

### 2.2 Dual Retrieval & Hybrid Fusion Matrix
RAGBench supports three primary retrieval paradigms under identical evaluation parameters:
1. **Dense Semantic Retrieval:** Vector similarity search using dense bi-encoder embeddings (e.g., `BAAI/bge-m3`, `sentence-transformers/all-MiniLM-L6-v2`) with cosine similarity:
   $$\text{Sim}(q, d) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$
2. **Lexical Retrieval (BM25 Okapi):** Inverted index retrieval with Okapi BM25 scoring ($k_1=1.5, b=0.75$):
   $$\text{Score}_{\text{BM25}}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$
3. **Hybrid Reciprocal Rank Fusion (RRF):** Rank-based fusion combining dense and lexical result sets with smoothing constant $k=60$:
   $$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + r_m(d)}$$

---

## 3. Mathematical Metric Formulations

### 3.1 Information Retrieval (IR) Metrics
For a query $q$, retrieved candidate list $R_K = [d_1, \dots, d_K]$, and binary ground-truth relevant documents $G$:
* **Recall@K:**
  $$\text{Recall}@K = \frac{|R_K \cap G|}{|G|}$$
* **Precision@K:**
  $$\text{Precision}@K = \frac{|R_K \cap G|}{K}$$
* **Mean Reciprocal Rank (MRR@K):**
  $$\text{MRR}@K = \begin{cases} \frac{1}{\min \{i \mid d_i \in G\}}, & \text{if } R_K \cap G \neq \emptyset \\ 0, & \text{otherwise} \end{cases}$$
* **Normalized Discounted Cumulative Gain (NDCG@K):**
  $$\text{DCG}@K = \sum_{i=1}^K \frac{\mathbb{I}(d_i \in G)}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

### 3.2 Generation & Factuality Metrics
* **Faithfulness / Groundedness:** Proportion of claims in generated output supported by retrieved context snippets:
  $$\text{Faithfulness} = \frac{|\text{Supported Claims}|}{|\text{Total Claims Extracted}|}$$
* **Citation Precision & Recall:** Precision and recall of inline attribution brackets relative to supporting context passages.

---

## 4. Controlled Statistical Methodology

To prevent pseudoreplication and false discovery across multi-condition benchmarks:
1. **Paired Observations:** Evaluations are computed across matched information units ($N=25$). The same query-document pairs are evaluated under each factorial variation.
2. **Primary Non-Parametric Test:** Two-sided Wilcoxon signed-rank test on paired differences:
   - Exact permutation when non-zero differences $N_{\text{nz}} \le 20$.
   - Asymptotic normal approximation with continuity correction when $N_{\text{nz}} > 20$.
3. **Secondary Parametric Test:** Two-sided paired Student's $t$-test ($df = N - 1$).
4. **Family-Wise Error Rate Control:** Step-Down Holm-Bonferroni correction applied across the pre-registered hypothesis family size ($m=4$):
   $$p_{(i)} \le \frac{\alpha}{m - i + 1}$$
5. **Effect Size:** Standardized paired Cohen's $d_z$:
   $$d_z = \frac{\bar{\Delta}}{s_{\Delta}}$$

---

## 5. Active Pareto Comparability & Control Signatures

In the Academic Research Workbench, multi-run comparisons require identical evaluation conditions. The backend enforces a canonical control signature:

```json
{
  "dataset_id": "multilingual_canonical",
  "dataset_version_id": "dd59f087-86ff-4872-925f-adb03fc8d9a2",
  "benchmark_hash": "75fa540c1041804dec61fe6df28854a62d3883f8c841d14f43465da497b72092",
  "query_population": "25 matched information units",
  "top_k": 5,
  "evaluation_protocol_version": "ragbench-protocol-v1.0",
  "metric_definition_version": "metrics-v1.0",
  "chunking_strategy": "fixed",
  "chunk_size": 200,
  "chunk_overlap": 20,
  "embedding_model": "BAAI/bge-m3",
  "embedding_dimension": 1024,
  "reranker": "none"
}
```

If any experimental control diverges (e.g., chunk size 200 vs. 512, or BGE-M3 vs. MiniLM), the Pareto comparison endpoint explicitly reports divergent dimensions to prevent invalid empirical comparisons.

---

## 6. Persistence & Relational Schema

Experimental records and benchmark baselines are persisted via SQLAlchemy into an asynchronous SQLite / Qdrant store:

* `experiments`: Experiment group definitions, metadata, and sweep parameters.
* `experiment_runs`: Individual runs recording execution parameters, commit hash, configuration SHA-256, and runtime latency.
* `metric_summaries`: Aggregated run-level summary metrics (Recall@K, MRR@K, NDCG@K, Faithfulness).
* `metric_observations`: Unit-level paired observations for statistical hypothesis testing.
* `benchmarks`: Benchmark corpora versions and ground truth evaluation query collections.
