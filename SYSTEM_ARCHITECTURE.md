# System Architecture Specification — RAGBench (Evaluation Laboratory)

**Document Version:** 1.0.0-PROD  
**Target Repository:** `RAGBench-Evaluation-Lab`  
**Classification:** Research Infrastructure & Comparative Benchmarking Platform  

---

## 1. High-Level Architectural Vision

RAGBench is an enterprise and academic-grade evaluation laboratory that treats Retrieval-Augmented Generation (RAG) pipelines as parameterizable scientific experiments. Rather than serving as a static question-answering assistant, RAGBench systematically benchmarks variations across:
- **Chunking Strategies** (Fixed, Overlap, Semantic, Recursive, Hierarchical)
- **Embedding Representations** (Dense vector embeddings, sparse BM25 token frequencies, late-interaction ColBERT style)
- **Retrieval Topologies** (Dense vector search, Lexical BM25, Reciprocal Rank Fusion / Relative Score Fusion)
- **Neural Rerankers** (Cross-Encoders, FlashRank, LLM-as-a-Reranker)
- **Query Transformers** (HyDE, Multi-Query Expansion, Step-Back Prompting)
- **Context Construction** (Sliding context budgets, Lost-in-the-Middle mitigation)
- **Generation & LLM Providers** (Quantized local models via Ollama/vLLM vs. frontier cloud models via LiteLLM)
- **Multilingual Generalization** (Cross-lingual retrieval parity across English and Bengali)

```
                        +-------------------------------------------------------+
                        |            React 19 / Vite Research UI                |
                        |   (Radar Comparisons, Pareto Frontiers, Trace Inspec) |
                        +---------------------------+---------------------------+
                                                    |
                                                    v
                        +-------------------------------------------------------+
                        |          FastAPI Dual-Mode Ingestion Gateway          |
                        |   (Async Task Dispatcher, Experiment Definition API)  |
                        +---------------------------+---------------------------+
                                                    |
                                                    v
                        +-------------------------------------------------------+
                        |          Celery Distributed Task Orchestrator         |
                        |   (Queues: 'indexer_lane', 'retrieval_lane', 'eval')   |
                        +---------------------------+---------------------------+
                                                    |
                 +----------------------------------+----------------------------------+
                 |                                                                     |
                 v                                                                     v
+----------------------------------+                                  +----------------------------------+
|       Corpus Indexing Bus        |                                  |     Query Evaluation Matrix      |
|                                  |                                  |                                  |
|  * Recursive / Semantic Chunker  |                                  |  * Query Transformation / HyDE   |
|  * FastEmbed / SentenceTrans     |                                  |  * Hybrid Search (BM25 + Dense)  |
|  * Tantivy BM25 / Qdrant Engine  |                                  |  * Cross-Encoder Neural Rerank   |
+-----------------+----------------+                                  +-----------------+----------------+
                  |                                                                     |
                  v                                                                     v
+----------------------------------+                                  +----------------------------------+
| Persistence & Vector Storage     |                                  |   Automated Evaluation Engine    |
|                                  |                                  |                                  |
|  * Qdrant (HNSW Cosine Vector)   |                                  |  * IR: Recall@K, MRR@K, NDCG@K   |
|  * SQLite WAL / PostgreSQL RLS   |                                  |  * RAGAS: Faithfulness, Rel      |
|  * Local MinIO / Disk Blob Store |                                  |  * Hallucination & Citation Check|
+----------------------------------+                                  +----------------------------------+
```

---

## 2. Mathematical Formalization of Evaluation Metrics

### 2.1 Information Retrieval (IR) Metrics

Given a query $q \in Q$, let $R_q$ denote the ground-truth set of relevant documents/passages for $q$, and let $\hat{R}_{q, K} = [d_1, d_2, \dots, d_K]$ denote the ranked list of top-$K$ retrieved passages returned by the retriever.

#### 1. Recall@K
The proportion of relevant items retrieved in the top $K$ results:
$$\text{Recall@K}(q) = \frac{\vert{}\hat{R}_{q, K} \cap R_q\vert{}}{\vert{}R_q\vert{}}$$
Mean Recall over dataset $Q$:
$$\text{Recall@K} = \frac{1}{\vert{}Q\vert{}} \sum_{q \in Q} \text{Recall@K}(q)$$

#### 2. Precision@K
The proportion of retrieved items in the top $K$ that are relevant:
$$\text{Precision@K}(q) = \frac{\vert{}\hat{R}_{q, K} \cap R_q\vert{}}{K}$$

#### 3. Mean Reciprocal Rank (MRR@K)
Evaluates the rank of the first relevant document retrieved:
$$\text{RR}(q) = \begin{cases} \frac{1}{\text{rank}_q}, & \text{if } \exists d \in \hat{R}_{q, K} \cap R_q \text{ at index } \text{rank}_q \le K \\ 0, & \text{otherwise} \end{cases}$$
$$\text{MRR@K} = \frac{1}{\vert{}Q\vert{}} \sum_{q \in Q} \text{RR}(q)$$

#### 4. Normalized Discounted Cumulative Gain (NDCG@K)
Measures ranking quality with graded relevance $r_i \in \{0, 1, 2, \dots\}$:
$$\text{DCG@K}(q) = \sum_{i=1}^{K} \frac{2^{r_i} - 1}{\log_2(i + 1)}$$
$$\text{NDCG@K}(q) = \frac{\text{DCG@K}(q)}{\text{IDCG@K}(q)}$$
where $\text{IDCG@K}(q)$ is the ideal DCG obtained by sorting items by true relevance.

---

### 2.2 RAG Generation & Factuality Metrics

#### 1. Answer Faithfulness (Factuality / Non-Hallucination)
Let $A$ denote the generated answer decomposed into a set of atomic propositional statements $S_A = \{s_1, s_2, \dots, s_m\}$. Let $C$ denote the concatenated retrieved context.
$$\text{Faithfulness}(A, C) = \frac{\vert{}\{s \in S_A \mid C \vdash s\}\vert{}}{\vert{}S_A\vert{}}$$
where $C \vdash s$ denotes that statement $s$ is directly entailed by context $C$, computed using natural language inference (NLI) cross-encoders or structured zero-shot prompting.

#### 2. Answer Relevance
Measures whether the generated answer directly addresses the question without containing extraneous fluff:
$$\text{Answer Relevance} = \frac{1}{N} \sum_{i=1}^{N} \cos\Big(\mathbf{e}(A), \mathbf{e}(q_{\text{gen}}^{(i)})\Big)$$
where $q_{\text{gen}}^{(i)}$ are synthetic questions generated by an LLM prompted only with answer $A$, and $\mathbf{e}(\cdot)$ represents dense semantic embeddings.

#### 3. Citation Precision & Recall
For generated answers containing inline citations $[c_1, c_2, \dots, c_p]$ linking to context chunks $d_k$:
$$\text{Citation Precision} = \frac{\text{Number of citations providing direct entailment}}{\text{Total citations made}}$$
$$\text{Citation Recall} = \frac{\text{Number of required context facts cited}}{\text{Total factual claims in generated answer}}$$

---

## 3. Core Component Contracts

### 3.1 Experiment Configuration Schema (`experiment.yaml`)

```yaml
experiment:
  name: "eval-hybrid-rag-chunk-sweep"
  version: "1.0.0"
  description: "Benchmarking dense vs hybrid retrieval across 3 chunk sizes on academic papers"

dataset:
  id: "academic-research-v1"
  domain: "academic_research"
  ground_truth_path: "datasets/academic_research/eval_qa.jsonl"
  corpus_path: "datasets/academic_research/corpus/"

pipeline_matrix:
  chunking:
    - strategy: "fixed_character"
      chunk_size: 256
      chunk_overlap: 32
    - strategy: "recursive"
      chunk_size: 512
      chunk_overlap: 64
    - strategy: "semantic"
      similarity_threshold: 0.82

  embeddings:
    - provider: "fastembed"
      model_name: "BAAI/bge-small-en-v1.5"
    - provider: "sentence_transformers"
      model_name: "BAAI/bge-base-en-v1.5"

  retrieval:
    - mode: "dense"
      top_k: 20
    - mode: "bm25"
      top_k: 20
    - mode: "hybrid"
      fusion: "rrf" # Reciprocal Rank Fusion
      rrf_k: 60
      top_k: 20

  reranker:
    - enabled: false
    - enabled: true
      model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
      top_n: 5

  generation:
    provider: "litellm"
    model: "ollama/qwen2.5-coder:7b" # local zero-cost evaluation default
    temperature: 0.0
    max_tokens: 512

evaluation:
  metrics:
    - "recall@5"
    - "recall@10"
    - "mrr@10"
    - "ndcg@10"
    - "faithfulness"
    - "answer_relevance"
    - "citation_precision"
    - "latency_p95"
    - "cost_per_query"
```

### 3.2 Evaluation Run Result Data Contract (`result.json`)

```json
{
  "run_id": "run_01J8ABCXYZ",
  "experiment_name": "eval-hybrid-rag-chunk-sweep",
  "configuration_hash": "a4f891b2c7e0",
  "timestamp": "2026-09-23T14:30:00Z",
  "pipeline_parameters": {
    "chunk_strategy": "recursive",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "retrieval_mode": "hybrid",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "generation_model": "ollama/qwen2.5-coder:7b"
  },
  "aggregate_metrics": {
    "retrieval": {
      "recall_at_5": 0.884,
      "recall_at_10": 0.941,
      "precision_at_5": 0.712,
      "mrr_at_10": 0.812,
      "ndcg_at_10": 0.865
    },
    "generation": {
      "faithfulness": 0.934,
      "answer_relevance": 0.912,
      "citation_precision": 0.890,
      "citation_recall": 0.875,
      "hallucination_rate": 0.066
    },
    "system": {
      "latency_retrieval_mean_ms": 42.1,
      "latency_rerank_mean_ms": 110.4,
      "latency_generation_mean_ms": 1120.0,
      "latency_e2e_p95_ms": 1420.0,
      "tokens_prompt_mean": 1845,
      "tokens_completion_mean": 210,
      "estimated_cost_usd_per_1k_queries": 0.42
    }
  }
}
```

## 4. Storage & Retrieval Strategy

- **Vector Embeddings & HNSW Indices:** Handled by Qdrant (running via Docker or embedded memory mode for CI). Collection names follow strict naming: `ragbench_{dataset_id}_{chunking_hash}_{embedding_model_hash}`.
- **Lexical Keyword Search:** Implemented via in-process Rank-BM25 (Python) for small local benchmarks, with Tantivy bindings for large monorepo corpora ($> 100\text{k}$ chunks).
- **Structured Relational Storage:** SQLite with WAL mode enabled for standalone local execution; PostgreSQL 16 for distributed Celery deployment. Stores datasets, experiment configurations, and individual query execution traces.
