# RAGBench-Evaluation-Lab — Backend Database Schema Specification

**Version:** 1.0  
**Status:** Authoritative database & domain model baseline  

---

## 1. Domain Data Model Hierarchy

```text
Dataset
  │
  ├── DatasetVersion
  │       │
  │       └── Document
  │              │
  │              └── DocumentChunk
  │
  └── Experiment
          │
          └── ExperimentRun
                  │
                  ├── QueryRun
                  │      ├── TransformedQuery
                  │      ├── RetrievedChunk
                  │      ├── RerankedChunk
                  │      ├── PackedContext
                  │      ├── GenerationResult
                  │      └── MetricResult
                  │
                  └── RunMetricSummary
```

---

## 2. Entity Specifications

### 2.1 Dataset Management

#### `Dataset`
The root organizational entity for a corpus of knowledge.
- `id` (VARCHAR(36), PK): UUID.
- `name` (VARCHAR(255), UNIQUE, NOT NULL): Human-readable unique name.
- `description` (TEXT): Description of corpus domain and purpose.
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): Creation timestamp.
- `updated_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): Last update timestamp.
- `current_version_id` (VARCHAR(36), NULL): Pointer to latest active `DatasetVersion`.
- `metadata` (JSONB, NOT NULL, DEFAULT '{}'): Flexible domain metadata.

#### `DatasetVersion`
An immutable snapshot of a dataset corpus.
- `id` (VARCHAR(36), PK): UUID.
- `dataset_id` (VARCHAR(36), FK -> `Dataset.id`, NOT NULL): Parent dataset.
- `version_number` (INTEGER, NOT NULL): Monotonically increasing version (1, 2, ...).
- `content_hash` (VARCHAR(64), NOT NULL): Deterministic Merkle SHA-256 hash of all document checksums.
- `document_count` (INTEGER, NOT NULL): Total documents in this version.
- `total_bytes` (BIGINT, NOT NULL): Total raw size in bytes.
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL): Release timestamp.
- `metadata` (JSONB, NOT NULL, DEFAULT '{}'): Release notes, tags.
- **Constraints:** `UNIQUE(dataset_id, version_number)`.

#### `Document`
A discrete ingested file belonging to a specific dataset version.
- `id` (VARCHAR(36), PK): UUID.
- `dataset_version_id` (VARCHAR(36), FK -> `DatasetVersion.id`, NOT NULL).
- `external_id` (VARCHAR(255), NULL): External tracking ID.
- `filename` (VARCHAR(255), NOT NULL): Original filename.
- `content` (TEXT, NOT NULL): Normalized text extracted from document.
- `content_hash` (VARCHAR(64), NOT NULL): SHA-256 checksum of raw content.
- `mime_type` (VARCHAR(64), NOT NULL): `text/plain`, `text/markdown`, `application/pdf`.
- `size_bytes` (BIGINT, NOT NULL): Raw byte size.
- `page_count` (INTEGER, NOT NULL, DEFAULT 1): Extracted page count.
- `metadata` (JSONB, NOT NULL, DEFAULT '{}'): Frontmatter, headers, document properties.
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL).
- **Constraints:** `UNIQUE(dataset_version_id, content_hash)`.

#### `DocumentChunk`
A discrete segmented passage derived from a Document.
- `id` (VARCHAR(64), PK): Deterministic SHA-256 hash computed as:
  $$	ext{SHA256}(	ext{doc\_id} : 	ext{chunk\_index} : 	ext{NFKC}(	ext{content}))$$
- `document_id` (VARCHAR(36), FK -> `Document.id`, NOT NULL).
- `chunk_index` (INTEGER, NOT NULL): 0-indexed position within document.
- `content` (TEXT, NOT NULL): Chunk text content.
- `content_hash` (VARCHAR(64), NOT NULL): SHA-256 of normalized text.
- `token_count` (INTEGER, NOT NULL): Token count via `tiktoken`.
- `char_count` (INTEGER, NOT NULL): Character count.
- `start_char` (INTEGER, NOT NULL): Start character offset in document.
- `end_char` (INTEGER, NOT NULL): End character offset in document.
- `strategy` (VARCHAR(64), NOT NULL): `fixed_token`, `recursive`, `sentence`, `semantic`.
- `chunking_config_hash` (VARCHAR(64), NOT NULL): Hash of chunker parameters.
- `metadata` (JSONB, NOT NULL, DEFAULT '{}'): Heading hierarchy, domain tags.
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL).
- **Constraints:** `UNIQUE(document_id, chunk_index)`.

---

### 2.2 Experiment Management

#### `Experiment`
Defines the hyperparameter specification and target dataset for an evaluation study.
- `id` (VARCHAR(36), PK): UUID.
- `name` (VARCHAR(255), NOT NULL): Name of evaluation experiment.
- `description` (TEXT): Description of hypothesis under test.
- `dataset_version_id` (VARCHAR(36), FK -> `DatasetVersion.id`, NOT NULL).
- `status` (VARCHAR(32), NOT NULL): `DRAFT`, `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`.
- `configuration` (JSONB, NOT NULL): Complete validated pipeline hyperparameter configuration.
- `configuration_hash` (VARCHAR(64), NOT NULL): SHA-256 hash of canonical configuration JSON.
- `created_by` (VARCHAR(255), NULL): Author identifier.
- `created_at` (TIMESTAMP WITH TIME ZONE, NOT NULL).
- `updated_at` (TIMESTAMP WITH TIME ZONE, NOT NULL).

#### `ExperimentRun`
Represents an individual physical execution of an Experiment.
- `id` (VARCHAR(36), PK): UUID.
- `experiment_id` (VARCHAR(36), FK -> `Experiment.id`, NOT NULL).
- `status` (VARCHAR(32), NOT NULL): `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`.
- `started_at` (TIMESTAMP WITH TIME ZONE, NOT NULL).
- `completed_at` (TIMESTAMP WITH TIME ZONE, NULL).
- `git_commit` (VARCHAR(40), NULL): Commit hash of repository during execution.
- `environment` (VARCHAR(64), NOT NULL): Hostname, OS, Python version.
- `random_seed` (INTEGER, NOT NULL, DEFAULT 42): Seed for reproducibility.
- `summary_metrics` (JSONB, NOT NULL, DEFAULT '{}'): Aggregate metric scores.
- `error` (TEXT, NULL): Failure message if aborted.

---

### 2.3 Evaluation Query Execution Trace

#### `QueryRun`
Represents execution of a single benchmark query within an ExperimentRun.
- `id` (VARCHAR(36), PK): UUID.
- `experiment_run_id` (VARCHAR(36), FK -> `ExperimentRun.id`, NOT NULL).
- `query_id` (VARCHAR(64), NOT NULL): Stable benchmark query ID from dataset.
- `original_query` (TEXT, NOT NULL): Raw user question.
- `expected_answer` (TEXT, NULL): Ground-truth answer.
- `ground_truth_chunks` (JSONB, NOT NULL, DEFAULT '[]'): List of relevant chunk IDs.
- `started_at` (TIMESTAMP WITH TIME ZONE, NOT NULL).
- `completed_at` (TIMESTAMP WITH TIME ZONE, NOT NULL).
- `latency_ms` (DOUBLE PRECISION, NOT NULL): Total query processing duration in ms.
- `status` (VARCHAR(32), NOT NULL): `SUCCESS`, `ERROR`.
- `metadata` (JSONB, NOT NULL, DEFAULT '{}'): Query domain, difficulty tags.

#### `TransformedQuery`
Stores query expansions produced by query pre-processing strategies.
- `id` (VARCHAR(36), PK): UUID.
- `query_run_id` (VARCHAR(36), FK -> `QueryRun.id`, NOT NULL).
- `transformation_type` (VARCHAR(32), NOT NULL): `none`, `hyde`, `multi_query`, `step_back`.
- `query_text` (TEXT, NOT NULL): Transformed query or hypothetical passage.
- `sequence_index` (INTEGER, NOT NULL): Position (0, 1, 2, ...).
- `metadata` (JSONB, NOT NULL, DEFAULT '{}'): LLM prompts, model used.

#### `RetrievedChunk`
Captures raw passage retrieval rankings before reranking.
- `id` (VARCHAR(36), PK): UUID.
- `query_run_id` (VARCHAR(36), FK -> `QueryRun.id`, NOT NULL).
- `retriever_type` (VARCHAR(32), NOT NULL): `dense`, `bm25`, `hybrid`.
- `chunk_id` (VARCHAR(64), NOT NULL): Foreign chunk ID.
- `rank` (INTEGER, NOT NULL): 1-based initial retrieval rank.
- `score` (DOUBLE PRECISION, NOT NULL): Retrieval relevance score (cosine, BM25, RRF).
- `metadata` (JSONB, NOT NULL, DEFAULT '{}').

#### `RerankedChunk`
Captures re-scoring and rank shifts produced by cross-encoders.
- `id` (VARCHAR(36), PK): UUID.
- `query_run_id` (VARCHAR(36), FK -> `QueryRun.id`, NOT NULL).
- `chunk_id` (VARCHAR(64), NOT NULL): Foreign chunk ID.
- `original_rank` (INTEGER, NOT NULL): Rank prior to reranking.
- `reranked_rank` (INTEGER, NOT NULL): 1-based post-reranking rank.
- `original_score` (DOUBLE PRECISION, NOT NULL): Pre-reranking score.
- `reranker_score` (DOUBLE PRECISION, NOT NULL): Cross-encoder logit/probability.
- `metadata` (JSONB, NOT NULL, DEFAULT '{}').

#### `PackedContext`
Captures the final context window provided to the generator.
- `id` (VARCHAR(36), PK): UUID.
- `query_run_id` (VARCHAR(36), FK -> `QueryRun.id`, UNIQUE, NOT NULL): 1-to-1 relationship.
- `context_text` (TEXT, NOT NULL): Formatted context text with citation headers.
- `token_count` (INTEGER, NOT NULL): Exact token length via `tiktoken`.
- `token_budget` (INTEGER, NOT NULL): Maximum token budget ceiling.
- `ordering_strategy` (VARCHAR(32), NOT NULL): `standard`, `lost_in_the_middle`.
- `chunk_ids` (JSONB, NOT NULL): Ordered array of chunk IDs included.
- `metadata` (JSONB, NOT NULL, DEFAULT '{}').

#### `GenerationResult`
Captures LLM completion and resource metrics.
- `id` (VARCHAR(36), PK): UUID.
- `query_run_id` (VARCHAR(36), FK -> `QueryRun.id`, UNIQUE, NOT NULL): 1-to-1 relationship.
- `model` (VARCHAR(128), NOT NULL): LLM identifier.
- `answer` (TEXT, NOT NULL): Generated answer text.
- `prompt_version` (VARCHAR(64), NOT NULL): Version tag of generation prompt.
- `input_tokens` (INTEGER, NOT NULL): Prompt token count.
- `output_tokens` (INTEGER, NOT NULL): Completion token count.
- `total_tokens` (INTEGER, NOT NULL): Combined token consumption.
- `latency_ms` (DOUBLE PRECISION, NOT NULL): LLM inference duration in ms.
- `estimated_cost` (DOUBLE PRECISION, NOT NULL, DEFAULT 0.0): Estimated cost in USD.
- `metadata` (JSONB, NOT NULL, DEFAULT '{}').

#### `MetricResult`
Captures individual evaluation metrics per query.
- `id` (VARCHAR(36), PK): UUID.
- `query_run_id` (VARCHAR(36), FK -> `QueryRun.id`, NOT NULL).
- `metric_name` (VARCHAR(64), NOT NULL): `recall@5`, `mrr@10`, `faithfulness`, etc.
- `metric_value` (DOUBLE PRECISION, NOT NULL): Computed numerical score.
- `metric_version` (VARCHAR(32), NOT NULL): Version of evaluation algorithm.
- `evaluator` (VARCHAR(64), NOT NULL): Evaluator class name.
- `metadata` (JSONB, NOT NULL, DEFAULT '{}'): Proposition breakdown, reasoning.
- **Constraints:** `UNIQUE(query_run_id, metric_name)`.

#### `RunMetricSummary`
Stores aggregate statistical metrics across all queries in a run.
- `id` (VARCHAR(36), PK): UUID.
- `experiment_run_id` (VARCHAR(36), FK -> `ExperimentRun.id`, NOT NULL).
- `metric_name` (VARCHAR(64), NOT NULL).
- `mean` (DOUBLE PRECISION, NOT NULL).
- `median` (DOUBLE PRECISION, NOT NULL).
- `min` (DOUBLE PRECISION, NOT NULL).
- `max` (DOUBLE PRECISION, NOT NULL).
- `stddev` (DOUBLE PRECISION, NOT NULL).
- `count` (INTEGER, NOT NULL).
- `metadata` (JSONB, NOT NULL, DEFAULT '{}').
- **Constraints:** `UNIQUE(experiment_run_id, metric_name)`.

---

## 3. Database Indexes & Performance Optimization

To support high-throughput analytical queries and fast UI comparisons, the following indexes are strictly established:

```sql
-- Fast lookup of active dataset versions
CREATE INDEX idx_dataset_version ON dataset_versions (dataset_id, version_number);

-- Document deduplication lookup
CREATE INDEX idx_document_hash ON documents (dataset_version_id, content_hash);

-- Chunk retrieval by document
CREATE INDEX idx_chunk_doc ON document_chunks (document_id, chunk_index);

-- Experiment lookups by dataset and status
CREATE INDEX idx_experiment_status ON experiments (dataset_version_id, status);

-- Query trace lookups by run
CREATE INDEX idx_query_run_exp ON query_runs (experiment_run_id, status);

-- Fast metric aggregation across runs
CREATE INDEX idx_metric_run ON metric_results (query_run_id, metric_name);
CREATE INDEX idx_summary_run ON run_metric_summaries (experiment_run_id, metric_name);
```

---

## 4. Persistence Responsibilities

1. **PostgreSQL (Authoritative Record):** Single source of truth for all configurations, document metadata, raw execution traces, generation outputs, and evaluation metrics.
2. **Qdrant (Vector Index):** Purely a search index storing dense vectors alongside minimum required payload (`doc_id`, `chunk_id`, `strategy`). It is NOT the experiment database.
3. **Redis (Broker & Cache):** Transient task queue broker and fast retrieval cache.
4. **Local MinIO / Disk Blob Store:** Raw file storage for binary PDFs and large datasets.

---

## 5. Schema Migration & Invariant Rules

- All database schema modifications must be authored as deterministic, reversible Alembic migrations.
- No direct manual edits to database schemas are permitted.
- **The Core Chain Invariant:**
  $$	ext{Configuration} \longrightarrow 	ext{Execution} \longrightarrow 	ext{Raw Evidence} \longrightarrow 	ext{Metrics} \longrightarrow 	ext{Analysis}$$
  Every metric score in the database must be traceable back to its underlying `QueryRun` raw evidence, and every `QueryRun` must be traceable to its immutable `Experiment` configuration.
