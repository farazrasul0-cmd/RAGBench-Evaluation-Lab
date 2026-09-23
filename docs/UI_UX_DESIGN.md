# RAGBench-Evaluation-Lab — UI/UX Design Specification

**Version:** 1.0  
**Status:** Authoritative UI baseline  

---

## 1. Design Philosophy

RAGBench is a **research workstation, developer tool, and analytics platform**—it is **NOT** a consumer chatbot application.

The visual language communicates:
- **Rigor & Transparency:** Expose exact configuration hashes, chunk boundaries, retrieval scores, and evaluation metrics.
- **Evidence-First Layout:** Never display an arbitrary "AI quality score" without providing direct links to the underlying retrieved chunks, citation mappings, and proposition entailment verdicts.
- **Comparative Density:** Provide dense, readable data tables, clean metric comparisons, and expandable technical details suitable for research analysis.

---

## 2. Global Navigation

The application navigation sidebar contains six primary destinations:

1. **Dashboard (`/`):** High-level overview of datasets, ongoing and completed experiments, quick stats, and global system health.
2. **Datasets (`/datasets`):** Dataset management, document ingestion, chunking configuration inspection, and version tracking.
3. **Experiments (`/experiments`):** Experiment configuration authoring, Cartesian parameter sweep matrix, run queues, and execution logs.
4. **Compare (`/compare`):** Multi-run comparison matrix evaluating metric trade-offs across configurations.
5. **Evaluations (`/evaluations`):** Deep-dive query-level inspection tool tracing individual query retrieval, reranking, context, and faithfulness.
6. **System (`/system`):** Vector database connectivity, LLM provider settings, queue status, and hardware resource monitoring.

---

## 3. Core Screens & Workflows

### 3.1 Research Dashboard
- **Top Summary Cards:** Active Datasets, Total Indexed Chunks, Completed Experiment Runs, Average System Faithfulness.
- **Recent Experiments Table:** Lists recent runs with status badges (`COMPLETED`, `RUNNING`, `FAILED`), configuration summary, Recall@5, MRR, Faithfulness, and execution duration.
- **Quick Action:** "New Experiment" button initiating the structured experiment builder workflow.

### 3.2 Dataset Detail Page (`/datasets/{id}`)
- **Overview Tab:** Document count, total tokens, content checksum, current version number, and indexing status.
- **Documents Tab:** Data table displaying uploaded files (PDF, Markdown, TXT), deterministic file hashes, file sizes, and parsing status.
- **Chunks Tab:** Interactive chunk explorer allowing researchers to inspect individual segmented passages, exact character offsets, token counts, and chunking strategy.
- **Versions Tab:** Immutable historical log of dataset releases (`v1`, `v2`, etc.) with document count diffs.

### 3.3 Visual Pipeline & Experiment Builder (`/experiments/new`)
A structured 10-step wizard with an interactive visual pipeline graph:

```text
[Dataset] → [Chunker] → [Embedding] → [Retriever] → [Reranker] → [Query Transform] → [Context] → [LLM] → [Evaluation]
```

- Each node displays active parameters (e.g. `Hybrid (RRF k=60, top_k=20)`).
- Allows defining Cartesian sweeps (e.g. selecting 3 chunk sizes $	imes$ 2 retrieval modes).
- Validates the complete pipeline configuration before submitting to the task queue.

### 3.4 Experiment Run Detail Page (`/experiments/{id}`)
- **Header:** Run status badge, execution duration, configuration hash, Git commit hash, timestamps.
- **Metric Summary Cards:** Recall@5, Precision@5, MRR@10, NDCG@10, Faithfulness, Hallucination Rate, Citation Precision, Citation Recall, Latency (p95), and Total Cost.
- **Tabs:**
  - **Metrics:** Tabular and graphical distributions of retrieval and generation metrics.
  - **Query Inspection:** Interactive table of all benchmark queries with filtering by score thresholds.
  - **Configuration:** Complete, immutable YAML/JSON configuration viewer.
  - **Execution Logs:** Live streaming logs from worker lanes.

### 3.5 End-to-End Query Inspection Tool (`/experiments/{id}/queries/{query_id}`)
The most critical diagnostic interface in the laboratory. For a single benchmark query, displays the entire pipeline trace in five expandable panels:
1. **Original & Transformed Queries:** Displays raw user query alongside HyDE hypothetical passages or Multi-Query expansions.
2. **Initial Retrieval Matrix:** Table of retrieved chunks showing initial Dense/BM25 scores and RRF/RSN ranks.
3. **Reranked Passages:** Highlights rank shifts caused by neural cross-encoders (e.g. "Chunk #4 promoted to Rank 1").
4. **Packed Context Window:** Shows exact formatted context text with citation headers (`[Source N]`) and Lost-in-the-Middle positioning.
5. **Generation & Faithfulness Audit:** Displays generated answer, inline citations, decomposed propositions, and entailment verdicts ($C dash s$) with reasoning.

### 3.6 Multi-Run Experiment Comparison Matrix (`/compare`)
- **Configuration Selector:** Multi-select dropdown to pick 2 to 5 experiment runs.
- **Metric Comparison Table:** Side-by-side values with delta indicators ($\Delta$) and statistical significance markers ($p < 0.01$).
- **Scatter Plot Visualizer:** Interactive charts plotting quality vs cost (e.g. Faithfulness vs Cost per 1K Queries) and quality vs latency (e.g. MRR@10 vs Latency p95).

---

## 4. UX State Handling

- **Loading States:** Skeleton loaders for data tables; spinner with progress bar showing completed queries ($M / N$) during active runs.
- **Error States:** Informative error banners specifying component name, error classification, and stack trace in collapsible developer drawer.
- **Empty States:** Clear calls-to-action explaining next steps (e.g. "No datasets found. Upload raw documents to begin.").

---

## 5. Visual Principles & Styling

- **Theme:** Clean, modern dark/light mode with high-contrast slate borders and monospace accents for hashes and identifiers.
- **Data Tables:** Dense layout with sticky headers, sortable columns, and numerical alignment.
- **Badges:** Semantically colored status badges (Green = Completed, Amber = Running, Red = Failed, Slate = Draft).
- **Responsive Target:** Optimized for Desktop/Laptop workstations (min-width 1280px); functional on tablets.
