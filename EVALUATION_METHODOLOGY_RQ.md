# Empirical Evaluation Methodology & Research Questions (RQ1–RQ6)

**Purpose:** This document specifies the scientific protocol for running formal benchmark experiments using RAGBench. This methodology ensures results are statistically rigorous and suitable for academic theses, publications, and enterprise whitepapers.

---

## 1. Formal Research Questions

### RQ1: Chunking Strategy & Boundary Dynamics
* **Question:** How do chunk sizing (256 vs 512 vs 1024 tokens) and chunk boundary logic (Fixed-token vs Recursive vs Semantic distance) impact retrieval recall and downstream answer faithfulness?
* **Hypothesis:** Semantic and recursive chunking will yield statistically significant gains in retrieval precision and answer faithfulness over fixed-size windowing on technical texts ($p < 0.01$).

### RQ2: Dense vs. Lexical vs. Hybrid Fusion (RRF)
* **Question:** Does hybrid retrieval (Dense vector + BM25) with Reciprocal Rank Fusion outperform single-representation dense retrieval in domain-specific technical documentation containing unique entity names and code symbols?
* **Hypothesis:** Dense retrieval alone degrades in keyword-heavy technical queries (e.g., API symbols, CVE identifiers), where Hybrid RRF achieves $\ge 15\%$ higher Recall@5.

### RQ3: Marginal Utility of Cross-Encoder Reranking
* **Question:** What is the quantitative trade-off between retrieval quality improvement and latency cost when applying cross-encoder neural rerankers after initial retrieval?
* **Hypothesis:** Reranking increases MRR@5 by $\ge 18\%$, but introduces a $3\times$ latency penalty during inference.

### RQ4: Context Window Depth vs. Faithfulness Trade-Off
* **Question:** Does increasing the context budget (Top-3 vs Top-5 vs Top-10 vs Top-20 chunks) monotonically improve generation quality, or does it introduce noise that impairs answer faithfulness?
* **Hypothesis:** Answer faithfulness peaks between Top-5 and Top-10 chunks; expanding to Top-20 introduces "Lost-in-the-Middle" distractors, resulting in a measurable increase in hallucinated propositions.

### RQ5: Disentangling Retrieval Quality from Generation Quality
* **Question:** To what extent does high retrieval recall guarantee a factual answer? Can generation fail despite optimal retrieval?
* **Hypothesis:** A high Recall@5 is a necessary but insufficient condition for high generation faithfulness. Factuality also depends on context density and prompt instruction following.

### RQ6: Model Efficiency & Retrieval Sensitivity
* **Question:** Can a compact local model (e.g., Qwen-2.5-7B or Llama-3.2-3B) match the answer faithfulness of frontier cloud models (e.g., Claude 3.5 Sonnet / GPT-4o) when supplied with high-precision reranked context?
* **Hypothesis:** When context is refined to Top-3 highly relevant chunks via Cross-Encoder reranking, compact local models achieve $\ge 92\%$ of the faithfulness score of frontier models at $< 5\%$ of the cost.

---

## 2. Standard Benchmark Datasets

| Dataset ID | Domain | Volume | Description | Evaluation Objective |
| :--- | :--- | :--- | :--- | :--- |
| `dataset-se-docs` | Software Engineering | 50 repos / 500 QA | FastHTML, FastAPI, PyTorch docs with exact code symbols | Measures keyword precision & code block chunking |
| `dataset-academic` | Scientific Papers | 100 Papers / 800 QA | ArXiv AI/ML research papers | Tests deep reasoning, multi-paragraph context, and citations |
| `dataset-general` | General Knowledge | 1,000 articles / 1,000 QA | Wikipedia public domain articles | General baseline benchmarking |
| `dataset-bilingual` | Bengali / English | 200 docs / 400 QA | Parallel government & technical publications | Cross-lingual retrieval transfer evaluation |

---

## 3. Ground Truth Data Format (`eval_qa.jsonl`)

Every sample in the evaluation dataset must follow this strict JSONL structure:

```json
{
  "query_id": "q_academic_042",
  "query": "What architectural change enables Mamba to achieve linear computational complexity?",
  "ground_truth_answer": "Mamba achieves linear computational complexity by replacing the quadratic attention matrix of Transformers with selective State Space Models (SSMs) that process sequence data through time-invariant recurrent parameters.",
  "ground_truth_passages": [
    {
      "doc_id": "paper_mamba_v1.pdf",
      "passage_id": "chunk_048",
      "text_snippet": "By designing a selective mechanism that conditions SSM parameters on the input, we retain state-space models' linear-time scaling..."
    }
  ],
  "domain": "academic_research",
  "language": "en",
  "difficulty": "hard"
}
```

---

## 4. Scientific Protocol for Reproducibility

- **Seed Anchoring:** All stochastic components (e.g., embedding projections, synthetic generation temperatures) must have fixed seeds (`seed=42`).
- **Cold-Start Latency Exclusion:** In latency benchmarking, the first 3 queries of any run are treated as cache warm-up and dropped from aggregate statistics.
- **Statistical Significance Testing:** All reported comparisons between configurations must include two-tailed paired Student's $t$-tests ($p$-values) and $95\%$ confidence intervals.
