# RAGBench — Retrieval-Augmented Generation Evaluation Laboratory

**A Parameterizable Scientific Benchmarking Platform for RAG Pipelines**

RAGBench is an enterprise and academic-grade evaluation laboratory that treats Retrieval-Augmented Generation (RAG) pipelines as parameterizable scientific experiments. Rather than serving as a static question-answering assistant, RAGBench systematically benchmarks variations across chunking strategies, embedding models, retrieval topologies, neural rerankers, query transformations, and generation models.

---

## Governance & Architecture Documentation

This repository is governed by four core architectural and engineering specifications:

| Document | Description |
| :--- | :--- |
| **[`SYSTEM_ARCHITECTURE.md`](./SYSTEM_ARCHITECTURE.md)** | System design, mathematical formalization of IR/Generation metrics, data contracts (`experiment.yaml`, `result.json`), and storage topology |
| **[`PHASED_IMPLEMENTATION_ROADMAP.md`](./PHASED_IMPLEMENTATION_ROADMAP.md)** | Step-by-step engineering tasks from Phase 0 to Phase M with strict verification gates |
| **[`EVALUATION_METHODOLOGY_RQ.md`](./EVALUATION_METHODOLOGY_RQ.md)** | Formal academic experimental design, RQ1–RQ6 research questions, benchmark dataset specifications, and JSONL schemas |
| **[`AGENT_EXECUTION_PROTOCOL.md`](./AGENT_EXECUTION_PROTOCOL.md)** | Hard guardrails, CLI rules, quality gates, and error handling for autonomous coding agents |

---

## Key Capabilities & Experimental Matrix

- **Chunking Strategies:** Fixed Token, Recursive Character, Sentence Boundary, Semantic Similarity.
- **Embedding Representations:** FastEmbed (ONNX CPU), Sentence Transformers (PyTorch), Cloud Embeddings (LiteLLM).
- **Retrieval Topologies:** Dense Vector Search (Qdrant), Lexical BM25 (Rank-BM25 / Tantivy), Reciprocal Rank Fusion (RRF), Relative Score Normalization (RSN).
- **Neural Rerankers:** Cross-Encoders (`ms-marco-MiniLM-L-6-v2`), FlashRank.
- **Query Transformers:** HyDE (Hypothetical Document Embeddings), Multi-Query Expansion, Step-Back Prompting.
- **Context Construction:** Sliding token budgets, Lost-in-the-Middle mitigation re-ordering.
- **Evaluation Metrics:**
  - *IR Metrics:* Recall@K, Precision@K, MRR@K, NDCG@K, Hit@K.
  - *Generation & Factuality:* Atomic statement decomposition, NLI entailment, answer relevance, citation precision/recall.
- **Multilingual Generalization:** Cross-lingual evaluation parity across English and Bengali.

---

## Roadmap Overview

- **Phase 0:** Foundations, Toolchain & Directory Scaffold
- **Phase A:** Document Ingestion, Parsing & Deterministic Chunking Engine
- **Phase B:** Dual-Mode Embedding & Vector Store Storage
- **Phase C:** Retrieval Engine Matrix (Dense, BM25 & Hybrid Fusion)
- **Phase D:** Query Transformations & Context Formulation
- **Phase E:** Automated Metric Calculation Engine (IR & Generation)
- **Phase F:** Experiment Orchestrator & CLI Runner
- **Phase G:** Multilingual Evaluation Extension (Bengali & English)
- **Phase H:** Academic Research Workbench Frontend (React / Vite)
- **Phase M:** Production Hardening, Release Sign-Off & Whitepaper Export
