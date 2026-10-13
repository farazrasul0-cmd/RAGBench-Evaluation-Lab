# RAGBench Research Workbench (Frontend)

Interactive visual workbench for exploring, analyzing, and validating Retrieval-Augmented Generation experiments.

Built with **React 19**, **TypeScript** (strict mode), **Tailwind CSS v4**, and **Vite**.

---

## Views & Capabilities

1. **Matrix Builder (`MatrixBuilderView`):**
   * Configures factorial sweeps across chunking strategies, embeddings, retrieval top-k, and rerankers.
   * Dynamically estimates combinatorial space and generates reproducible execution specifications.
2. **Pareto Frontier (`CompareDashboardView`):**
   * Plots accuracy vs. retrieval latency trade-offs.
   * Enforces canonical control signature matching to guarantee valid comparisons.
3. **Metric Radar (`CompareDashboardView`):**
   * Multi-axial comparison across Recall@K, Precision@K, MRR@K, NDCG@K, and Faithfulness.
4. **Citation Trace Inspector (`TraceInspectorView`):**
   * Granular claim-level context attribution inspector distinguishing supported claims from unresolvable provenance.

---

## Development & Testing

```bash
# Install dependencies
npm ci || npm install

# Start development server (http://localhost:5173)
npm run dev

# Run Vitest test suite
npm run test:run

# Run Oxlint static analysis
npm run lint

# Run strict TypeScript compiler check
npm run typecheck

# Build production distribution
npm run build
```
