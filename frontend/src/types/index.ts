/**
 * Core Data Models for RAGBench Evaluation Laboratory (Phase H Academic Workbench)
 */

export interface AggregateMetrics {
  retrieval: {
    recall_at_5: number
    recall_at_10: number
    precision_at_5: number
    mrr_at_10: number
    ndcg_at_10: number
  }
  generation: {
    faithfulness: number
    answer_relevance: number
    citation_precision: number
    citation_recall: number
    hallucination_rate: number
  }
  system: {
    latency_retrieval_mean_ms: number
    latency_rerank_mean_ms: number
    latency_generation_mean_ms: number
    latency_e2e_p95_ms: number
    tokens_prompt_mean: number
    tokens_completion_mean: number
    estimated_cost_usd_per_1k_queries: number
  }
}

export interface EvaluationRunResult {
  run_id: string
  experiment_name: string
  configuration_hash: string
  timestamp: string
  pipeline_parameters: Record<string, unknown>
  aggregate_metrics: AggregateMetrics
}

export type WorkbenchTab = 'overview' | 'matrix' | 'compare' | 'trace'

/**
 * Matrix Builder Parameter Selection
 */
export interface MatrixSweepSelection {
  name: string
  description: string
  dataset_id: string
  dataset_version_id: string
  chunking_strategies: string[]
  chunk_sizes: number[]
  chunk_overlaps: number[]
  embedding_models: string[]
  retrieval_strategies: string[]
  rerankers: string[]
  generation_models: string[]
  top_k_values: number[]
}

export interface MatrixConfigurationPoint {
  index: number
  chunking: { strategy: string; chunk_size: number; chunk_overlap: number }
  embedding: { model_name: string; dimension: number }
  retrieval: { strategy: string; top_k: number }
  reranker: { enabled: boolean; strategy: string }
  generation: { model_name: string }
}

export interface MatrixPreviewData {
  total_combinations: number
  complexity_category: 'LOW' | 'MODERATE' | 'HEAVY'
  estimated_queries_per_run: number
  total_pipeline_points: number
  sample_configurations: MatrixConfigurationPoint[]
}

/**
 * Comparative Dashboard Models (Pareto & Radar)
 */
export interface ParetoPoint {
  run_id: string
  name: string
  strategy: 'dense' | 'bm25' | 'hybrid' | string
  modality?: string
  recall_at_5: number
  latency_ms: number
  estimated_cost_usd: number
  ndcg_at_5: number
  is_pareto_optimal: boolean
}

export interface RadarMetricData {
  run_id: string
  name: string
  color: string
  metrics: {
    recall_at_10: number
    precision_at_5: number
    faithfulness: number
    citation_accuracy: number
    cost_efficiency: number // 0.0 to 1.0
  }
}

export interface ComparisonRunSummary {
  run_id: string
  experiment_id: string
  name: string
  strategy: string
  modality: string
  recall_at_5: number
  mrr_at_5: number
  ndcg_at_5: number
  latency_ms: number
  faithfulness?: number
  citation_precision?: number
}

/**
 * Per-Question Trace Inspector Models
 */
export interface RetrievedChunkView {
  chunk_id: string
  doc_id: string
  chunk_index: number
  score: number
  rank: number
  content: string
  start_char?: number
  end_char?: number
}

export interface SpanAnnotation {
  text: string
  is_entailing: boolean
  citation_ids: string[]
}

export interface QueryTraceDetail {
  query_run_id: string
  query_id: string
  original_query: string
  expected_answer: string
  ground_truth_chunks: string[]
  domain: string
  language: string
  modality: string
  latency_ms: number
  status: string
  retrieved_chunks: RetrievedChunkView[]
  generation_output: string
  annotated_spans: SpanAnnotation[]
  metric_scores: {
    recall_at_5: number
    mrr_at_5: number
    ndcg_at_5: number
    precision_at_5: number
    faithfulness: number
    citation_precision: number
  }
}
