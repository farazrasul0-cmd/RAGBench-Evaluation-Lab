/**
 * Core Data Models for RAGBench Evaluation Laboratory
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
