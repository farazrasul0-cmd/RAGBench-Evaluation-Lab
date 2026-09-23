"""Evaluation metrics package for Information Retrieval and Generation factuality."""

from app.engine.metrics.citation import (
    CitationMetricsResult,
    evaluate_citations,
    extract_citations,
)
from app.engine.metrics.generation import (
    BaseEntailmentClassifier,
    FaithfulnessResult,
    LLMEntailmentClassifier,
    MockEntailmentClassifier,
    calculate_answer_relevance,
    decompose_propositions,
    evaluate_faithfulness,
)
from app.engine.metrics.ir import (
    IRMetricsResult,
    dcg_at_k,
    evaluate_ir_metrics,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

__all__ = [
    "BaseEntailmentClassifier",
    "CitationMetricsResult",
    "FaithfulnessResult",
    "IRMetricsResult",
    "LLMEntailmentClassifier",
    "MockEntailmentClassifier",
    "calculate_answer_relevance",
    "dcg_at_k",
    "decompose_propositions",
    "evaluate_citations",
    "evaluate_faithfulness",
    "evaluate_ir_metrics",
    "extract_citations",
    "hit_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
]
