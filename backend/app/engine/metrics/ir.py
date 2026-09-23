"""Information Retrieval (IR) evaluation metrics according to SYSTEM_ARCHITECTURE.md."""

import math
from collections.abc import Sequence

from pydantic import BaseModel


class IRMetricsResult(BaseModel):
    """Container for computed Information Retrieval metrics at cutoff K."""

    k: int
    recall: float
    precision: float
    mrr: float
    ndcg: float
    hit: float


def recall_at_k(
    retrieved: Sequence[str],
    ground_truth: Sequence[str] | set[str],
    k: int,
) -> float:
    """Recall@K = |Retrieved[:K] ∩ GroundTruth| / |GroundTruth|"""
    if k <= 0 or not ground_truth:
        return 0.0

    gt_set = set(ground_truth)
    hits = sum(1 for item in retrieved[:k] if item in gt_set)
    return hits / float(len(gt_set))


def precision_at_k(
    retrieved: Sequence[str],
    ground_truth: Sequence[str] | set[str],
    k: int,
) -> float:
    """Precision@K = |Retrieved[:K] ∩ GroundTruth| / K"""
    if k <= 0:
        return 0.0

    gt_set = set(ground_truth)
    hits = sum(1 for item in retrieved[:k] if item in gt_set)
    return hits / float(k)


def reciprocal_rank_at_k(
    retrieved: Sequence[str],
    ground_truth: Sequence[str] | set[str],
    k: int,
) -> float:
    """MRR@K = 1 / rank of first relevant item in retrieved[:k], else 0.0."""
    if k <= 0 or not ground_truth:
        return 0.0

    gt_set = set(ground_truth)
    for idx, item in enumerate(retrieved[:k], start=1):
        if item in gt_set:
            return 1.0 / float(idx)
    return 0.0


def dcg_at_k(relevance_scores: Sequence[float], k: int) -> float:
    """Discounted Cumulative Gain: DCG@K = sum_{i=1}^K (2^{r_i} - 1) / log2(i + 1)."""
    dcg = 0.0
    for idx, score in enumerate(relevance_scores[:k], start=1):
        gain = (math.pow(2.0, score) - 1.0) / math.log2(idx + 1.0)
        dcg += gain
    return dcg


def ndcg_at_k(
    retrieved: Sequence[str],
    ground_truth_relevance: dict[str, float] | Sequence[str] | set[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain (NDCG@K) supporting binary and graded relevance."""
    if k <= 0:
        return 0.0

    if isinstance(ground_truth_relevance, dict):
        rel_map = ground_truth_relevance
    else:
        # Binary relevance: 1.0 if present in ground truth set, else 0.0
        rel_map = dict.fromkeys(ground_truth_relevance, 1.0)

    if not rel_map or all(v <= 0.0 for v in rel_map.values()):
        return 0.0

    # Actual gains for retrieved items
    actual_gains = [rel_map.get(item, 0.0) for item in retrieved[:k]]
    actual_dcg = dcg_at_k(actual_gains, k)

    # Ideal gains sorted in descending order
    ideal_gains = sorted(rel_map.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal_gains, k)

    if ideal_dcg <= 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def hit_at_k(
    retrieved: Sequence[str],
    ground_truth: Sequence[str] | set[str],
    k: int,
) -> float:
    """Hit@K = 1.0 if any relevant item is found in retrieved[:K], else 0.0."""
    if k <= 0 or not ground_truth:
        return 0.0

    gt_set = set(ground_truth)
    for item in retrieved[:k]:
        if item in gt_set:
            return 1.0
    return 0.0


def evaluate_ir_metrics(
    retrieved: Sequence[str],
    ground_truth: Sequence[str] | set[str],
    k: int = 10,
    graded_relevance: dict[str, float] | None = None,
) -> IRMetricsResult:
    """Calculate comprehensive IR metrics bundle at cutoff K."""
    return IRMetricsResult(
        k=k,
        recall=recall_at_k(retrieved, ground_truth, k),
        precision=precision_at_k(retrieved, ground_truth, k),
        mrr=reciprocal_rank_at_k(retrieved, ground_truth, k),
        ndcg=ndcg_at_k(retrieved, graded_relevance or ground_truth, k),
        hit=hit_at_k(retrieved, ground_truth, k),
    )
