"""Unit tests and mathematical verification for IR, Faithfulness, and Citation metrics."""

import math

import pytest

from app.engine.embeddings.mock_provider import DeterministicMockEmbeddingProvider
from app.engine.metrics.citation import evaluate_citations, extract_citations
from app.engine.metrics.generation import (
    MockEntailmentClassifier,
    calculate_answer_relevance,
    decompose_propositions,
    evaluate_faithfulness,
)
from app.engine.metrics.ir import (
    evaluate_ir_metrics,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from app.engine.query_transforms.base import MockLLMClient

# =========================================================================
# 1. Information Retrieval Metrics Tests
# =========================================================================


def test_ir_metrics_mathematical_conformance() -> None:
    """Manual pen-and-paper verification:

    Retrieved: ["d1", "d2", "d3", "d4", "d5"]
    Ground Truth: {"d2", "d4"} (2 relevant items)
    Cutoff K = 3
    Retrieved[:3] = ["d1", "d2", "d3"]
    Hits: {"d2"} -> 1 hit
    - Recall@3 = 1 / 2 = 0.5
    - Precision@3 = 1 / 3
    - MRR@3 = 1 / 2 = 0.5 (first hit at rank 2)
    - Hit@3 = 1.0
    - Binary NDCG@3:
      DCG@3 = (2^0 - 1)/log2(2) + (2^1 - 1)/log2(3) + (2^0 - 1)/log2(4)
            = 1.0 / log2(3) = 0.63092975
      Ideal ranking: ["d2", "d4", "d_other"] -> gains [1.0, 1.0, 0.0]
      IDCG@3 = (2^1 - 1)/log2(2) + (2^1 - 1)/log2(3)
             = 1.0 + 1.0 / log2(3) = 1.63092975
      NDCG@3 = DCG@3 / IDCG@3 = 0.63092975 / 1.63092975 = 0.3868528
    """
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    gt = {"d2", "d4"}
    k = 3

    r = recall_at_k(retrieved, gt, k)
    p = precision_at_k(retrieved, gt, k)
    mrr = reciprocal_rank_at_k(retrieved, gt, k)
    hit = hit_at_k(retrieved, gt, k)
    ndcg = ndcg_at_k(retrieved, gt, k)

    assert r == 0.5
    assert pytest.approx(p, rel=1e-6) == (1.0 / 3.0)
    assert mrr == 0.5
    assert hit == 1.0

    expected_dcg = 1.0 / math.log2(3.0)
    expected_idcg = 1.0 + (1.0 / math.log2(3.0))
    expected_ndcg = expected_dcg / expected_idcg
    assert pytest.approx(ndcg, rel=1e-6) == expected_ndcg


def test_ir_metrics_edge_cases_and_zero_division() -> None:
    # 1. Empty ground truth
    assert recall_at_k(["d1"], [], k=5) == 0.0
    assert precision_at_k(["d1"], [], k=5) == 0.0
    assert reciprocal_rank_at_k(["d1"], [], k=5) == 0.0
    assert hit_at_k(["d1"], [], k=5) == 0.0
    assert ndcg_at_k(["d1"], [], k=5) == 0.0

    # 2. Empty retrieved list
    assert recall_at_k([], ["d1"], k=5) == 0.0
    assert precision_at_k([], ["d1"], k=5) == 0.0
    assert reciprocal_rank_at_k([], ["d1"], k=5) == 0.0
    assert hit_at_k([], ["d1"], k=5) == 0.0
    assert ndcg_at_k([], ["d1"], k=5) == 0.0

    # 3. K <= 0
    assert recall_at_k(["d1"], ["d1"], k=0) == 0.0
    assert precision_at_k(["d1"], ["d1"], k=-1) == 0.0

    # 4. Zero overlap
    assert recall_at_k(["d1", "d2"], ["d3"], k=2) == 0.0
    assert reciprocal_rank_at_k(["d1", "d2"], ["d3"], k=2) == 0.0


def test_graded_ndcg() -> None:
    graded_rel = {"d1": 3.0, "d2": 2.0, "d3": 1.0, "d4": 0.0}

    # Perfect ranking: ["d1", "d2", "d3"]
    perfect_ndcg = ndcg_at_k(["d1", "d2", "d3"], graded_rel, k=3)
    assert pytest.approx(perfect_ndcg, rel=1e-6) == 1.0

    # Sub-optimal ranking: ["d3", "d2", "d1"]
    subopt_ndcg = ndcg_at_k(["d3", "d2", "d1"], graded_rel, k=3)
    assert 0.0 < subopt_ndcg < 1.0


def test_evaluate_ir_metrics_bundle() -> None:
    retrieved = ["a", "b", "c"]
    gt = ["b"]
    result = evaluate_ir_metrics(retrieved, gt, k=2)
    assert result.k == 2
    assert result.recall == 1.0
    assert result.precision == 0.5
    assert result.mrr == 0.5
    assert result.hit == 1.0


# =========================================================================
# 2. Generation & Faithfulness Metrics Tests
# =========================================================================


def test_proposition_decomposition() -> None:
    text = "FastAPI is fast. Python is dynamically typed! Is PostgreSQL ACID compliant?"
    props = decompose_propositions(text)
    assert len(props) == 3
    assert "FastAPI is fast" in props[0]
    assert "Python is dynamically typed" in props[1]
    assert "PostgreSQL" in props[2]


def test_faithfulness_prelabeled_ten_claims() -> None:
    """Synthetic test: 10 pre-labeled claim-evidence pairs checking exact bounds."""
    entailment_map = {
        "claim_true_1": True,
        "claim_true_2": True,
        "claim_true_3": True,
        "claim_true_4": True,
        "claim_true_5": True,
        "claim_false_1": False,
        "claim_false_2": False,
        "claim_false_3": False,
        "claim_false_4": False,
        "claim_false_5": False,
    }
    classifier = MockEntailmentClassifier(entailment_map=entailment_map)
    context = "Reference technical documentation context."

    # 1. Perfect entailment (5 true claims) -> Faithfulness 100%, Hallucination 0%
    true_answer = "claim_true_1. claim_true_2. claim_true_3. claim_true_4. claim_true_5."
    res_perfect = evaluate_faithfulness(true_answer, context, classifier)
    assert res_perfect.faithfulness_score == 1.0
    assert res_perfect.hallucination_rate == 0.0
    assert res_perfect.entailed_propositions_count == 5

    # 2. Complete hallucination (5 false claims) -> Faithfulness 0%, Hallucination 100%
    false_answer = "claim_false_1. claim_false_2. claim_false_3. claim_false_4. claim_false_5."
    res_zero = evaluate_faithfulness(false_answer, context, classifier)
    assert res_zero.faithfulness_score == 0.0
    assert res_zero.hallucination_rate == 1.0
    assert res_zero.entailed_propositions_count == 0

    # 3. Mixed: 3 true, 2 false -> Faithfulness 60%, Hallucination 40%
    mixed_answer = "claim_true_1. claim_true_2. claim_true_3. claim_false_1. claim_false_2."
    res_mixed = evaluate_faithfulness(mixed_answer, context, classifier)
    assert pytest.approx(res_mixed.faithfulness_score, rel=1e-6) == 0.6
    assert pytest.approx(res_mixed.hallucination_rate, rel=1e-6) == 0.4
    assert res_mixed.total_propositions == 5


def test_faithfulness_empty_answer() -> None:
    classifier = MockEntailmentClassifier()
    res = evaluate_faithfulness("", "Context", classifier)
    assert res.faithfulness_score == 1.0
    assert res.hallucination_rate == 0.0
    assert res.total_propositions == 0


def test_answer_relevance() -> None:
    provider = DeterministicMockEmbeddingProvider(dimension=16)
    llm = MockLLMClient(
        canned_responses={
            "distinct questions that the answer addresses": (
                "What is FastAPI?\n"
                "Why is FastAPI high performance?\n"
                "What web framework uses Python type hints?\n"
            )
        }
    )
    score = calculate_answer_relevance(
        answer="FastAPI is a high performance Python web framework.",
        original_query="What is FastAPI web framework performance?",
        embedding_provider=provider,
        llm_client=llm,
        num_synthetic_queries=3,
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# =========================================================================
# 3. Citation Precision & Recall Tests
# =========================================================================


def test_extract_citations() -> None:
    text = "FastAPI uses Pydantic [Source 1] and Starlette [Source 2]. Other details in [3]."
    cits = extract_citations(text)
    assert cits == [1, 2, 3]


def test_citation_precision_and_recall() -> None:
    context_sources = {
        1: "FastAPI is built on Starlette and Pydantic.",
        2: "PostgreSQL is an ACID database supporting transactions.",
    }
    classifier = MockEntailmentClassifier()

    # Statement 1 correctly cites [Source 1]; Statement 2 correctly cites [Source 2]
    good_answer = "FastAPI uses Starlette [Source 1]. PostgreSQL supports transactions [Source 2]."
    res_good = evaluate_citations(good_answer, context_sources, classifier)
    assert res_good.total_citations == 2
    assert res_good.citation_precision == 1.0
    assert res_good.citation_recall == 1.0

    # Bad citation: Statement cites wrong source [Source 2] for FastAPI
    bad_answer = "FastAPI uses Starlette [Source 2]."
    res_bad = evaluate_citations(bad_answer, context_sources, classifier)
    assert res_bad.total_citations == 1
    assert res_bad.valid_citations == 0
    assert res_bad.citation_precision == 0.0
    assert res_bad.citation_recall == 0.0


def test_citation_edge_cases() -> None:
    classifier = MockEntailmentClassifier()
    # No citations in answer
    no_cit_answer = "FastAPI is a modern web framework."
    res_no_cit = evaluate_citations(no_cit_answer, {1: "context"}, classifier)
    assert res_no_cit.total_citations == 0
    assert res_no_cit.citation_precision == 0.0
    assert res_no_cit.citation_recall == 0.0

    # Empty answer
    res_empty = evaluate_citations("", {}, classifier)
    assert res_empty.citation_precision == 0.0
    assert res_empty.citation_recall == 0.0


# =========================================================================
# 4. Corrective Audit Regression Tests (Duplicate IDs & Nonexistent Sources)
# =========================================================================


def test_duplicate_retrieval_regression() -> None:
    """Regression tests verifying duplicate retrieved IDs do not inflate metrics."""
    # 1. Precision@2 with duplicate retrieved items: denominator is 2, distinct hits is 1
    p2 = precision_at_k(["A", "A"], {"A"}, 2)
    assert p2 == 0.5

    # 2. Recall@2 with duplicate retrieved items: distinct hits is 1, total relevant is 1
    r2 = recall_at_k(["A", "A"], {"A"}, 2)
    assert r2 == 1.0
    assert r2 <= 1.0

    # 3. NDCG@3 with duplicate retrieved items: only first 'A' gets relevance gain
    ndcg3 = ndcg_at_k(["A", "A", "A"], {"A": 1.0}, 3)
    assert ndcg3 == 1.0
    assert ndcg3 <= 1.0

    # 4. Mixed ranking where duplicates occur before and after other relevant items
    # retrieved = ["A", "A", "B"], ground_truth = {"A": 1.0, "B": 1.0}, k = 3
    # Actual gains: [1.0, 0.0, 1.0] -> DCG@3 = 1.0/log2(2) + 0 + 1.0/log2(4) = 1.0 + 0.5 = 1.5
    # Ideal gains: [1.0, 1.0] -> IDCG@3 = 1.0/log2(2) + 1.0/log2(3) = 1.0 + 1/log2(3)
    mixed_ndcg = ndcg_at_k(["A", "A", "B"], {"A": 1.0, "B": 1.0}, 3)
    expected_mixed_dcg = 1.5
    expected_mixed_idcg = 1.0 + (1.0 / math.log2(3.0))
    expected_mixed_ndcg = expected_mixed_dcg / expected_mixed_idcg
    assert pytest.approx(mixed_ndcg, rel=1e-6) == expected_mixed_ndcg
    assert 0.0 <= mixed_ndcg <= 1.0


def test_citation_nonexistent_source_id_regression() -> None:
    """Verify citations to nonexistent source IDs are safely handled and not credited."""
    classifier = MockEntailmentClassifier()
    context_sources = {1: "FastAPI is a modern asynchronous framework."}

    # Answer cites [Source 99], which is absent from context_sources
    answer = "FastAPI offers automated OpenAPI docs [Source 99]."
    result = evaluate_citations(answer, context_sources, classifier)

    assert result.total_citations == 1
    assert result.valid_citations == 0
    assert result.citation_precision == 0.0
    assert result.citation_recall == 0.0
    assert len(result.citation_mappings) == 1
    mapping = result.citation_mappings[0]
    assert mapping["source_id"] == 99
    assert mapping["is_entailed"] is False
    assert mapping["reason"] == "Source ID not found in context"


def test_independent_mathematical_fixtures() -> None:
    """Hand-calculated mathematical verification fixtures independently tested."""
    # Fixture A: retrieved = ["A", "A"], relevant = {"A"}, K = 2
    res_a = evaluate_ir_metrics(["A", "A"], {"A"}, k=2)
    assert res_a.precision == 0.5
    assert res_a.recall == 1.0
    assert res_a.mrr == 1.0
    assert res_a.hit == 1.0
    assert res_a.ndcg == 1.0

    # Fixture B: retrieved = ["X", "A", "A"], relevant = {"A"}, K = 3
    # DCG@3 = 0.0 + (2^1 - 1)/log2(3) + 0.0 = 1.0 / log2(3)
    # IDCG@3 = (2^1 - 1)/log2(2) = 1.0
    # NDCG@3 = (1.0 / log2(3)) / 1.0 = 1.0 / log2(3) ? 0.63092975
    res_b = evaluate_ir_metrics(["X", "A", "A"], {"A"}, k=3)
    assert pytest.approx(res_b.precision, rel=1e-6) == (1.0 / 3.0)
    assert res_b.recall == 1.0
    assert res_b.mrr == 0.5
    assert res_b.hit == 1.0
    expected_ndcg_b = 1.0 / math.log2(3.0)
    assert pytest.approx(res_b.ndcg, rel=1e-6) == expected_ndcg_b

    # Fixture C: Graded relevance duplicate case
    # retrieved = ["A", "A", "B"], ground_truth = {"A": 3.0, "B": 1.0}, K = 3
    # Gains: Rank 1 ("A") -> (2^3 - 1)/log2(2) = 7.0
    #        Rank 2 ("A", duplicate) -> 0.0
    #        Rank 3 ("B") -> (2^1 - 1)/log2(4) = 0.5
    # Total DCG@3 = 7.5
    # Ideal gains sorted: [3.0, 1.0] -> IDCG@3 = 7.0/log2(2) + 1.0/log2(3) = 7.0 + 1/log2(3)
    # Expected NDCG@3 = 7.5 / (7.0 + 1/log2(3)) ? 0.982842
    graded_rel = {"A": 3.0, "B": 1.0}
    ndcg_c = ndcg_at_k(["A", "A", "B"], graded_rel, k=3)
    expected_dcg_c = 7.5
    expected_idcg_c = 7.0 + (1.0 / math.log2(3.0))
    expected_ndcg_c = expected_dcg_c / expected_idcg_c
    assert pytest.approx(ndcg_c, rel=1e-6) == expected_ndcg_c
    assert 0.0 <= ndcg_c <= 1.0
