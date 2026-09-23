"""Unit tests and rigorous behavioral verification for RRF and RSN score fusion."""

import time

import pytest

from app.engine.retrievers.hybrid import (
    reciprocal_rank_fusion,
    relative_score_normalization,
)
from app.schemas.chunk import DocumentChunk


@pytest.fixture
def test_chunks() -> dict[str, DocumentChunk]:
    chunks: dict[str, DocumentChunk] = {}
    for letter in ["A", "B", "C", "D", "E", "X", "Y"]:
        chunks[letter] = DocumentChunk.create(
            doc_id=f"doc_{letter}",
            chunk_index=0,
            content=f"Content for candidate passage {letter}",
            token_count=5,
            start_char=0,
            end_char=35,
            strategy="test",
            chunk_id=f"chunk-{letter}",
            metadata={"tag": letter, "depth": 1},
        )
    return chunks


def test_supervisor_example_both_channels(test_chunks: dict[str, DocumentChunk]) -> None:
    """Exact Supervisor Specification:

    Given:
      query = Q
      Dense returns: A, B, C
      BM25 returns: B, D, E
      Hybrid RRF should return: B before A/C/D/E
    """
    ch = test_chunks
    dense_ranking = [(ch["A"], 0.90), (ch["B"], 0.80), (ch["C"], 0.70)]
    bm25_ranking = [(ch["B"], 15.0), (ch["D"], 12.0), (ch["E"], 9.0)]

    fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=60, top_k=5)

    assert len(fused) == 5
    top_chunk, _ = fused[0]
    # B must be strictly rank 1
    assert top_chunk.chunk_id == "chunk-B"

    # All returned chunk IDs
    fused_ids = [c.chunk_id for c, _ in fused]
    assert fused_ids[0] == "chunk-B"
    assert "chunk-A" in fused_ids
    assert "chunk-C" in fused_ids
    assert "chunk-D" in fused_ids
    assert "chunk-E" in fused_ids


def test_rrf_mathematical_conformance(test_chunks: dict[str, DocumentChunk]) -> None:
    ch = test_chunks
    c1, c2, c3, c4 = ch["A"], ch["B"], ch["C"], ch["D"]
    ranking_dense = [(c1, 0.95), (c2, 0.85), (c3, 0.70)]
    ranking_bm25 = [(c2, 12.5), (c1, 10.0), (c4, 8.2)]

    k = 60
    fused = reciprocal_rank_fusion(
        rankings=[ranking_dense, ranking_bm25],
        k=k,
        top_k=4,
    )

    expected_c1_c2 = (1.0 / 61.0) + (1.0 / 62.0)
    expected_c3_c4 = 1.0 / 63.0

    scores_by_id = {chunk.chunk_id: score for chunk, score in fused}

    assert pytest.approx(scores_by_id["chunk-A"], rel=1e-6) == expected_c1_c2
    assert pytest.approx(scores_by_id["chunk-B"], rel=1e-6) == expected_c1_c2
    assert pytest.approx(scores_by_id["chunk-C"], rel=1e-6) == expected_c3_c4
    assert pytest.approx(scores_by_id["chunk-D"], rel=1e-6) == expected_c3_c4


def test_rrf_tie_breaking_deterministic(test_chunks: dict[str, DocumentChunk]) -> None:
    ch = test_chunks
    c1, c2 = ch["A"], ch["B"]
    # Exactly identical rankings in reverse order -> identical score
    ranking1 = [(c1, 1.0), (c2, 0.5)]
    ranking2 = [(c2, 1.0), (c1, 0.5)]

    fused = reciprocal_rank_fusion([ranking1, ranking2], k=60, top_k=2)
    # Both have score 1/61 + 1/62. Ascending chunk_id tie-breaking: chunk-A before chunk-B
    assert fused[0][0].chunk_id == "chunk-A"
    assert fused[1][0].chunk_id == "chunk-B"


def test_rrf_smoothing_constant_sensitivity(test_chunks: dict[str, DocumentChunk]) -> None:
    """Demonstrate changing k changes relative ranking as expected."""
    ch = test_chunks
    # Document X has rank 1 in ranking 1, but rank 50 in ranking 2
    # Document Y has rank 10 in ranking 1, and rank 2 in ranking 2
    r1 = [(ch["X"], 1.0)] + [(ch["A"], 0.5)] * 8 + [(ch["Y"], 0.2)]
    r2 = [(ch["A"], 1.0), (ch["Y"], 0.9)] + [(ch["B"], 0.5)] * 47 + [(ch["X"], 0.1)]

    # With k=1:
    # RRF(X) = 1/(1+1) + 1/(1+50) = 0.5 + 0.0196 = 0.5196
    # RRF(Y) = 1/(1+10) + 1/(1+2) = 0.0909 + 0.3333 = 0.4242 -> X wins
    fused_k1 = reciprocal_rank_fusion([r1, r2], k=1, top_k=5)
    winner_k1 = [c.chunk_id for c, _ in fused_k1 if c.chunk_id in ("chunk-X", "chunk-Y")][0]
    assert winner_k1 == "chunk-X"

    # With k=100:
    # RRF(X) = 1/101 + 1/150 = 0.0099 + 0.00667 = 0.01657
    # RRF(Y) = 1/110 + 1/102 = 0.00909 + 0.00980 = 0.01889 -> Y wins
    fused_k100 = reciprocal_rank_fusion([r1, r2], k=100, top_k=5)
    winner_k100 = [c.chunk_id for c, _ in fused_k100 if c.chunk_id in ("chunk-X", "chunk-Y")][0]
    assert winner_k100 == "chunk-Y"


def test_empty_and_single_channel_fusion(test_chunks: dict[str, DocumentChunk]) -> None:
    ch = test_chunks
    single_list = [(ch["A"], 0.9), (ch["B"], 0.8)]

    # 1. Dense empty, BM25 present
    fused_bm25_only = reciprocal_rank_fusion([[], single_list], k=60, top_k=2)
    assert len(fused_bm25_only) == 2
    assert fused_bm25_only[0][0].chunk_id == "chunk-A"

    # 2. Both channels empty
    assert reciprocal_rank_fusion([[], []], k=60, top_k=2) == []
    assert relative_score_normalization([], [], alpha=0.5, top_k=2) == []


def test_rsn_parameter_sensitivity(test_chunks: dict[str, DocumentChunk]) -> None:
    """Changing alpha flips winner between Dense-favored and BM25-favored candidates."""
    ch = test_chunks
    # X is top in Dense (10.0 vs 0.0 -> norm 1.0 vs 0.0)
    # Y is top in BM25 (100.0 vs 0.0 -> norm 1.0 vs 0.0)
    dense_results = [(ch["X"], 10.0), (ch["Y"], 0.0)]
    bm25_results = [(ch["Y"], 100.0), (ch["X"], 0.0)]

    # Alpha = 0.8 (Dense-heavy): X gets 0.8*1.0 + 0.2*0.0 = 0.8, Y gets 0.2
    fused_dense_heavy = relative_score_normalization(dense_results, bm25_results, alpha=0.8)
    assert fused_dense_heavy[0][0].chunk_id == "chunk-X"

    # Alpha = 0.2 (BM25-heavy): X gets 0.2*1.0 + 0.8*0.0 = 0.2, Y gets 0.8
    fused_bm25_heavy = relative_score_normalization(dense_results, bm25_results, alpha=0.2)
    assert fused_bm25_heavy[0][0].chunk_id == "chunk-Y"


def test_rsn_identical_scores_and_single_candidate(test_chunks: dict[str, DocumentChunk]) -> None:
    ch = test_chunks
    # Single candidate
    single = relative_score_normalization([(ch["A"], 5.0)], [], alpha=0.5, top_k=1)
    assert len(single) == 1
    assert pytest.approx(single[0][1], rel=1e-4) == 0.5  # 0.5 * 1.0 + 0.5 * 0.0

    # Identical positive scores
    identical = relative_score_normalization(
        [(ch["A"], 5.0), (ch["B"], 5.0)],
        [(ch["A"], 10.0), (ch["B"], 10.0)],
        alpha=0.5,
        top_k=2,
    )
    assert len(identical) == 2
    assert pytest.approx(identical[0][1], rel=1e-4) == 1.0
    assert pytest.approx(identical[1][1], rel=1e-4) == 1.0


def test_metadata_preservation_through_fusion(test_chunks: dict[str, DocumentChunk]) -> None:
    ch = test_chunks
    r1 = [(ch["A"], 0.9)]
    r2 = [(ch["A"], 10.0)]

    fused = reciprocal_rank_fusion([r1, r2], k=60, top_k=1)
    chunk, _ = fused[0]
    assert chunk.metadata == {"tag": "A", "depth": 1}
    assert chunk.doc_id == "doc_A"


def test_hybrid_fusion_latency_benchmark() -> None:
    dense_candidates: list[tuple[DocumentChunk, float]] = []
    bm25_candidates: list[tuple[DocumentChunk, float]] = []
    for i in range(100):
        ch = DocumentChunk.create(
            doc_id=f"doc_{i}",
            chunk_index=0,
            content=f"Benchmark passage content {i}",
            token_count=5,
            start_char=0,
            end_char=20,
            strategy="bench",
            chunk_id=f"bench_chunk_{i}",
        )
        dense_candidates.append((ch, 1.0 - (i * 0.005)))
        bm25_candidates.append((ch, 50.0 - (i * 0.4)))

    _ = reciprocal_rank_fusion([dense_candidates, bm25_candidates], k=60, top_k=20)

    start_time = time.perf_counter()
    iterations = 50
    for _ in range(iterations):
        _ = reciprocal_rank_fusion([dense_candidates, bm25_candidates], k=60, top_k=20)
    duration_per_run_ms = ((time.perf_counter() - start_time) / iterations) * 1000.0

    msg = f"RRF latency {duration_per_run_ms:.2f}ms exceeded 10ms limit"
    assert duration_per_run_ms < 10.0, msg
