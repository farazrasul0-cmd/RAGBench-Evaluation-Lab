"""Unit tests and mathematical verification for RRF and RSN score fusion."""

import time

import pytest

from app.engine.retrievers.hybrid import (
    HybridRetrieverError,
    reciprocal_rank_fusion,
    relative_score_normalization,
)
from app.schemas.chunk import DocumentChunk


@pytest.fixture
def fixed_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk.create(
            doc_id="d1",
            chunk_index=0,
            content="Document 1 content",
            token_count=3,
            start_char=0,
            end_char=18,
            strategy="fixed",
            chunk_id="chunk-1",
        ),
        DocumentChunk.create(
            doc_id="d2",
            chunk_index=0,
            content="Document 2 content",
            token_count=3,
            start_char=0,
            end_char=18,
            strategy="fixed",
            chunk_id="chunk-2",
        ),
        DocumentChunk.create(
            doc_id="d3",
            chunk_index=0,
            content="Document 3 content",
            token_count=3,
            start_char=0,
            end_char=18,
            strategy="fixed",
            chunk_id="chunk-3",
        ),
        DocumentChunk.create(
            doc_id="d4",
            chunk_index=0,
            content="Document 4 content",
            token_count=3,
            start_char=0,
            end_char=18,
            strategy="fixed",
            chunk_id="chunk-4",
        ),
    ]


def test_rrf_mathematical_conformance(fixed_chunks: list[DocumentChunk]) -> None:
    c1, c2, c3, c4 = fixed_chunks
    # Ranking 1: [c1 (rank 1), c2 (rank 2), c3 (rank 3)]
    # Ranking 2: [c2 (rank 1), c1 (rank 2), c4 (rank 3)]
    ranking_dense = [(c1, 0.95), (c2, 0.85), (c3, 0.70)]
    ranking_bm25 = [(c2, 12.5), (c1, 10.0), (c4, 8.2)]

    k = 60
    fused = reciprocal_rank_fusion(
        rankings=[ranking_dense, ranking_bm25],
        k=k,
        top_k=4,
    )

    # Manual mathematical expectation:
    # RRF(c1) = 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    # RRF(c2) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61
    # RRF(c3) = 1/(60+3) = 1/63
    # RRF(c4) = 1/(60+3) = 1/63
    expected_c1_c2 = (1.0 / 61.0) + (1.0 / 62.0)
    expected_c3_c4 = 1.0 / 63.0

    scores_by_id = {chunk.chunk_id: score for chunk, score in fused}

    assert pytest.approx(scores_by_id["chunk-1"], rel=1e-6) == expected_c1_c2
    assert pytest.approx(scores_by_id["chunk-2"], rel=1e-6) == expected_c1_c2
    assert pytest.approx(scores_by_id["chunk-3"], rel=1e-6) == expected_c3_c4
    assert pytest.approx(scores_by_id["chunk-4"], rel=1e-6) == expected_c3_c4


def test_rrf_deduplication(fixed_chunks: list[DocumentChunk]) -> None:
    c1, c2, _, _ = fixed_chunks
    ranking1 = [(c1, 0.9), (c2, 0.8)]
    ranking2 = [(c1, 0.85), (c2, 0.75)]
    ranking3 = [(c1, 0.8), (c2, 0.7)]

    fused = reciprocal_rank_fusion([ranking1, ranking2, ranking3], k=60, top_k=10)
    chunk_ids = [c.chunk_id for c, _ in fused]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert len(chunk_ids) == 2


def test_rsn_mathematical_conformance(fixed_chunks: list[DocumentChunk]) -> None:
    c1, c2, c3, _ = fixed_chunks
    dense_results = [(c1, 10.0), (c2, 5.0), (c3, 0.0)]
    bm25_results = [(c1, 20.0), (c2, 10.0), (c3, 0.0)]

    # Normalized scores:
    # c1: dense=(10-0)/10 = 1.0, bm25=(20-0)/20 = 1.0 -> combined = 0.5*1.0 + 0.5*1.0 = 1.0
    # c2: dense=(5-0)/10 = 0.5, bm25=(10-0)/20 = 0.5 -> combined = 0.5*0.5 + 0.5*0.5 = 0.5
    # c3: dense=(0-0)/10 = 0.0, bm25=(0-0)/20 = 0.0 -> combined = 0.0
    fused = relative_score_normalization(
        dense_results=dense_results,
        bm25_results=bm25_results,
        alpha=0.5,
        top_k=3,
    )
    scores = {c.chunk_id: s for c, s in fused}
    assert pytest.approx(scores["chunk-1"], rel=1e-4) == 1.0
    assert pytest.approx(scores["chunk-2"], rel=1e-4) == 0.5
    assert pytest.approx(scores["chunk-3"], abs=1e-4) == 0.0


def test_rsn_invalid_alpha(fixed_chunks: list[DocumentChunk]) -> None:
    c1, _, _, _ = fixed_chunks
    with pytest.raises(HybridRetrieverError):
        relative_score_normalization([(c1, 1.0)], [(c1, 1.0)], alpha=1.5)


def test_hybrid_fusion_latency_benchmark(fixed_chunks: list[DocumentChunk]) -> None:
    """Assert that score fusion adds < 10ms overhead."""
    # Synthesize two 100-candidate rankings
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

    # Warmup
    _ = reciprocal_rank_fusion([dense_candidates, bm25_candidates], k=60, top_k=20)

    # Benchmark RRF
    start_time = time.perf_counter()
    iterations = 50
    for _ in range(iterations):
        _ = reciprocal_rank_fusion([dense_candidates, bm25_candidates], k=60, top_k=20)
    duration_per_run_ms = ((time.perf_counter() - start_time) / iterations) * 1000.0

    # Verification gate requirement: hybrid fusion adds < 10ms overhead
    msg = f"RRF latency {duration_per_run_ms:.2f}ms exceeded 10ms limit"
    assert duration_per_run_ms < 10.0, msg
