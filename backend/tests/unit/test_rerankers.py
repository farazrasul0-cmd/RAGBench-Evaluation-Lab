"""Unit tests for passage rerankers."""

import pytest

from app.engine.rerankers import get_reranker
from app.engine.rerankers.flashrank import FlashRankReranker
from app.engine.rerankers.mock_reranker import DeterministicMockReranker
from app.schemas.chunk import DocumentChunk


@pytest.fixture
def candidates() -> list[DocumentChunk]:
    return [
        DocumentChunk.create(
            doc_id="d1",
            chunk_index=0,
            content="Bananas are rich in potassium and healthy nutrients.",
            token_count=9,
            start_char=0,
            end_char=52,
            strategy="fixed",
            chunk_id="chunk-banana",
        ),
        DocumentChunk.create(
            doc_id="d2",
            chunk_index=0,
            content="PostgreSQL supports ACID transactions, foreign keys, and complex queries.",
            token_count=10,
            start_char=0,
            end_char=73,
            strategy="fixed",
            chunk_id="chunk-postgres",
        ),
        DocumentChunk.create(
            doc_id="d3",
            chunk_index=0,
            content="Relational databases like Postgres store structured relational data securely.",
            token_count=10,
            start_char=0,
            end_char=77,
            strategy="fixed",
            chunk_id="chunk-relational",
        ),
    ]


def test_deterministic_mock_reranker(candidates: list[DocumentChunk]) -> None:
    reranker = DeterministicMockReranker()
    results = reranker.rerank(
        query="PostgreSQL database queries",
        candidates=candidates,
        top_n=2,
    )
    assert len(results) == 2
    # chunk-postgres should score highest due to keyword overlap
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "chunk-postgres"
    assert score > 0.0

    # Test rerank_ranked returning RankedChunk
    ranked = reranker.rerank_ranked("PostgreSQL queries", candidates, top_n=2)
    assert len(ranked) == 2
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2
    assert ranked[0].chunk.chunk_id == "chunk-postgres"


def test_mock_reranker_with_mapping(candidates: list[DocumentChunk]) -> None:
    mapping = {
        "chunk-banana": 0.99,
        "chunk-postgres": 0.40,
        "chunk-relational": 0.10,
    }
    reranker = DeterministicMockReranker(score_mapping=mapping)
    results = reranker.rerank("any query", candidates, top_n=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "chunk-banana"
    assert results[0][1] == 0.99


def test_flashrank_reranker(candidates: list[DocumentChunk]) -> None:
    reranker = FlashRankReranker(model_name="ms-marco-TinyBERT-L-2-v2")
    results = reranker.rerank(
        query="What features does PostgreSQL database provide?",
        candidates=candidates,
        top_n=2,
    )
    assert len(results) == 2
    # Top candidate should be postgres
    assert results[0][0].chunk_id == "chunk-postgres"
    assert results[0][1] > results[1][1]


def test_empty_candidates() -> None:
    mock_r = DeterministicMockReranker()
    flash_r = FlashRankReranker()
    assert mock_r.rerank("query", []) == []
    assert flash_r.rerank("query", []) == []


def test_get_reranker_factory() -> None:
    r_mock = get_reranker("mock")
    assert isinstance(r_mock, DeterministicMockReranker)

    r_flash = get_reranker("flashrank")
    assert isinstance(r_flash, FlashRankReranker)

    with pytest.raises(ValueError):
        get_reranker("unknown_reranker")
