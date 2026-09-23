"""Unit tests and independent verification for passage rerankers."""

import pytest

from app.engine.rerankers import get_reranker
from app.engine.rerankers.cross_encoder import CrossEncoderReranker, CrossEncoderRerankerError
from app.engine.rerankers.flashrank import FlashRankReranker
from app.engine.rerankers.mock_reranker import DeterministicMockReranker
from app.schemas.chunk import DocumentChunk, RankedChunk


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
            metadata={"category": "fruit"},
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
            metadata={"category": "db"},
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
            metadata={"category": "db"},
        ),
    ]


def test_deterministic_mock_reranker(candidates: list[DocumentChunk]) -> None:
    reranker = DeterministicMockReranker()

    # 1. Empty candidates
    assert reranker.rerank("query", []) == []

    # 2. Single candidate
    single = reranker.rerank("queries", candidates[:1], top_n=1)
    assert len(single) == 1

    # 3. Multiple candidates and top_n truncation
    results = reranker.rerank(
        query="PostgreSQL database queries",
        candidates=candidates,
        top_n=2,
    )
    assert len(results) == 2
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "chunk-postgres"
    assert top_chunk.metadata == {"category": "db"}
    assert score > 0.0

    # 4. Structured RankedChunk verification
    ranked = reranker.rerank_ranked("PostgreSQL queries", candidates, top_n=2)
    assert len(ranked) == 2
    assert isinstance(ranked[0], RankedChunk)
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
    assert results[0][0].chunk_id == "chunk-postgres"
    assert results[0][0].metadata["category"] == "db"
    assert results[0][1] > results[1][1]


def test_cross_encoder_lazy_loading_error() -> None:
    # Test that invalid model name or missing dependency raises CrossEncoderRerankerError
    reranker = CrossEncoderReranker(model_name="non-existent-cross-encoder-model-path-xyz")
    with pytest.raises(CrossEncoderRerankerError):
        reranker.rerank("query", [])  # Returns [] for empty candidates without loading model
        # With candidate, triggers model loading and raises exception gracefully
        dummy_chunk = DocumentChunk.create(
            doc_id="d",
            chunk_index=0,
            content="test",
            token_count=1,
            start_char=0,
            end_char=4,
            strategy="test",
        )
        reranker.rerank("query", [dummy_chunk])


def test_get_reranker_factory() -> None:
    r_mock = get_reranker("mock")
    assert isinstance(r_mock, DeterministicMockReranker)

    r_flash = get_reranker("flashrank")
    assert isinstance(r_flash, FlashRankReranker)

    with pytest.raises(ValueError):
        get_reranker("unknown_reranker")
