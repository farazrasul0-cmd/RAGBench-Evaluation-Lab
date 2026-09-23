"""Retrieval result contracts and research reproducibility tests."""

import pytest

from app.engine.embeddings.mock_provider import DeterministicMockEmbeddingProvider
from app.engine.rerankers.mock_reranker import DeterministicMockReranker
from app.engine.retrievers.bm25 import BM25Retriever
from app.engine.retrievers.dense import DenseRetriever
from app.engine.retrievers.hybrid import HybridRetriever
from app.engine.retrievers.reranked import RerankedRetriever
from app.schemas.chunk import DocumentChunk
from app.services.vector_store import InMemoryVectorStore


@pytest.fixture
def corpus_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk.create(
            doc_id="doc_py",
            chunk_index=0,
            content="Python is an interpreted high-level general-purpose programming language.",
            token_count=10,
            start_char=0,
            end_char=74,
            strategy="fixed",
            chunk_id="chunk-python",
            metadata={"lang": "en", "author": "Guido", "year": 1991},
        ),
        DocumentChunk.create(
            doc_id="doc_rs",
            chunk_index=0,
            content="Rust is a multi-paradigm, general-purpose programming language with safety.",
            token_count=10,
            start_char=0,
            end_char=75,
            strategy="fixed",
            chunk_id="chunk-rust",
            metadata={"lang": "en", "author": "Graydon", "year": 2010},
        ),
        DocumentChunk.create(
            doc_id="doc_bn",
            chunk_index=0,
            content="বাংলা ভাষা বাংলাদেশ ও ভারতের পশ্চিমবঙ্গের রাষ্ট্রভাষা ও মাতৃভাষা।",
            token_count=11,
            start_char=0,
            end_char=65,
            strategy="fixed",
            chunk_id="chunk-bengali",
            metadata={"lang": "bn", "region": "bengal", "year": 1952},
        ),
    ]


def test_contract_preservation_across_all_retrievers(
    corpus_chunks: list[DocumentChunk],
) -> None:
    provider = DeterministicMockEmbeddingProvider(dimension=8)
    vector_store = InMemoryVectorStore()
    coll = "contract_test_coll"
    vector_store.create_collection(coll, vector_size=8)
    vectors = provider.embed_texts([c.content for c in corpus_chunks])
    vector_store.upsert_chunks(coll, corpus_chunks, vectors)

    dense = DenseRetriever(provider, vector_store, coll)
    bm25 = BM25Retriever(corpus_chunks)
    hybrid = HybridRetriever(dense, bm25, fusion_method="rrf", rrf_k=60)
    reranked = RerankedRetriever(hybrid, DeterministicMockReranker(), retrieve_top_k=3)

    engines = [dense, bm25, hybrid, reranked]

    for engine in engines:
        results = engine.retrieve("programming language", top_k=2)
        assert len(results) == 2
        for chunk, score in results:
            # 1. Type contracts
            assert isinstance(chunk, DocumentChunk)
            assert isinstance(score, float)

            # 2. Field preservation
            assert chunk.chunk_id in ("chunk-python", "chunk-rust", "chunk-bengali")
            assert chunk.doc_id.startswith("doc_")
            assert len(chunk.content) > 0
            assert "year" in chunk.metadata
            assert chunk.token_count > 0

        # 3. Monotonic score descending
        assert results[0][1] >= results[1][1]


def test_research_reproducibility(corpus_chunks: list[DocumentChunk]) -> None:
    """Scientific benchmarking guarantee: identical inputs yield identical outputs."""
    provider = DeterministicMockEmbeddingProvider(dimension=8)
    vector_store = InMemoryVectorStore()
    coll = "reproducibility_coll"
    vector_store.create_collection(coll, vector_size=8)
    vectors = provider.embed_texts([c.content for c in corpus_chunks])
    vector_store.upsert_chunks(coll, corpus_chunks, vectors)

    dense = DenseRetriever(provider, vector_store, coll)
    bm25 = BM25Retriever(corpus_chunks)
    hybrid = HybridRetriever(dense, bm25, fusion_method="rrf", rrf_k=60)

    query = "general-purpose programming language safety"

    # Run 10 times and verify identical outputs
    baseline = hybrid.retrieve(query, top_k=3)
    baseline_ids = [c.chunk_id for c, _ in baseline]
    baseline_scores = [s for _, s in baseline]

    for _ in range(10):
        current = hybrid.retrieve(query, top_k=3)
        current_ids = [c.chunk_id for c, _ in current]
        current_scores = [s for _, s in current]
        assert current_ids == baseline_ids
        for b_s, c_s in zip(baseline_scores, current_scores, strict=True):
            assert pytest.approx(b_s, rel=1e-9) == c_s
