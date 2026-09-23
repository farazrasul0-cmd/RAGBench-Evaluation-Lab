"""Unit tests for Dense, BM25, and Hybrid retrieval engines."""

import tempfile
from pathlib import Path

import pytest

from app.engine.embeddings.mock_provider import DeterministicMockEmbeddingProvider
from app.engine.retrievers.bm25 import BM25Retriever, BM25RetrieverError, tokenize_text
from app.engine.retrievers.dense import DenseRetriever
from app.engine.retrievers.hybrid import HybridRetriever, HybridRetrieverError
from app.schemas.chunk import DocumentChunk
from app.services.vector_store import InMemoryVectorStore


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk.create(
            doc_id="doc1",
            chunk_index=0,
            content="FastAPI is a modern high performance web framework for Python.",
            token_count=14,
            start_char=0,
            end_char=63,
            strategy="fixed_token",
            metadata={"domain": "web", "topic": "framework"},
        ),
        DocumentChunk.create(
            doc_id="doc2",
            chunk_index=0,
            content="PostgreSQL is a powerful open source object-relational database system.",
            token_count=10,
            start_char=0,
            end_char=71,
            strategy="fixed_token",
            metadata={"domain": "database", "topic": "sql"},
        ),
        DocumentChunk.create(
            doc_id="doc3",
            chunk_index=0,
            content="Qdrant is a vector similarity search engine for AI applications.",
            token_count=13,
            start_char=0,
            end_char=64,
            strategy="fixed_token",
            metadata={"domain": "database", "topic": "vector"},
        ),
        DocumentChunk.create(
            doc_id="doc4",
            chunk_index=0,
            content="বাংলা ভাষা অত্যন্ত সমৃদ্ধ এবং সুন্দর একটি ভাষা। এটি চমৎকার।",
            token_count=16,
            start_char=0,
            end_char=58,
            strategy="sentence",
            metadata={"domain": "nlp", "language": "bengali"},
        ),
    ]


def test_tokenize_text_multilingual() -> None:
    en_tokens = tokenize_text("FastAPI & Python 3.12!")
    assert "fastapi" in en_tokens
    assert "python" in en_tokens
    assert "3" in en_tokens
    assert "12" in en_tokens

    bn_tokens = tokenize_text("বাংলা ভাষা অত্যন্ত সুন্দর। এটি চমৎকার॥")
    assert "বাংলা" in bn_tokens
    assert "ভাষা" in bn_tokens
    assert "সুন্দর" in bn_tokens
    assert "।" not in bn_tokens
    assert "॥" not in bn_tokens


def test_bm25_retriever_empty_error() -> None:
    with pytest.raises(BM25RetrieverError):
        BM25Retriever(chunks=[])


def test_bm25_retrieval(sample_chunks: list[DocumentChunk]) -> None:
    retriever = BM25Retriever(chunks=sample_chunks)
    assert len(retriever.chunks) == 4

    results = retriever.retrieve("FastAPI Python web framework", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == sample_chunks[0].chunk_id
    assert score > 0.0

    # Bengali retrieval
    bn_results = retriever.retrieve("বাংলা ভাষা সুন্দর", top_k=2)
    assert len(bn_results) > 0
    assert bn_results[0][0].chunk_id == sample_chunks[3].chunk_id
    assert bn_results[0][1] > 0.0


def test_bm25_save_and_load(sample_chunks: list[DocumentChunk]) -> None:
    retriever = BM25Retriever(chunks=sample_chunks)
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "bm25_index.json"
        retriever.save(save_path)
        assert save_path.exists()

        loaded = BM25Retriever.load(save_path)
        assert len(loaded.chunks) == 4
        res = loaded.retrieve("PostgreSQL database", top_k=1)
        assert len(res) == 1
        assert res[0][0].chunk_id == sample_chunks[1].chunk_id


def test_dense_retriever(sample_chunks: list[DocumentChunk]) -> None:
    provider = DeterministicMockEmbeddingProvider(dimension=8)
    vector_store = InMemoryVectorStore()
    coll = "test_collection"
    vector_store.create_collection(coll, vector_size=8)

    vectors = provider.embed_texts([c.content for c in sample_chunks])
    vector_store.upsert_chunks(coll, sample_chunks, vectors)

    dense = DenseRetriever(
        embedding_provider=provider,
        vector_store=vector_store,
        collection_name=coll,
    )

    results = dense.retrieve("database system", top_k=2)
    assert len(results) == 2
    assert all(isinstance(score, float) for _, score in results)

    # Filtered search
    filtered = dense.retrieve(
        "database system",
        top_k=2,
        filter_metadata={"domain": "database"},
    )
    assert len(filtered) <= 2
    for chunk, _ in filtered:
        assert chunk.metadata["domain"] == "database"


def test_hybrid_retriever(sample_chunks: list[DocumentChunk]) -> None:
    provider = DeterministicMockEmbeddingProvider(dimension=8)
    vector_store = InMemoryVectorStore()
    coll = "hybrid_collection"
    vector_store.create_collection(coll, vector_size=8)
    vectors = provider.embed_texts([c.content for c in sample_chunks])
    vector_store.upsert_chunks(coll, sample_chunks, vectors)

    dense = DenseRetriever(provider, vector_store, coll)
    bm25 = BM25Retriever(sample_chunks)

    # Test RRF
    hybrid_rrf = HybridRetriever(dense, bm25, fusion_method="rrf", rrf_k=60)
    rrf_results = hybrid_rrf.retrieve("FastAPI web framework", top_k=2)
    assert len(rrf_results) == 2
    assert rrf_results[0][0].chunk_id == sample_chunks[0].chunk_id
    assert rrf_results[0][1] > 0.0

    # Test RSN
    hybrid_rsn = HybridRetriever(dense, bm25, fusion_method="rsn", alpha=0.6)
    rsn_results = hybrid_rsn.retrieve("Qdrant vector similarity", top_k=2)
    assert len(rsn_results) == 2
    assert rsn_results[0][1] >= rsn_results[1][1]

    # Test invalid method
    with pytest.raises(HybridRetrieverError):
        HybridRetriever(dense, bm25, fusion_method="unknown_method")
