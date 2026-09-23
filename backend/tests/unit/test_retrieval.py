"""Unit tests for Dense, BM25, Hybrid, and Composable Reranked retrieval engines."""

import tempfile
from pathlib import Path

import pytest

from app.engine.embeddings.base import EmbeddingError
from app.engine.embeddings.mock_provider import DeterministicMockEmbeddingProvider
from app.engine.rerankers.mock_reranker import DeterministicMockReranker
from app.engine.retrievers.bm25 import BM25Retriever, BM25RetrieverError, tokenize_text
from app.engine.retrievers.dense import DenseRetriever
from app.engine.retrievers.hybrid import HybridRetriever
from app.engine.retrievers.reranked import RerankedRetriever
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
            metadata={"domain": "web", "topic": "framework", "version": 1},
        ),
        DocumentChunk.create(
            doc_id="doc2",
            chunk_index=0,
            content="PostgreSQL is a powerful open source object-relational database system.",
            token_count=10,
            start_char=0,
            end_char=71,
            strategy="fixed_token",
            metadata={"domain": "database", "topic": "sql", "version": 2},
        ),
        DocumentChunk.create(
            doc_id="doc3",
            chunk_index=0,
            content="Qdrant is a vector similarity search engine for AI applications.",
            token_count=13,
            start_char=0,
            end_char=64,
            strategy="fixed_token",
            metadata={"domain": "database", "topic": "vector", "version": 3},
        ),
        DocumentChunk.create(
            doc_id="doc4",
            chunk_index=0,
            content="বাংলা ভাষা অত্যন্ত সমৃদ্ধ এবং সুন্দর একটি ভাষা। এটি চমৎকার।",
            token_count=16,
            start_char=0,
            end_char=58,
            strategy="sentence",
            metadata={"domain": "nlp", "language": "bengali", "version": 4},
        ),
    ]


def test_tokenize_text_multilingual_and_code() -> None:
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

    # Code identifiers and technical symbols
    code_tokens = tokenize_text("fastapi_router.py get_chunk_id() model-name-v2")
    assert "fastapi" in code_tokens
    assert "router" in code_tokens
    assert "py" in code_tokens
    assert "get" in code_tokens
    assert "chunk" in code_tokens
    assert "id" in code_tokens
    assert "model" in code_tokens
    assert "name" in code_tokens
    assert "v2" in code_tokens


def test_bm25_empty_and_invalid_corpora() -> None:
    # Empty chunk list
    with pytest.raises(BM25RetrieverError):
        BM25Retriever(chunks=[])

    # Chunks with zero valid tokens
    empty_chunk = DocumentChunk.create(
        doc_id="d_empty",
        chunk_index=0,
        content="   ... !!! --- ",
        token_count=1,
        start_char=0,
        end_char=15,
        strategy="fixed",
    )
    with pytest.raises(BM25RetrieverError):
        BM25Retriever(chunks=[empty_chunk])


def test_bm25_term_frequency_effect() -> None:
    # Document with repeated term should score higher than document with single term
    c_single = DocumentChunk.create(
        doc_id="d_single",
        chunk_index=0,
        content="Python is great for scripting.",
        token_count=5,
        start_char=0,
        end_char=30,
        strategy="fixed",
    )
    c_repeated = DocumentChunk.create(
        doc_id="d_repeated",
        chunk_index=0,
        content="Python Python Python is the best Python programming language.",
        token_count=10,
        start_char=0,
        end_char=62,
        strategy="fixed",
    )
    c_other = DocumentChunk.create(
        doc_id="d_other",
        chunk_index=0,
        content="Rust and Golang are compiled systems languages.",
        token_count=8,
        start_char=0,
        end_char=47,
        strategy="fixed",
    )
    bm25 = BM25Retriever([c_single, c_repeated, c_other])
    results = bm25.retrieve("Python", top_k=2)
    assert len(results) == 2
    top_chunk, top_score = results[0]
    second_chunk, second_score = results[1]
    assert top_chunk.chunk_id == c_repeated.chunk_id
    assert second_chunk.chunk_id == c_single.chunk_id
    assert top_score > second_score


def test_bm25_save_and_load_equivalence(sample_chunks: list[DocumentChunk]) -> None:
    retriever = BM25Retriever(chunks=sample_chunks)
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "bm25_index.json"
        retriever.save(save_path)
        assert save_path.exists()

        loaded = BM25Retriever.load(save_path)
        assert len(loaded.chunks) == len(retriever.chunks)

        # Before and after search must produce identical rankings and scores
        query = "PostgreSQL database"
        res_original = retriever.retrieve(query, top_k=3)
        res_loaded = loaded.retrieve(query, top_k=3)

        assert len(res_original) == len(res_loaded)
        for (ch_orig, sc_orig), (ch_load, sc_load) in zip(res_original, res_loaded, strict=True):
            assert ch_orig.chunk_id == ch_load.chunk_id
            assert ch_orig.metadata == ch_load.metadata
            assert pytest.approx(sc_orig, rel=1e-6) == sc_load


def test_bm25_metadata_filtering(sample_chunks: list[DocumentChunk]) -> None:
    retriever = BM25Retriever(chunks=sample_chunks)
    # Search database domain
    db_results = retriever.retrieve(
        "database system",
        top_k=5,
        filter_metadata={"domain": "database"},
    )
    assert len(db_results) == 2
    for chunk, _ in db_results:
        assert chunk.metadata["domain"] == "database"


def test_dense_retriever_audit(sample_chunks: list[DocumentChunk]) -> None:
    provider = DeterministicMockEmbeddingProvider(dimension=8)
    vector_store = InMemoryVectorStore()
    coll = "test_collection_audit"
    vector_store.create_collection(coll, vector_size=8)

    vectors = provider.embed_texts([c.content for c in sample_chunks])
    vector_store.upsert_chunks(coll, sample_chunks, vectors)

    dense = DenseRetriever(
        embedding_provider=provider,
        vector_store=vector_store,
        collection_name=coll,
    )

    # 1. Basic retrieval and top_k
    results = dense.retrieve("database system", top_k=2)
    assert len(results) == 2
    assert all(isinstance(score, float) for _, score in results)

    # 2. Metadata filtering
    filtered = dense.retrieve(
        "database system",
        top_k=2,
        filter_metadata={"domain": "database"},
    )
    assert len(filtered) == 2
    for chunk, _ in filtered:
        assert chunk.metadata["domain"] == "database"

    # 3. Empty results when filtering with non-existent tag
    empty_filtered = dense.retrieve(
        "database system",
        top_k=2,
        filter_metadata={"domain": "non_existent"},
    )
    assert len(empty_filtered) == 0

    # 4. Dimension validation
    with pytest.raises(EmbeddingError):
        provider.validate_dimensions([[0.1, 0.2]])  # dimension 2 != 8


def test_composable_six_topologies(sample_chunks: list[DocumentChunk]) -> None:
    """Verify Dense, Dense+Reranker, BM25, BM25+Reranker, Hybrid, Hybrid+Reranker all compose."""
    provider = DeterministicMockEmbeddingProvider(dimension=8)
    vector_store = InMemoryVectorStore()
    coll = "six_topologies_coll"
    vector_store.create_collection(coll, vector_size=8)
    vectors = provider.embed_texts([c.content for c in sample_chunks])
    vector_store.upsert_chunks(coll, sample_chunks, vectors)

    dense = DenseRetriever(provider, vector_store, coll)
    bm25 = BM25Retriever(sample_chunks)
    hybrid = HybridRetriever(dense, bm25, fusion_method="rrf", rrf_k=60)
    reranker = DeterministicMockReranker()

    # 6 Topologies
    t1_dense = dense
    t2_dense_rerank = RerankedRetriever(dense, reranker, retrieve_top_k=4)
    t3_bm25 = bm25
    t4_bm25_rerank = RerankedRetriever(bm25, reranker, retrieve_top_k=4)
    t5_hybrid = hybrid
    t6_hybrid_rerank = RerankedRetriever(hybrid, reranker, retrieve_top_k=4)

    topologies = [
        t1_dense,
        t2_dense_rerank,
        t3_bm25,
        t4_bm25_rerank,
        t5_hybrid,
        t6_hybrid_rerank,
    ]

    query = "FastAPI web framework"
    for idx, engine in enumerate(topologies, start=1):
        res = engine.retrieve(query, top_k=2)
        assert len(res) == 2, f"Topology {idx} failed top_k retrieval"
        for chunk, score in res:
            assert isinstance(chunk, DocumentChunk)
            assert isinstance(score, float)
            assert "domain" in chunk.metadata
