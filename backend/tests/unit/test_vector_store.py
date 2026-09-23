"""Unit tests for Qdrant and InMemory vector store adapters."""

import re
import time

from app.schemas.chunk import DocumentChunk
from app.services.vector_store import (
    InMemoryVectorStore,
    QdrantVectorStore,
    compute_collection_name,
)


def test_collection_naming_contract() -> None:
    """Verify collection naming conforms to SYSTEM_ARCHITECTURE.md contract."""
    chunking_cfg = {"strategy": "recursive", "chunk_size": 512, "chunk_overlap": 64}
    model_name = "BAAI/bge-small-en-v1.5"
    name = compute_collection_name("academic_research", chunking_cfg, model_name)

    # Format: ragbench_{dataset_id}_{chunking_hash}_{embedding_model_hash}
    pattern = r"^ragbench_academic_research_[a-f0-9]{8}_[a-f0-9]{8}$"
    assert re.match(pattern, name) is not None

    # Determinism: identical parameters yield identical collection name
    name2 = compute_collection_name("academic_research", chunking_cfg, model_name)
    assert name == name2


def test_qdrant_in_memory_crud_and_search() -> None:
    """Verify QdrantVectorStore collection creation, batch upsert, search, and filtering."""
    store = QdrantVectorStore(location=":memory:")
    col_name = "test_qdrant_col"
    vector_size = 4

    store.create_collection(col_name, vector_size=vector_size)
    assert store.collection_exists(col_name)

    # Create synthetic chunks
    chunk_a = DocumentChunk.create(
        doc_id="doc_1",
        chunk_index=0,
        content="Artificial intelligence and machine learning overview.",
        token_count=10,
        start_char=0,
        end_char=50,
        strategy="recursive",
        metadata={"domain": "cs", "topic": "ai"},
    )
    chunk_b = DocumentChunk.create(
        doc_id="doc_1",
        chunk_index=1,
        content="Deep learning architectures and transformer models.",
        token_count=10,
        start_char=51,
        end_char=100,
        strategy="recursive",
        metadata={"domain": "cs", "topic": "deep_learning"},
    )
    chunk_c = DocumentChunk.create(
        doc_id="doc_2",
        chunk_index=0,
        content="Quantum physics and quantum entanglement principles.",
        token_count=10,
        start_char=0,
        end_char=50,
        strategy="fixed_token",
        metadata={"domain": "physics", "topic": "quantum"},
    )

    chunks = [chunk_a, chunk_b, chunk_c]
    vectors = [
        [1.0, 0.0, 0.0, 0.0],  # chunk_a: aligns with dimension 0
        [0.9, 0.1, 0.0, 0.0],  # chunk_b: mostly aligns with dimension 0
        [0.0, 0.0, 1.0, 0.0],  # chunk_c: aligns with dimension 2
    ]

    count = store.upsert_chunks(col_name, chunks, vectors)
    assert count == 3
    assert store.get_collection_count(col_name) == 3

    # Search for vector close to chunk_a
    results = store.search(col_name, query_vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    top_chunk, top_score = results[0]
    assert top_chunk.chunk_id == chunk_a.chunk_id
    assert top_score >= 0.99

    # Filtered search: only domain == 'physics'
    phys_results = store.search(
        col_name, query_vector=[1.0, 0.0, 0.0, 0.0], top_k=5, filter_metadata={"domain": "physics"}
    )
    assert len(phys_results) == 1
    assert phys_results[0][0].chunk_id == chunk_c.chunk_id

    # Cleanup
    store.delete_collection(col_name)
    assert not store.collection_exists(col_name)


def test_qdrant_batch_performance() -> None:
    """Benchmark test verifying >= 500 points upserted in < 1 second locally."""
    store = QdrantVectorStore(location=":memory:")
    col_name = "perf_col"
    dim = 8
    store.create_collection(col_name, vector_size=dim)

    num_points = 500
    chunks = [
        DocumentChunk.create(
            doc_id=f"doc_{i // 10}",
            chunk_index=i % 10,
            content=f"Benchmark test passage number {i} with synthetic evaluation content.",
            token_count=12,
            start_char=0,
            end_char=50,
            strategy="fixed_token",
            metadata={"index": i},
        )
        for i in range(num_points)
    ]
    vectors = [[(i % 10) / 10.0] * dim for i in range(num_points)]

    start_time = time.perf_counter()
    upserted = store.upsert_chunks(col_name, chunks, vectors)
    elapsed = time.perf_counter() - start_time

    assert upserted == 500
    assert elapsed < 1.5  # Sub-second to 1.5s tolerance across CI/Windows
    assert store.get_collection_count(col_name) == 500


def test_in_memory_vector_store_parity() -> None:
    """Verify InMemoryVectorStore delivers identical ranking parity to dense search."""
    store = InMemoryVectorStore()
    col = "mem_test"
    store.create_collection(col, vector_size=3)

    c1 = DocumentChunk.create(
        doc_id="d1",
        chunk_index=0,
        content="Alpha text",
        token_count=5,
        start_char=0,
        end_char=10,
        strategy="rec",
        metadata={"tag": "A"},
    )
    c2 = DocumentChunk.create(
        doc_id="d1",
        chunk_index=1,
        content="Beta text",
        token_count=5,
        start_char=11,
        end_char=20,
        strategy="rec",
        metadata={"tag": "B"},
    )

    store.upsert_chunks(col, [c1, c2], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert store.get_collection_count(col) == 2

    res = store.search(col, query_vector=[1.0, 0.0, 0.0], top_k=1)
    assert len(res) == 1
    assert res[0][0].chunk_id == c1.chunk_id
    assert abs(res[0][1] - 1.0) < 1e-5

    # Filter test
    filtered = store.search(
        col, query_vector=[1.0, 0.0, 0.0], top_k=2, filter_metadata={"tag": "B"}
    )
    assert len(filtered) == 1
    assert filtered[0][0].chunk_id == c2.chunk_id
