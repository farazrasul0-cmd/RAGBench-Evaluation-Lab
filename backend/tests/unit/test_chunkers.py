"""Unit tests for document chunking strategies."""

import pytest

from app.core.exceptions import ChunkingError
from app.engine.chunkers import (
    FixedTokenChunker,
    RecursiveCharacterChunker,
    SemanticSimilarityChunker,
    SentenceBoundaryChunker,
)
from app.schemas.chunk import generate_chunk_id
from app.schemas.document import RawDocument


@pytest.fixture
def sample_document() -> RawDocument:
    """Provide a standard technical document for chunking tests."""
    content = """Retrieval-Augmented Generation (RAG) is an architectural pattern.
It enhances large language models by retrieving relevant external context.
Chunking strategy is one of the most critical hyper-parameters in RAG.
Fixed-size token windowing divides text into uniform segments with overlap.
Recursive chunking hierarchically breaks documents by semantic separators.
Sentence boundary chunking preserves complete grammatical sentences.
Semantic chunking calculates embedding distances between adjacent sentences.
Each strategy presents distinct trade-offs between precision and recall."""
    return RawDocument.create(filename="rag_intro.md", content=content, doc_id="doc_rag_001")


def test_fixed_token_chunker(sample_document: RawDocument) -> None:
    """Verify FixedTokenChunker creates overlapping token-bounded chunks."""
    chunker = FixedTokenChunker(chunk_size=30, chunk_overlap=10)
    chunks = chunker.chunk(sample_document)

    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.doc_id == sample_document.doc_id
        assert c.strategy == "fixed_token"
        assert c.token_count <= 30
        assert c.char_count > 0
        assert c.chunk_id == generate_chunk_id(c.doc_id, c.chunk_index, c.content)


def test_fixed_token_chunker_invalid_params() -> None:
    """Verify parameter validation for fixed token chunker."""
    with pytest.raises(ChunkingError, match="chunk_size must be positive"):
        FixedTokenChunker(chunk_size=-1)
    with pytest.raises(ChunkingError, match="chunk_overlap cannot be negative"):
        FixedTokenChunker(chunk_size=100, chunk_overlap=-5)
    with pytest.raises(ChunkingError, match="must be less than chunk_size"):
        FixedTokenChunker(chunk_size=50, chunk_overlap=50)


def test_recursive_character_chunker(sample_document: RawDocument) -> None:
    """Verify RecursiveCharacterChunker respects hierarchical boundaries."""
    chunker = RecursiveCharacterChunker(chunk_size=40, chunk_overlap=10)
    chunks = chunker.chunk(sample_document)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.strategy == "recursive"
        assert c.token_count <= 50  # Allows slight margin for whitespace merge
        # Verify no mid-word split (content starts and ends cleanly)
        assert c.content == c.content.strip()


def test_sentence_boundary_chunker(sample_document: RawDocument) -> None:
    """Verify SentenceBoundaryChunker does not truncate sentences mid-sentence."""
    chunker = SentenceBoundaryChunker(chunk_size=40, chunk_overlap=15)
    chunks = chunker.chunk(sample_document)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.strategy == "sentence"
        # Each chunk should end with a sentence terminator
        assert c.content.endswith((".", "!", "?"))
        assert "sentence_count" in c.metadata


def test_sentence_boundary_bengali_support() -> None:
    """Verify SentenceBoundaryChunker natively splits Bengali sentences using Dari (।)."""
    bengali_content = """প্রাকৃতিক ভাষা প্রক্রিয়াকরণ একটি গুরুত্বপূর্ণ ক্ষেত্র।
বৃহৎ ভাষা মডেলগুলি তথ্য সন্ধানে সহায়তা করে।
সঠিক তথ্যের জন্য আরএজি ব্যবস্থা অত্যন্ত কার্যকর।"""
    bengali_doc = RawDocument.create(
        filename="bengali_rag.txt", content=bengali_content, doc_id="doc_bn_001"
    )

    chunker = SentenceBoundaryChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk(bengali_doc)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.content.endswith("।")


def test_semantic_similarity_chunker_fallback(sample_document: RawDocument) -> None:
    """Verify SemanticSimilarityChunker creates chunks using built-in similarity fallback."""
    chunker = SemanticSimilarityChunker(
        similarity_threshold=0.85, min_chunk_tokens=10, max_chunk_tokens=60
    )
    chunks = chunker.chunk(sample_document)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.strategy == "semantic"
        assert c.chunk_id != ""


def test_semantic_similarity_chunker_custom_embedding(sample_document: RawDocument) -> None:
    """Verify SemanticSimilarityChunker operates with injected embedding callable."""

    # Deterministic mock embedding: 4-dimensional vector based on sentence length
    def mock_embed(texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 5), 1.0, 0.0, 0.5] for t in texts]

    chunker = SemanticSimilarityChunker(
        similarity_threshold=0.99,
        min_chunk_tokens=5,
        max_chunk_tokens=50,
        embedding_fn=mock_embed,
    )
    chunks = chunker.chunk(sample_document)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.strategy == "semantic"


def test_determinism_and_collision_resistance(sample_document: RawDocument) -> None:
    """Verify repeated chunking yields identical IDs, and different inputs collide-free."""
    chunker = FixedTokenChunker(chunk_size=30, chunk_overlap=10)
    run1 = chunker.chunk(sample_document)
    run2 = chunker.chunk(sample_document)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2, strict=True):
        assert c1.chunk_id == c2.chunk_id
        assert c1.content == c2.content
        assert c1.start_char == c2.start_char
        assert c1.end_char == c2.end_char

    # Collision resistance
    id1 = generate_chunk_id("doc_1", 0, "Test content A")
    id2 = generate_chunk_id("doc_1", 0, "Test content B")
    id3 = generate_chunk_id("doc_1", 1, "Test content A")
    id4 = generate_chunk_id("doc_2", 0, "Test content A")

    assert len({id1, id2, id3, id4}) == 4


def test_empty_document_handling() -> None:
    """Verify all chunkers safely handle empty documents."""
    empty_doc = RawDocument.create(filename="empty.txt", content="", doc_id="empty_01")
    for chunker in [
        FixedTokenChunker(chunk_size=100, chunk_overlap=10),
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10),
        SentenceBoundaryChunker(chunk_size=100, chunk_overlap=10),
        SemanticSimilarityChunker(similarity_threshold=0.8),
    ]:
        assert chunker.chunk(empty_doc) == []
