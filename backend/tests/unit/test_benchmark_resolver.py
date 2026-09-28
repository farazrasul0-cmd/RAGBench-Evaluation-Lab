"""Unit tests for GroundTruthChunkResolver demonstrating RQ1 chunking independence.

Verifies:
1. The exact same BenchmarkQuerySet dynamically resolves against Fixed Token, Recursive Character,
   and Sentence Boundary chunking without modifying benchmark ground truth.
2. Character span intersection, substring containment, and multi-chunk span matching.
3. Ambiguity rejection across multiple documents.
4. Unresolved passage error handling and allow-unresolved flag support.
"""

import pytest

from app.engine.benchmark.resolver import (
    AmbiguousPassageError,
    GroundTruthChunkResolver,
    UnresolvedPassageError,
)
from app.engine.chunkers.fixed_token import FixedTokenChunker
from app.engine.chunkers.recursive import RecursiveCharacterChunker
from app.engine.chunkers.sentence import SentenceBoundaryChunker
from app.schemas.benchmark import BenchmarkQuery, BenchmarkQuerySet, GroundTruthPassage
from app.schemas.chunk import DocumentChunk
from app.schemas.document import RawDocument


@pytest.fixture
def sample_benchmark() -> BenchmarkQuerySet:
    """Canonical chunking-independent benchmark query set."""
    q1 = BenchmarkQuery(
        query_id="q1",
        query="What is the speed of light in vacuum?",
        ground_truth_answer="Approximately 299,792,458 meters per second.",
        ground_truth_passages=[
            GroundTruthPassage(
                doc_id="physics.txt",
                text_snippet=(
                    "The speed of light in vacuum, commonly denoted c, is exactly 299,792,458 m/s."
                ),
            )
        ],
    )
    q2 = BenchmarkQuery(
        query_id="q2",
        query="What defines general relativity?",
        ground_truth_answer="Gravitation as spacetime curvature.",
        ground_truth_passages=[
            GroundTruthPassage(
                doc_id="physics.txt",
                text_snippet=(
                    "General relativity explains gravitational force as the curvature of spacetime."
                ),
            )
        ],
    )
    return BenchmarkQuerySet(
        name="PhysicsBenchmark",
        version=1,
        queries=[q1, q2],
    )


@pytest.fixture
def physics_document() -> RawDocument:
    """Source physics document."""
    text = (
        "The speed of light in vacuum, commonly denoted c, is exactly 299,792,458 m/s. "
        "It is a universal physical constant fundamental to many areas of modern physics. "
        "Special relativity establishes c is the upper limit for matter and info. "
        "General relativity explains gravitational force as the curvature of spacetime. "
        "Mass and energy determine the curvature of the surrounding spacetime continuum."
    )
    return RawDocument.create(
        filename="physics.txt",
        content=text,
        doc_id="physics.txt",
    )


def test_resolver_across_multiple_chunking_strategies(
    sample_benchmark: BenchmarkQuerySet, physics_document: RawDocument
) -> None:
    """Verify benchmark resolves across Fixed, Recursive, and Sentence chunkers."""
    # 1. Fixed Token Chunking (small tokens -> multiple chunks)
    fixed_chunker = FixedTokenChunker(chunk_size=15, chunk_overlap=3)
    fixed_chunks = fixed_chunker.chunk(physics_document)
    assert len(fixed_chunks) >= 4

    resolved_fixed = GroundTruthChunkResolver.resolve_benchmark(sample_benchmark, fixed_chunks)
    assert len(resolved_fixed) == 2
    # Verify both queries resolved to concrete chunk IDs
    assert len(resolved_fixed[0].ground_truth_chunks) >= 1
    assert len(resolved_fixed[1].ground_truth_chunks) >= 1
    # Verify resolved chunks actually contain the gold snippet text
    c_fixed_q1 = [c for c in fixed_chunks if c.chunk_id in resolved_fixed[0].ground_truth_chunks]
    assert any("299,792,458" in c.content for c in c_fixed_q1)

    # 2. Recursive Character Chunking
    recursive_chunker = RecursiveCharacterChunker(chunk_size=30, chunk_overlap=5)
    rec_chunks = recursive_chunker.chunk(physics_document)
    assert len(rec_chunks) >= 2

    resolved_rec = GroundTruthChunkResolver.resolve_benchmark(sample_benchmark, rec_chunks)
    assert len(resolved_rec) == 2
    assert len(resolved_rec[0].ground_truth_chunks) >= 1
    assert len(resolved_rec[1].ground_truth_chunks) >= 1

    # 3. Sentence Boundary Chunking
    sentence_chunker = SentenceBoundaryChunker(chunk_size=25, chunk_overlap=0)
    sentence_chunks = sentence_chunker.chunk(physics_document)
    assert len(sentence_chunks) >= 3

    resolved_sent = GroundTruthChunkResolver.resolve_benchmark(sample_benchmark, sentence_chunks)
    assert len(resolved_sent) == 2
    assert len(resolved_sent[0].ground_truth_chunks) == 1
    assert len(resolved_sent[1].ground_truth_chunks) == 1

    # Chunk IDs differ between strategies, but canonical query IDs and answers remain invariant
    assert (
        resolved_fixed[0].query_id == resolved_rec[0].query_id == resolved_sent[0].query_id == "q1"
    )
    assert (
        resolved_fixed[0].ground_truth_chunks != resolved_rec[0].ground_truth_chunks
        or resolved_fixed[0].ground_truth_chunks != resolved_sent[0].ground_truth_chunks
    )


def test_resolver_character_span_offsets() -> None:
    """Verify character offset span overlap resolution."""
    c1 = DocumentChunk.create(
        doc_id="doc1",
        chunk_index=0,
        content="Alpha Beta Gamma Delta",
        token_count=4,
        start_char=0,
        end_char=22,
        strategy="fixed",
    )
    passage = GroundTruthPassage(
        doc_id="doc1",
        text_snippet="Beta Gamma",
        start_char=6,
        end_char=16,
    )
    matched = GroundTruthChunkResolver.resolve_chunks_for_passage(passage, [c1])
    assert matched == [c1.chunk_id]


def test_resolver_ambiguous_across_unrelated_documents() -> None:
    """Verify error when snippet matches multiple docs without explicit doc_id."""
    c1 = DocumentChunk.create(
        doc_id="docA",
        chunk_index=0,
        content="Common ubiquitous fact statement in document A.",
        token_count=7,
        start_char=0,
        end_char=48,
        strategy="fixed",
    )
    c2 = DocumentChunk.create(
        doc_id="docB",
        chunk_index=0,
        content="Common ubiquitous fact statement in document B.",
        token_count=7,
        start_char=0,
        end_char=48,
        strategy="fixed",
    )
    passage = GroundTruthPassage(
        doc_id="",  # No doc_id specified
        text_snippet="Common ubiquitous fact statement",
    )
    with pytest.raises(AmbiguousPassageError, match="AMBIGUOUS across multiple documents"):
        GroundTruthChunkResolver.resolve_chunks_for_passage(passage, [c1, c2])


def test_resolver_unresolved_passage_handling() -> None:
    """Verify UnresolvedPassageError raised by default and permitted when configured."""
    c1 = DocumentChunk.create(
        doc_id="doc1",
        chunk_index=0,
        content="Completely unrelated content.",
        token_count=3,
        start_char=0,
        end_char=29,
        strategy="fixed",
    )
    q = BenchmarkQuery(
        query_id="q-unres",
        query="Missing query?",
        ground_truth_answer="Answer.",
        ground_truth_passages=[
            GroundTruthPassage(doc_id="doc1", text_snippet="Non-existent snippet")
        ],
    )
    with pytest.raises(UnresolvedPassageError, match="could not be resolved"):
        GroundTruthChunkResolver.resolve_query(q, [c1], allow_unresolved=False)

    eq_allowed = GroundTruthChunkResolver.resolve_query(q, [c1], allow_unresolved=True)
    assert eq_allowed.ground_truth_chunks == []
