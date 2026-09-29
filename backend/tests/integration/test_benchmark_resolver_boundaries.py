"""Exhaustive integration tests for GroundTruthChunkResolver across 8 boundary conditions.

Covers:
1. 1:1 match: passage maps cleanly to exactly one chunk.
2. 2-chunk split: passage spans across 2 consecutive chunks (both chunk IDs resolved).
3. 3-chunk split: passage spans across 3 consecutive chunks (all three chunk IDs resolved).
4. Head boundary: passage aligns exactly with the very start of a chunk.
5. Tail boundary: passage aligns exactly with the very end of a chunk.
6. Unicode/Indic Bengali: Bengali script with combining characters and diacritics.
7. Cross-doc disambiguation: two documents with identical text, resolved strictly by doc_id.
8. Intra-doc repeated passages: repeated within document, disambiguated by char offsets.
"""

from app.engine.benchmark.resolver import (
    GroundTruthChunkResolver,
)
from app.schemas.benchmark import BenchmarkQuery, GroundTruthPassage
from app.schemas.chunk import DocumentChunk


def make_chunk(
    chunk_id: str,
    doc_id: str,
    content: str,
    start_char: int,
    end_char: int,
    chunk_index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        start_char=start_char,
        end_char=end_char,
        chunk_index=chunk_index,
        token_count=len(content.split()),
        char_count=len(content),
        strategy="fixed",
    )


def test_boundary_1_one_to_one_exact_match() -> None:
    """Boundary 1: Passage is fully contained within exactly one chunk."""
    chunks = [
        make_chunk(
            chunk_id="chunk-doc1-0",
            doc_id="doc1",
            content="Deep learning models require substantial computational resources.",
            start_char=0,
            end_char=68,
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-doc1-1",
            doc_id="doc1",
            content="Reinforcement learning utilizes reward signals to optimize policies.",
            start_char=69,
            end_char=138,
            chunk_index=1,
        ),
    ]

    passage = GroundTruthPassage(
        doc_id="doc1",
        text_snippet="Deep learning models require substantial computational resources.",
        start_char=0,
        end_char=68,
    )

    resolved = GroundTruthChunkResolver.resolve_chunks_for_passage(passage, chunks)
    assert resolved == ["chunk-doc1-0"]


def test_boundary_2_two_chunk_split() -> None:
    """Boundary 2: Passage spans across two consecutive chunks."""
    chunks = [
        make_chunk(
            chunk_id="chunk-doc1-0",
            doc_id="doc1",
            content="The experiment measured temperature variations across 30 distinct regions.",
            start_char=0,
            end_char=79,
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-doc1-1",
            doc_id="doc1",
            content="Statistical significance was confirmed with p-values consistently below 0.01.",
            start_char=80,
            end_char=159,
            chunk_index=1,
        ),
    ]

    passage = GroundTruthPassage(
        doc_id="doc1",
        text_snippet="across thirty distinct regions. Statistical significance was confirmed",
        start_char=50,
        end_char=120,
    )

    resolved = GroundTruthChunkResolver.resolve_chunks_for_passage(passage, chunks)
    assert "chunk-doc1-0" in resolved
    assert "chunk-doc1-1" in resolved
    assert len(resolved) == 2


def test_boundary_3_three_chunk_split() -> None:
    """Boundary 3: Passage spans across three consecutive chunks."""
    chunks = [
        make_chunk(
            chunk_id="chunk-doc1-0",
            doc_id="doc1",
            content="Phase one involved data collection and pre-processing pipeline stabilization.",
            start_char=0,
            end_char=78,
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-doc1-1",
            doc_id="doc1",
            content="Phase two performed vector embeddings calculation and hybrid search indexing.",
            start_char=79,
            end_char=157,
            chunk_index=1,
        ),
        make_chunk(
            chunk_id="chunk-doc1-2",
            doc_id="doc1",
            content="Phase three conducted cross-validation and hypothesis significance testing.",
            start_char=158,
            end_char=243,
            chunk_index=2,
        ),
    ]

    passage = GroundTruthPassage(
        doc_id="doc1",
        text_snippet="stabilization. Phase two performed vector embeddings calculation and "
        "hybrid search indexing. Phase three conducted empirical",
        start_char=65,
        end_char=185,
    )

    resolved = GroundTruthChunkResolver.resolve_chunks_for_passage(passage, chunks)
    assert resolved == ["chunk-doc1-0", "chunk-doc1-1", "chunk-doc1-2"]


def test_boundary_4_head_boundary_alignment() -> None:
    """Boundary 4: Passage aligns exactly at the head (start index 0) of a chunk."""
    chunks = [
        make_chunk(
            chunk_id="chunk-doc1-0",
            doc_id="doc1",
            content="Alpha findings: RNA sequencing revealed unique cell differentiation patterns.",
            start_char=0,
            end_char=85,
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-doc1-1",
            doc_id="doc1",
            content="Beta findings: Control groups exhibited standard baseline degradation curves.",
            start_char=86,
            end_char=163,
            chunk_index=1,
        ),
    ]

    passage = GroundTruthPassage(
        doc_id="doc1",
        text_snippet="Beta findings: Control groups exhibited standard baseline",
        start_char=86,
        end_char=143,
    )

    resolved = GroundTruthChunkResolver.resolve_chunks_for_passage(passage, chunks)
    assert resolved == ["chunk-doc1-1"]


def test_boundary_5_tail_boundary_alignment() -> None:
    """Boundary 5: Passage aligns exactly at the tail (end_char) of a chunk."""
    chunks = [
        make_chunk(
            chunk_id="chunk-doc1-0",
            doc_id="doc1",
            content="Previous literature failed to account for confounding noise factors.",
            start_char=0,
            end_char=83,
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-doc1-1",
            doc_id="doc1",
            content="Consequently, retrospective re-analysis was needed for historical validity.",
            start_char=84,
            end_char=166,
            chunk_index=1,
        ),
    ]

    passage = GroundTruthPassage(
        doc_id="doc1",
        text_snippet="confounding environmental noise factors.",
        start_char=43,
        end_char=83,
    )

    resolved = GroundTruthChunkResolver.resolve_chunks_for_passage(passage, chunks)
    assert resolved == ["chunk-doc1-0"]


def test_boundary_6_unicode_indic_bengali() -> None:
    """Boundary 6: Resolves Unicode Bengali script with diacritics and complex graphemes."""
    bengali_text_0 = "রিসার্চ ল্যাবরেটরিতে কৃত্রিম বুদ্ধিমত্তা মডেলের কার্যকারিতা মূল্যায়ন করা হচ্ছে।"
    bengali_text_1 = "বাংলা প্রাকৃতিক ভাষা প্রক্রিয়াকরণ ক্ষেত্রে নতুন দিগন্ত উন্মোচিত হয়েছে।"

    len_0 = len(bengali_text_0)
    len_1 = len(bengali_text_1)

    chunks = [
        make_chunk(
            chunk_id="chunk-bn-0",
            doc_id="bn_corpus",
            content=bengali_text_0,
            start_char=0,
            end_char=len_0,
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-bn-1",
            doc_id="bn_corpus",
            content=bengali_text_1,
            start_char=len_0 + 1,
            end_char=len_0 + 1 + len_1,
            chunk_index=1,
        ),
    ]

    passage = GroundTruthPassage(
        doc_id="bn_corpus",
        text_snippet="বাংলা প্রাকৃতিক ভাষা প্রক্রিয়াকরণ ক্ষেত্রে নতুন দিগন্ত",
        start_char=len_0 + 1,
        end_char=len_0 + 1 + 54,
    )

    resolved = GroundTruthChunkResolver.resolve_chunks_for_passage(passage, chunks)
    assert resolved == ["chunk-bn-1"]


def test_boundary_7_cross_doc_disambiguation() -> None:
    """Boundary 7: Identical text across multiple docs resolves strictly to the target doc_id."""
    shared_text = "The quick brown fox jumps over the lazy dog."

    chunks = [
        make_chunk(
            chunk_id="chunk-docA-0",
            doc_id="docA",
            content=f"Report A: {shared_text}",
            start_char=0,
            end_char=54,
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-docB-0",
            doc_id="docB",
            content=f"Report B: {shared_text}",
            start_char=0,
            end_char=54,
            chunk_index=0,
        ),
    ]

    passage_a = GroundTruthPassage(
        doc_id="docA",
        text_snippet=shared_text,
    )
    resolved_a = GroundTruthChunkResolver.resolve_chunks_for_passage(passage_a, chunks)
    assert resolved_a == ["chunk-docA-0"]

    passage_b = GroundTruthPassage(
        doc_id="docB",
        text_snippet=shared_text,
    )
    resolved_b = GroundTruthChunkResolver.resolve_chunks_for_passage(passage_b, chunks)
    assert resolved_b == ["chunk-docB-0"]


def test_boundary_8_intra_doc_repeated_passages() -> None:
    """Boundary 8: Repeated text in the same doc is disambiguated by character offsets."""
    repeated = "Target phrase occurs here."

    chunks = [
        make_chunk(
            chunk_id="chunk-rep-early",
            doc_id="doc_repeat",
            content=f"Introduction: {repeated} Additional introductory context here.",
            start_char=0,
            end_char=70,
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-rep-mid",
            doc_id="doc_repeat",
            content="Middle section discussion unrelated to target phrase.",
            start_char=71,
            end_char=125,
            chunk_index=1,
        ),
        make_chunk(
            chunk_id="chunk-rep-late",
            doc_id="doc_repeat",
            content=f"Conclusion summary: {repeated} Final conclusions and closing remarks.",
            start_char=126,
            end_char=197,
            chunk_index=2,
        ),
    ]

    passage_late = GroundTruthPassage(
        doc_id="doc_repeat",
        text_snippet=repeated,
        start_char=146,
        end_char=172,
    )
    resolved_late = GroundTruthChunkResolver.resolve_chunks_for_passage(passage_late, chunks)
    assert resolved_late == ["chunk-rep-late"]

    passage_early = GroundTruthPassage(
        doc_id="doc_repeat",
        text_snippet=repeated,
        start_char=14,
        end_char=40,
    )
    resolved_early = GroundTruthChunkResolver.resolve_chunks_for_passage(passage_early, chunks)
    assert resolved_early == ["chunk-rep-early"]


def test_resolve_query_full_e2e_with_boundary_passages() -> None:
    """End-to-end resolve_query verifies passage-to-chunk mapping across multiple passages."""
    chunks = [
        make_chunk(
            chunk_id="c0",
            doc_id="d1",
            content="Fact one is true.",
            start_char=0,
            end_char=17,
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="c1",
            doc_id="d1",
            content="Fact two is also true.",
            start_char=18,
            end_char=40,
            chunk_index=1,
        ),
    ]
    query = BenchmarkQuery(
        query_id="q_multi",
        query="What facts are established?",
        ground_truth_answer="Fact one and two.",
        ground_truth_passages=[
            GroundTruthPassage(
                doc_id="d1", text_snippet="Fact one is true.", start_char=0, end_char=17
            ),
            GroundTruthPassage(
                doc_id="d1", text_snippet="Fact two is also true.", start_char=18, end_char=40
            ),
        ],
    )
    eval_query = GroundTruthChunkResolver.resolve_query(query, chunks)
    assert eval_query.query_id == "q_multi"
    assert set(eval_query.ground_truth_chunks) == {"c0", "c1"}
