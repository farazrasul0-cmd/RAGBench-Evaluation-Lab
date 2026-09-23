"""Unit tests for context window builder, budget enforcement, and Lost-in-the-Middle reordering."""

import pytest

from app.engine.context import (
    ContextBuilder,
    ContextBuilderError,
    reorder_lost_in_the_middle,
    reorder_standard,
)
from app.schemas.chunk import DocumentChunk


@pytest.fixture
def ranked_candidates() -> list[tuple[DocumentChunk, float]]:
    chunks: list[tuple[DocumentChunk, float]] = []
    contents = [
        "First most relevant passage discussing foundational architecture principles.",
        "Second passage detailing configuration parameters and performance tuning.",
        "Third passage explaining database replication and transactional safety.",
        "Fourth passage covering security policies, TLS certificates, and auth.",
        "Fifth passage summarizing deployment pipelines and container orchestration.",
        "Sixth passage providing diagnostic commands and troubleshooting playbooks.",
    ]
    for idx, text in enumerate(contents, start=1):
        ch = DocumentChunk.create(
            doc_id=f"doc_{idx}.pdf",
            chunk_index=0,
            content=text,
            token_count=10,
            start_char=0,
            end_char=len(text),
            strategy="fixed",
            chunk_id=f"chunk_00{idx}",
            metadata={"priority": idx},
        )
        score = 1.0 - (idx * 0.1)
        chunks.append((ch, score))
    return chunks


def test_reorder_standard() -> None:
    items = ["d1", "d2", "d3", "d4", "d5"]
    assert reorder_standard(items) == ["d1", "d2", "d3", "d4", "d5"]


def test_reorder_lost_in_the_middle_formula() -> None:
    # 5 items: [d1, d2, d3, d4, d5] -> [d1, d3, d5, d4, d2]
    # Even indices (0, 2, 4) -> [d1, d3, d5]
    # Odd indices (1, 3) -> [d2, d4] reversed -> [d4, d2]
    items_5 = ["d1", "d2", "d3", "d4", "d5"]
    assert reorder_lost_in_the_middle(items_5) == ["d1", "d3", "d5", "d4", "d2"]

    # 6 items: [d1, d2, d3, d4, d5, d6] -> [d1, d3, d5, d6, d4, d2]
    # Even indices (0, 2, 4) -> [d1, d3, d5]
    # Odd indices (1, 3, 5) -> [d2, d4, d6] reversed -> [d6, d4, d2]
    items_6 = ["d1", "d2", "d3", "d4", "d5", "d6"]
    assert reorder_lost_in_the_middle(items_6) == ["d1", "d3", "d5", "d6", "d4", "d2"]

    # Small lists
    assert reorder_lost_in_the_middle(["d1"]) == ["d1"]
    assert reorder_lost_in_the_middle(["d1", "d2"]) == ["d1", "d2"]


def test_context_builder_strict_token_budget(
    ranked_candidates: list[tuple[DocumentChunk, float]],
) -> None:
    # Tight budget: 60 tokens should fit roughly 1 or 2 chunks with headers
    builder_tight = ContextBuilder(token_budget=60, reorder_strategy="standard")
    packed_tight = builder_tight.build(ranked_candidates)

    assert packed_tight.total_tokens <= 60
    assert packed_tight.included_chunks_count > 0
    assert packed_tight.truncated_chunks_count > 0
    assert packed_tight.included_chunks_count + packed_tight.truncated_chunks_count == len(
        ranked_candidates
    )

    # Ample budget: 2000 tokens should fit all 6 chunks
    builder_ample = ContextBuilder(token_budget=2000, reorder_strategy="standard")
    packed_ample = builder_ample.build(ranked_candidates)

    assert packed_ample.total_tokens <= 2000
    assert packed_ample.included_chunks_count == 6
    assert packed_ample.truncated_chunks_count == 0


def test_context_builder_citation_header_injection(
    ranked_candidates: list[tuple[DocumentChunk, float]],
) -> None:
    builder = ContextBuilder(token_budget=1000, reorder_strategy="standard")
    packed = builder.build(ranked_candidates[:2])

    assert "[Source 1] (doc: doc_1.pdf, chunk: chunk_001)" in packed.text
    assert "[Source 2] (doc: doc_2.pdf, chunk: chunk_002)" in packed.text
    assert len(packed.chunks) == 2
    assert packed.chunks[0].chunk_id == "chunk_001"
    assert packed.chunks[0].metadata == {"priority": 1}


def test_context_builder_lost_in_the_middle(
    ranked_candidates: list[tuple[DocumentChunk, float]],
) -> None:
    builder = ContextBuilder(token_budget=2000, reorder_strategy="lost_in_the_middle")
    packed = builder.build(ranked_candidates)

    # 6 candidates reordered: [d1, d3, d5, d6, d4, d2]
    chunk_ids = [c.chunk_id for c in packed.chunks]
    expected_order = [
        "chunk_001",  # d1
        "chunk_003",  # d3
        "chunk_005",  # d5
        "chunk_006",  # d6
        "chunk_004",  # d4
        "chunk_002",  # d2
    ]
    assert chunk_ids == expected_order

    # Provenance tracks original rank
    assert packed.chunks[0].original_rank == 1
    assert packed.chunks[-1].original_rank == 2


def test_context_builder_empty_candidates() -> None:
    builder = ContextBuilder(token_budget=500)
    packed = builder.build([])
    assert packed.text == ""
    assert packed.total_tokens == 0
    assert packed.included_chunks_count == 0
    assert packed.truncated_chunks_count == 0


def test_context_builder_invalid_inputs() -> None:
    with pytest.raises(ContextBuilderError):
        ContextBuilder(token_budget=0)

    with pytest.raises(ContextBuilderError):
        ContextBuilder(reorder_strategy="unsupported_strategy")
