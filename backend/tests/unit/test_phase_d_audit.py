"rPhase D retrospective supervisor audit verification tests."

from typing import Any

import pytest

from app.engine.context import (
    ContextBuilder,
    reorder_lost_in_the_middle,
)
from app.engine.query_transforms import (
    BaseLLMClient,
    BaseQueryTransformer,
    HyDETransformer,
    IdentityTransformer,
    MockLLMClient,
    MultiQueryExpander,
    QueryTransformationResult,
    StepBackTransformer,
    TransformedRetriever,
)
from app.engine.retrievers.bm25 import BM25Retriever
from app.schemas.chunk import DocumentChunk


class FailingLLMClient(BaseLLMClient):
    """Mock LLM client simulating network or API failures."""

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        raise RuntimeError("Simulated LLM API failure")


@pytest.fixture
def sample_corpus() -> list[DocumentChunk]:
    return [
        DocumentChunk.create(
            doc_id="doc_alpha",
            chunk_index=0,
            content="FastAPI uses Starlette for async routing and Pydantic for data validation.",
            token_count=12,
            start_char=0,
            end_char=75,
            strategy="fixed",
            chunk_id="c_alpha",
        ),
        DocumentChunk.create(
            doc_id="doc_beta",
            chunk_index=0,
            content="PostgreSQL utilizes MVCC for ACID safety.",
            token_count=11,
            start_char=0,
            end_char=44,
            strategy="fixed",
            chunk_id="c_beta",
        ),
        DocumentChunk.create(
            doc_id="doc_gamma",
            chunk_index=0,
            content="Asynchronous web servers rely on non-blocking I/O event loops.",
            token_count=10,
            start_char=0,
            end_char=62,
            strategy="fixed",
            chunk_id="c_gamma",
        ),
    ]


# 1. Base Transformer Contract
def test_base_transformer_contract() -> None:
    assert issubclass(IdentityTransformer, BaseQueryTransformer)
    assert issubclass(HyDETransformer, BaseQueryTransformer)
    assert issubclass(MultiQueryExpander, BaseQueryTransformer)
    assert issubclass(StepBackTransformer, BaseQueryTransformer)


# 2. Identity Transformer Audit
def test_identity_transformer_multilingual_and_edge_cases() -> None:
    transformer = IdentityTransformer()
    cases = [
        "Standard natural language query",
        "",
        "Unicode test: café, naïve, facçade, Schrödinger",
        "Τύ﹄﹑﹔︼︄﹃︨",  # Bengali unicode placeholder
        "def retrieve(q: str, top_k: int = 10) -> list[tuple[DocumentChunk, float]]:",
        "   Leading and trailing whitespace query   ",
    ]
    for query in cases:
        res = transformer.transform(query)
        assert isinstance(res, QueryTransformationResult)
        assert res.original_query == query
        assert res.transformed_queries == [query]
        assert res.strategy == "none"
        assert res.metadata == {}

    # Repeated execution produces bitwise identical results
    for _ in range(10):
        repeat_res = transformer.transform("bengali_query")
        assert repeat_res.transformed_queries == ["bengali_query"]


# 3. HyDE Transformer Audit
def test_hyde_transformer_mock_and_failures() -> None:
    canned = {"Question: What is concurrency?": "Concurrency is the execution of multiple tasks."}
    mock = MockLLMClient(canned_responses=canned)
    hyde = HyDETransformer(llm_client=mock)

    # 1. Prompt formatting verification
    assert "{query}" in hyde.prompt_template
    formatted = hyde.prompt_template.format(query="What is concurrency?")
    assert "What is concurrency?" in formatted

    # 2. Successful transformation
    res = hyde.transform("What is concurrency?")
    assert res.original_query == "What is concurrency?"
    assert res.strategy == "hyde"
    assert len(res.transformed_queries) == 1
    assert "Concurrency is the execution" in res.transformed_queries[0]
    assert res.metadata["hypothetical_passage"] == res.transformed_queries[0]

    # 3. Empty LLM response handling
    empty_mock = MockLLMClient(canned_responses={"Question: empty": "   "})
    hyde_empty = HyDETransformer(llm_client=empty_mock)
    res_empty = hyde_empty.transform("empty")
    assert res_empty.transformed_queries == [""]

    # 4. LLM failure propagation
    failing_hyde = HyDETransformer(llm_client=FailingLLMClient())
    with pytest.raises(RuntimeError, match="Simulated LLM API failure"):
        failing_hyde.transform("Will fail")


# 4. Multi-Query Expander Audit
def test_multi_query_expander_scenarios(sample_corpus: list[DocumentChunk]) -> None:
    # Scenario: requested = 3, actual = 3
    mock_3 = MockLLMClient(
        canned_responses={
            "Original question: test3": "1. Alternative 1\n2. Alternative 2\n3. Alternative 3"
        }
    )
    expander_3 = MultiQueryExpander(llm_client=mock_3, num_queries=3, include_original=True)
    res_3 = expander_3.transform("test3")
    assert res_3.original_query == "test3"
    assert res_3.transformed_queries == ["test3", "Alternative 1", "Alternative 2", "Alternative 3"]
    assert res_3.metadata["generated_count"] == 4

    # Scenario: requested = 3, actual = 1
    mock_1 = MockLLMClient(canned_responses={"Original question: test1": "- Just one alternative"})
    expander_1 = MultiQueryExpander(llm_client=mock_1, num_queries=3, include_original=True)
    res_1 = expander_1.transform("test1")
    assert res_1.transformed_queries == ["test1", "Just one alternative"]

    # Malformed output with bullets, dots, empty lines
    mock_malformed = MockLLMClient(
        canned_responses={
            "Original question: malformed": "\n\n* Bullet Alpha\n1. Numbered Beta\n- Dash Gamma\n\n"
        }
    )
    expander_malformed = MultiQueryExpander(
        llm_client=mock_malformed, num_queries=3, include_original=False
    )
    res_malformed = expander_malformed.transform("malformed")
    assert res_malformed.transformed_queries == ["Bullet Alpha", "Numbered Beta", "Dash Gamma"]

    # LLM failure propagation
    failing_expander = MultiQueryExpander(llm_client=FailingLLMClient())
    with pytest.raises(RuntimeError, match="Simulated LLM API failure"):
        failing_expander.transform("fail")

    # Verify generated queries can be passed independently to a retriever
    retriever = BM25Retriever(sample_corpus)
    for q in res_3.transformed_queries:
        sub_results = retriever.retrieve(q, top_k=2)
        assert isinstance(sub_results, list)


# 5. Step-Back Transformer Audit
def test_step_back_transformer_scenarios() -> None:
    mock = MockLLMClient(
        canned_responses={
            "Specific question: Why does my query fail on index scan?": (
                "How do database index scans and query planners work?"
            )
        }
    )
    step_back = StepBackTransformer(llm_client=mock, include_original=True)

    # Prompt check: verifies it asks for high-level conceptual question
    assert "conceptual question" in step_back.prompt_template
    assert "background context" in step_back.prompt_template

    res = step_back.transform("Why does my query fail on index scan?")
    assert res.original_query == "Why does my query fail on index scan?"
    assert res.strategy == "step_back"
    assert len(res.transformed_queries) == 2
    assert "query planners work" in res.transformed_queries[0]
    assert res.transformed_queries[1] == "Why does my query fail on index scan?"
    assert "step_back_query" in res.metadata

    # include_original = False
    step_back_no_orig = StepBackTransformer(llm_client=mock, include_original=False)
    res_no_orig = step_back_no_orig.transform("Why does my query fail on index scan?")
    assert len(res_no_orig.transformed_queries) == 1
    assert "query planners work" in res_no_orig.transformed_queries[0]

    # Failure propagation
    failing_sb = StepBackTransformer(llm_client=FailingLLMClient())
    with pytest.raises(RuntimeError, match="Simulated LLM API failure"):
        failing_sb.transform("error query")


# 6 & 7. TransformedRetriever & Original Ranking Preservation
def test_transformed_retriever_ranking_preservation(sample_corpus: list[DocumentChunk]) -> None:
    mock = MockLLMClient(
        canned_responses={
            "Specific question: Asynchronous Python": (
                "How does the event loop handle I/O in async runtimes?"
            )
        }
    )
    bm25 = BM25Retriever(sample_corpus)
    step_back = StepBackTransformer(llm_client=mock, include_original=True)
    t_retriever = TransformedRetriever(retriever=bm25, transformer=step_back)

    # 1. Direct baseline retrieval on underlying retriever
    baseline_ranking = t_retriever.retriever.retrieve("Asynchronous Python", top_k=3)
    baseline_ids = [c.chunk_id for c, _ in baseline_ranking]

    # 2. Transformed retrieval with Step-Back RRF fusion
    transformed_ranking = t_retriever.retrieve("Asynchronous Python", top_k=3)
    transformed_ids = [c.chunk_id for c, _ in transformed_ranking]

    # Transformed query history is preserved
    assert t_retriever.last_transformation is not None
    assert t_retriever.last_transformation.strategy == "step_back"
    assert len(t_retriever.last_transformation.transformed_queries) == 2

    # Baseline retriever was not mutated and remains identical
    repeat_baseline = t_retriever.retriever.retrieve("Asynchronous Python", top_k=3)
    assert [c.chunk_id for c, _ in repeat_baseline] == baseline_ids

    # Researcher can inspect both baseline and transformed rankings
    assert len(baseline_ids) > 0
    assert len(transformed_ids) > 0


# 8 & 9. ContextBuilder Budget & Edge Cases
def test_context_builder_edge_cases() -> None:
    long_text = "passage exceeding token budget ceiling on deliberate test. " * 10
    chunks = [
        (
            DocumentChunk.create(
                doc_id="doc1",
                chunk_index=0,
                content=long_text,
                token_count=100,
                start_char=0,
                end_char=len(long_text),
                strategy="fixed",
                chunk_id="c_long",
            ),
            0.9,
        ),
        (
            DocumentChunk.create(
                doc_id="doc2",
                chunk_index=0,
                content="Small concise snippet.",
                token_count=5,
                start_char=0,
                end_char=22,
                strategy="fixed",
                chunk_id="c_short",
            ),
            0.8,
        ),
    ]

    # Case: First chunk exceeds budget, second fits
    builder_selective = ContextBuilder(token_budget=50, reorder_strategy="standard")
    packed = builder_selective.build(chunks)
    assert packed.total_tokens <= 50
    assert packed.included_chunks_count == 1
    assert packed.chunks[0].chunk_id == "c_short"
    assert packed.truncated_chunks_count == 1

    # Case: No chunks fit (budget too small even for smallest chunk)
    builder_tiny = ContextBuilder(token_budget=5)
    packed_empty = builder_tiny.build(chunks)
    assert packed_empty.text == ""
    assert packed_empty.total_tokens == 0
    assert packed_empty.included_chunks_count == 0
    assert packed_empty.truncated_chunks_count == 2

    # Case: Empty input
    builder_normal = ContextBuilder(token_budget=500)
    packed_none = builder_normal.build([])
    assert packed_none.text == ""
    assert packed_none.total_tokens == 0
    assert packed_none.included_chunks_count == 0


# 10. Lost-in-the-Middle Mathematical Verification for all lengths
def test_lost_in_the_middle_comprehensive() -> None:
    assert reorder_lost_in_the_middle([]) == []
    assert reorder_lost_in_the_middle(["d1"]) == ["d1"]
    assert reorder_lost_in_the_middle(["d1", "d2"]) == ["d1", "d2"]
    assert reorder_lost_in_the_middle(["d1", "d2", "d3"]) == ["d1", "d3", "d2"]
    assert reorder_lost_in_the_middle(["d1", "d2", "d3", "d4"]) == ["d1", "d3", "d4", "d2"]
    assert reorder_lost_in_the_middle(["d1", "d2", "d3", "d4", "d5"]) == [
        "d1",
        "d3",
        "d5",
        "d4",
        "d2",
    ]
    assert reorder_lost_in_the_middle(["d1", "d2", "d3", "d4", "d5", "d6"]) == [
        "d1",
        "d3",
        "d5",
        "d6",
        "d4",
        "d2",
    ]
    assert reorder_lost_in_the_middle(["d1", "d2", "d3", "d4", "d5", "d6", "d7"]) == [
        "d1",
        "d3",
        "d5",
        "d7",
        "d6",
        "d4",
        "d2",
    ]

    # Determinism across repeated executions
    seq = [f"item_{i}" for i in range(10)]
    expected = reorder_lost_in_the_middle(seq)
    for _ in range(10):
        assert reorder_lost_in_the_middle(seq) == expected


# 11. Context Ordering Traceability
def test_context_ordering_traceability(sample_corpus: list[DocumentChunk]) -> None:
    candidates = [
        (sample_corpus[0], 0.95),  # original rank 1
        (sample_corpus[1], 0.85),  # original rank 2
        (sample_corpus[2], 0.75),  # original rank 3
    ]
    builder = ContextBuilder(token_budget=1000, reorder_strategy="lost_in_the_middle")
    packed = builder.build(candidates)

    # 3 items reordered: [d1, d3, d2]
    assert packed.chunks[0].chunk_id == "c_alpha"
    assert packed.chunks[0].original_rank == 1
    assert packed.chunks[0].source_index == 1

    assert packed.chunks[1].chunk_id == "c_gamma"
    assert packed.chunks[1].original_rank == 3
    assert packed.chunks[1].source_index == 2

    assert packed.chunks[2].chunk_id == "c_beta"
    assert packed.chunks[2].original_rank == 2
    assert packed.chunks[2].source_index == 3


# 12. Research Reproducibility: 10 repeated executions
def test_phase_d_pipeline_reproducibility_10_runs(sample_corpus: list[DocumentChunk]) -> None:
    canned = {"Specific question: Concurrency": "How do thread pools and event loops differ?"}
    mock = MockLLMClient(canned_responses=canned)
    bm25 = BM25Retriever(sample_corpus)
    step_back = StepBackTransformer(llm_client=mock, include_original=True)
    t_retriever = TransformedRetriever(retriever=bm25, transformer=step_back)
    builder = ContextBuilder(token_budget=1000, reorder_strategy="lost_in_the_middle")

    baseline_output = None
    for run_idx in range(10):
        results = t_retriever.retrieve("Concurrency", top_k=3)
        packed = builder.build(results)

        output_snapshot = (
            [c.chunk_id for c, _ in results],
            [score for _, score in results],
            packed.text,
            packed.total_tokens,
            [c.chunk_id for c in packed.chunks],
            [c.source_index for c in packed.chunks],
            [c.original_rank for c in packed.chunks],
        )

        if baseline_output is None:
            baseline_output = output_snapshot
        else:
            # Must be bitwise identical on every single iteration
            assert output_snapshot == baseline_output, (
                f"Non-deterministic run detected at iteration {run_idx}"
            )
