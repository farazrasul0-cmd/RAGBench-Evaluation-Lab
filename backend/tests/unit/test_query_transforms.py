"""Unit tests for query transformations and transformed retriever composition."""

import pytest

from app.engine.query_transforms import (
    HyDETransformer,
    IdentityTransformer,
    MockLLMClient,
    MultiQueryExpander,
    StepBackTransformer,
    TransformedRetriever,
    get_query_transformer,
)
from app.engine.retrievers.bm25 import BM25Retriever
from app.schemas.chunk import DocumentChunk


@pytest.fixture
def mock_llm() -> MockLLMClient:
    canned = {
        "Please write a comprehensive": (
            "FastAPI leverages Python type hints and Pydantic for high throughput."
        ),
        "versions of the given technical question": (
            "1. How does FastAPI achieve high performance with ASGI?\n"
            "2. What are the key architectural features of FastAPI?\n"
            "3. High performance web frameworks in Python comparison\n"
        ),
        "formulate a more general, conceptual question": (
            "How do modern asynchronous web frameworks achieve concurrency?"
        ),
    }
    return MockLLMClient(canned_responses=canned)


@pytest.fixture
def corpus_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk.create(
            doc_id="d1",
            chunk_index=0,
            content="FastAPI leverages Starlette and Pydantic for asynchronous web APIs.",
            token_count=10,
            start_char=0,
            end_char=71,
            strategy="fixed",
            chunk_id="c-fastapi",
        ),
        DocumentChunk.create(
            doc_id="d2",
            chunk_index=0,
            content="PostgreSQL handles ACID transactions with robust multi-version concurrency.",
            token_count=10,
            start_char=0,
            end_char=75,
            strategy="fixed",
            chunk_id="c-postgres",
        ),
        DocumentChunk.create(
            doc_id="d3",
            chunk_index=0,
            content="Modern asynchronous web frameworks use event loops for high throughput.",
            token_count=11,
            start_char=0,
            end_char=79,
            strategy="fixed",
            chunk_id="c-async",
        ),
    ]


def test_identity_transformer() -> None:
    transformer = IdentityTransformer()
    assert transformer.strategy_name == "none"

    result = transformer.transform("What is FastAPI?")
    assert result.original_query == "What is FastAPI?"
    assert result.strategy == "none"
    assert result.transformed_queries == ["What is FastAPI?"]


def test_hyde_transformer(mock_llm: MockLLMClient) -> None:
    transformer = HyDETransformer(llm_client=mock_llm)
    assert transformer.strategy_name == "hyde"

    result = transformer.transform("FastAPI performance")
    assert result.original_query == "FastAPI performance"
    assert len(result.transformed_queries) == 1
    assert "FastAPI leverages Python type hints" in result.transformed_queries[0]
    assert "hypothetical_passage" in result.metadata


def test_multi_query_expander(mock_llm: MockLLMClient) -> None:
    transformer = MultiQueryExpander(llm_client=mock_llm, num_queries=3, include_original=True)
    assert transformer.strategy_name == "multi_query"

    result = transformer.transform("FastAPI performance")
    assert result.original_query == "FastAPI performance"
    # Should include original plus 3 alternatives (or max 4)
    assert len(result.transformed_queries) >= 3
    assert result.transformed_queries[0] == "FastAPI performance"
    assert any("ASGI" in q for q in result.transformed_queries)


def test_step_back_transformer(mock_llm: MockLLMClient) -> None:
    transformer = StepBackTransformer(llm_client=mock_llm, include_original=True)
    assert transformer.strategy_name == "step_back"

    result = transformer.transform("Why is FastAPI fast?")
    assert len(result.transformed_queries) == 2
    assert "asynchronous web frameworks" in result.transformed_queries[0]
    assert result.transformed_queries[1] == "Why is FastAPI fast?"


def test_transformed_retriever_single_query(
    mock_llm: MockLLMClient, corpus_chunks: list[DocumentChunk]
) -> None:
    bm25 = BM25Retriever(corpus_chunks)
    hyde = HyDETransformer(llm_client=mock_llm)
    t_retriever = TransformedRetriever(retriever=bm25, transformer=hyde)

    results = t_retriever.retrieve("FastAPI performance", top_k=2)
    assert len(results) > 0
    assert results[0][0].chunk_id == "c-fastapi"
    assert t_retriever.last_transformation is not None
    assert t_retriever.last_transformation.strategy == "hyde"


def test_transformed_retriever_multi_query_fusion(
    mock_llm: MockLLMClient, corpus_chunks: list[DocumentChunk]
) -> None:
    bm25 = BM25Retriever(corpus_chunks)
    multi_q = MultiQueryExpander(llm_client=mock_llm, num_queries=3)
    t_retriever = TransformedRetriever(retriever=bm25, transformer=multi_q, rrf_k=60)

    results = t_retriever.retrieve("FastAPI performance", top_k=2)
    assert len(results) > 0
    assert results[0][0].chunk_id == "c-fastapi"
    # Deduplication and valid scores
    chunk_ids = [c.chunk_id for c, _ in results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_get_query_transformer_factory(mock_llm: MockLLMClient) -> None:
    t_none = get_query_transformer("none")
    assert isinstance(t_none, IdentityTransformer)

    t_hyde = get_query_transformer("hyde", llm_client=mock_llm)
    assert isinstance(t_hyde, HyDETransformer)

    with pytest.raises(ValueError):
        get_query_transformer("hyde", llm_client=None)

    with pytest.raises(ValueError):
        get_query_transformer("invalid_strategy")
