"""Unit and integration tests for Phase F4: Single-Run Pipeline Executor."""

import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.db import drop_db, get_async_engine, get_session_factory, init_db
from app.db.repositories.dataset import DatasetRepository
from app.db.repositories.experiment import ExperimentRepository
from app.db.repositories.query_trace import QueryTraceRepository
from app.engine.embeddings.mock_provider import DeterministicMockEmbeddingProvider
from app.engine.query_transforms.base import MockLLMClient
from app.engine.query_transforms.step_back import StepBackTransformer
from app.engine.rerankers.mock_reranker import DeterministicMockReranker
from app.engine.retrievers.bm25 import BM25Retriever
from app.engine.retrievers.dense import DenseRetriever
from app.engine.retrievers.hybrid import HybridRetriever
from app.engine.runner import (
    EvaluationQuery,
    SingleRunExecutor,
    SingleRunResult,
    build_pipeline_components,
)
from app.schemas.chunk import DocumentChunk
from app.schemas.experiment import (
    ContextConfig,
    DatasetConfig,
    EmbeddingConfig,
    EvaluationConfig,
    GenerationConfig,
    PipelineConfig,
    QueryTransformConfig,
    RerankerConfig,
    RetrievalConfig,
)


@pytest.fixture
async def test_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create isolated SQLite database engine with WAL & foreign keys."""
    db_file = Path("test_runner.db")
    engine = get_async_engine(f"sqlite+aiosqlite:///{db_file}")
    await init_db(engine)
    try:
        yield engine
    finally:
        await drop_db(engine)
        await engine.dispose()
        for f in (db_file, Path(f"{db_file}-wal"), Path(f"{db_file}-shm")):
            if f.exists():
                with contextlib.suppress(OSError):
                    f.unlink()


@pytest.fixture
async def session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Yield async session for testing."""
    session_factory = get_session_factory(test_engine)
    async with session_factory() as s:
        yield s
        await s.rollback()


def create_sample_chunks() -> list[DocumentChunk]:
    """Helper to create sample DocumentChunk instances."""
    return [
        DocumentChunk.create(
            doc_id="doc-1",
            chunk_index=0,
            content=(
                "Photosynthesis is the process by which green plants "
                "convert sunlight into chemical energy."
            ),
            token_count=15,
            start_char=0,
            end_char=93,
            strategy="fixed",
            chunk_id="chunk-photo-001",
        ),
        DocumentChunk.create(
            doc_id="doc-1",
            chunk_index=1,
            content=(
                "Chlorophyll is the pigment in chloroplasts responsible for absorbing light during"
                " photosynthesis."
            ),
            token_count=14,
            start_char=94,
            end_char=191,
            strategy="fixed",
            chunk_id="chunk-chloro-002",
        ),
        DocumentChunk.create(
            doc_id="doc-2",
            chunk_index=0,
            content=(
                "Cellular respiration converts glucose into ATP in the "
                "mitochondria of eukaryotic cells."
            ),
            token_count=13,
            start_char=0,
            end_char=87,
            strategy="fixed",
            chunk_id="chunk-resp-003",
        ),
    ]


def test_build_pipeline_components_modes() -> None:
    """Verify component factory instantiates dense, bm25, hybrid, reranker, and transforms."""
    chunks = create_sample_chunks()
    mock_emb = DeterministicMockEmbeddingProvider(dimension=64)

    # 1. BM25 mode
    cfg_bm25 = PipelineConfig(
        dataset=DatasetConfig(dataset_id="bio", dataset_version_id="v1"),
        retrieval=RetrievalConfig(mode="bm25", top_k=5),
        reranker=RerankerConfig(enabled=False),
        query_transform=QueryTransformConfig(strategy="none"),
    )
    comp_bm25 = build_pipeline_components(cfg_bm25, corpus_chunks=chunks)
    assert isinstance(comp_bm25.retriever, BM25Retriever)
    assert comp_bm25.reranker is None

    # 2. Dense mode
    cfg_dense = PipelineConfig(
        dataset=DatasetConfig(dataset_id="bio", dataset_version_id="v1"),
        embedding=EmbeddingConfig(provider="mock", dimension=64),
        retrieval=RetrievalConfig(mode="dense", top_k=5),
        reranker=RerankerConfig(enabled=True, strategy="mock", top_n=2),
    )
    comp_dense = build_pipeline_components(
        cfg_dense, corpus_chunks=chunks, embedding_provider=mock_emb
    )
    assert isinstance(comp_dense.retriever, DenseRetriever)
    assert isinstance(comp_dense.reranker, DeterministicMockReranker)

    # 3. Hybrid mode with step_back transform
    cfg_hybrid = PipelineConfig(
        dataset=DatasetConfig(dataset_id="bio", dataset_version_id="v1"),
        embedding=EmbeddingConfig(provider="mock", dimension=64),
        retrieval=RetrievalConfig(mode="hybrid", top_k=5, hybrid_fusion="rrf"),
        query_transform=QueryTransformConfig(strategy="step_back"),
        reranker=RerankerConfig(enabled=True, strategy="mock", top_n=2),
    )
    mock_llm = MockLLMClient()
    comp_hybrid = build_pipeline_components(
        cfg_hybrid,
        corpus_chunks=chunks,
        llm_client=mock_llm,
        embedding_provider=mock_emb,
    )
    assert isinstance(comp_hybrid.retriever, HybridRetriever)
    assert isinstance(comp_hybrid.transformer, StepBackTransformer)
    assert isinstance(comp_hybrid.reranker, DeterministicMockReranker)


@pytest.mark.anyio
async def test_single_run_executor_e2e_hybrid_full_lineage(session: AsyncSession) -> None:
    """Full end-to-end execution of a concrete pipeline run with evidence trail reconstruction."""
    dataset_repo = DatasetRepository(session)
    exp_repo = ExperimentRepository(session)
    trace_repo = QueryTraceRepository(session)

    # 1. Seed database hierarchy (Dataset -> Version -> Document -> DocumentChunks)
    ds = await dataset_repo.create_dataset(name="Biology QA", description="Test biology dataset")
    ver = await dataset_repo.create_version(
        dataset_id=ds.id,
        version_number=1,
        content_hash="ver-hash-bio-v1",
    )
    docs = await dataset_repo.add_documents(
        ver.id,
        [
            {
                "external_id": "doc-1",
                "filename": "photosynthesis.txt",
                "content": "Photosynthesis details and chlorophyll information.",
                "content_hash": "hash-doc-1",
            },
            {
                "external_id": "doc-2",
                "filename": "respiration.txt",
                "content": "Cellular respiration and mitochondria ATP information.",
                "content_hash": "hash-doc-2",
            },
        ],
    )
    chunks = create_sample_chunks()
    await dataset_repo.add_chunks(
        docs[0].id,
        [
            {
                "id": "chunk-photo-001",
                "chunk_index": 0,
                "content": chunks[0].content,
                "content_hash": "hash-cp-001",
                "token_count": 15,
                "strategy": "fixed",
                "chunking_config_hash": "chunk-cfg-hash",
            },
            {
                "id": "chunk-chloro-002",
                "chunk_index": 1,
                "content": chunks[1].content,
                "content_hash": "hash-cc-002",
                "token_count": 14,
                "strategy": "fixed",
                "chunking_config_hash": "chunk-cfg-hash",
            },
        ],
    )
    await dataset_repo.add_chunks(
        docs[1].id,
        [
            {
                "id": "chunk-resp-003",
                "chunk_index": 0,
                "content": chunks[2].content,
                "content_hash": "hash-cr-003",
                "token_count": 13,
                "strategy": "fixed",
                "chunking_config_hash": "chunk-cfg-hash",
            }
        ],
    )

    # 2. Create Experiment
    exp = await exp_repo.create_experiment(
        name="Hybrid + Reranker + StepBack Evaluation",
        dataset_version_id=ver.id,
        configuration={"name": "e2e_exp"},
        configuration_hash="exp-config-hash",
    )

    # 3. Configure concrete PipelineConfig
    pipeline_cfg = PipelineConfig(
        dataset=DatasetConfig(dataset_id=ds.id, dataset_version_id=ver.id),
        embedding=EmbeddingConfig(provider="mock", dimension=64),
        retrieval=RetrievalConfig(mode="hybrid", top_k=3, hybrid_fusion="rrf", rrf_k=60),
        reranker=RerankerConfig(enabled=True, strategy="mock", top_n=2),
        query_transform=QueryTransformConfig(
            strategy="step_back", num_queries=2, include_original=True
        ),
        context=ContextConfig(token_budget=512, reorder_strategy="standard"),
        generation=GenerationConfig(model_name="mock"),
        evaluation=EvaluationConfig(
            metrics=["recall", "precision", "mrr", "ndcg", "hit", "faithfulness", "citations"],
            k_values=[1, 2, 3],
        ),
    )

    # 4. Define evaluation queries
    queries = [
        EvaluationQuery(
            query_id="q-1",
            query_text="What pigment absorbs sunlight in photosynthesis?",
            expected_answer="Chlorophyll is the pigment responsible for absorbing light.",
            ground_truth_chunks=["chunk-chloro-002"],
            metadata={"domain": "botany", "difficulty": "easy"},
        ),
        EvaluationQuery(
            query_id="q-2",
            query_text="Where is ATP produced in cellular respiration?",
            expected_answer="ATP is produced in the mitochondria.",
            ground_truth_chunks=["chunk-resp-003"],
            metadata={"domain": "cellular_biology", "difficulty": "medium"},
        ),
    ]

    mock_llm = MockLLMClient()
    mock_emb = DeterministicMockEmbeddingProvider(dimension=64)

    # 5. Execute single run
    executor = SingleRunExecutor()
    result: SingleRunResult = await executor.execute(
        experiment_id=exp.id,
        pipeline_config=pipeline_cfg,
        queries=queries,
        corpus_chunks=chunks,
        session=session,
        llm_client=mock_llm,
        embedding_provider=mock_emb,
        validate_chunk_references=True,
    )

    # 6. Verify SingleRunResult
    assert result.status == "COMPLETED"
    assert result.total_queries == 2
    assert result.completed_queries == 2
    assert result.failed_queries == 0
    assert result.duration_ms > 0
    assert "recall@1" in result.mean_metrics
    assert "precision@1" in result.mean_metrics
    assert "mrr@1" in result.mean_metrics
    assert "faithfulness" in result.mean_metrics
    assert "citation_precision" in result.mean_metrics

    # 7. Verify relational persistence in DB
    run = await exp_repo.get_run(result.experiment_run_id)
    assert run is not None
    assert run.status == "COMPLETED"
    assert run.pipeline_config_hash == pipeline_cfg.compute_configuration_hash()
    assert run.cache_hash == result.cache_hash
    assert run.completed_at is not None

    # Check RunMetricSummary rollups
    summaries = await exp_repo.get_metric_summaries(run.id)
    assert len(summaries) > 0
    summary_names = {s.metric_name for s in summaries}
    assert "recall@1" in summary_names
    assert "recall@3" in summary_names
    assert "mrr@3" in summary_names
    assert "faithfulness" in summary_names
    assert "citation_precision" in summary_names
    for s in summaries:
        assert s.count == 2
        assert s.min <= s.mean <= s.max

    # 8. Verify complete granular Evidence Trail reconstruction
    trails = await trace_repo.get_run_evidence_trails(run.id)
    assert len(trails) == 2

    q1_trail = next(t for t in trails if t.query_id == "q-1")
    assert q1_trail.status == "SUCCESS"
    assert q1_trail.original_query == "What pigment absorbs sunlight in photosynthesis?"
    assert q1_trail.ground_truth_chunks == ["chunk-chloro-002"]

    # Transformed queries (StepBack produced step-back abstraction + original)
    assert len(q1_trail.transformed_queries) >= 1
    assert any(t["transformation_type"] == "step_back" for t in q1_trail.transformed_queries)

    # Retrieved chunks
    assert len(q1_trail.retrieved_chunks) > 0
    for rc in q1_trail.retrieved_chunks:
        assert rc["retriever_type"] == "hybrid"
        assert rc["chunk_id"] in {"chunk-photo-001", "chunk-chloro-002", "chunk-resp-003"}

    # Reranked chunks
    assert len(q1_trail.reranked_chunks) > 0
    for rk in q1_trail.reranked_chunks:
        assert rk["original_rank"] >= 1
        assert rk["reranked_rank"] >= 1

    # Packed context
    assert q1_trail.packed_context is not None
    assert q1_trail.packed_context["token_budget"] == 512
    assert q1_trail.packed_context["token_count"] <= 512
    assert len(q1_trail.packed_context["chunk_ids"]) > 0

    # Generation result
    assert q1_trail.generation_result is not None
    assert q1_trail.generation_result["model"] == "mock"
    assert len(q1_trail.generation_result["answer"]) > 0
    assert q1_trail.generation_result["input_tokens"] > 0

    # Metric results
    assert "recall@1" in q1_trail.metric_results
    assert "faithfulness" in q1_trail.metric_results
    assert len(q1_trail.detailed_metrics) > 0


@pytest.mark.anyio
async def test_single_run_executor_pure_bm25(session: AsyncSession) -> None:
    """Verify execution of baseline BM25 pipeline without reranker or transforms."""
    dataset_repo = DatasetRepository(session)
    exp_repo = ExperimentRepository(session)
    trace_repo = QueryTraceRepository(session)

    ds = await dataset_repo.create_dataset(name="BM25 DS")
    ver = await dataset_repo.create_version(
        dataset_id=ds.id, version_number=1, content_hash="hash-bm25-v1"
    )
    docs = await dataset_repo.add_documents(
        ver.id,
        [
            {
                "filename": "doc.txt",
                "content": "Sample content about quantum computing.",
                "content_hash": "h-qc",
            }
        ],
    )
    chunks = [
        DocumentChunk.create(
            doc_id=docs[0].id,
            chunk_index=0,
            content="Quantum computing utilizes qubits for superposition.",
            token_count=8,
            start_char=0,
            end_char=52,
            strategy="fixed",
            chunk_id="chunk-qc-1",
        )
    ]
    await dataset_repo.add_chunks(
        docs[0].id,
        [
            {
                "id": "chunk-qc-1",
                "chunk_index": 0,
                "content": chunks[0].content,
                "content_hash": "h-qc-chunk",
                "token_count": 8,
                "strategy": "fixed",
                "chunking_config_hash": "cfg-qc",
            }
        ],
    )
    exp = await exp_repo.create_experiment("BM25 Exp", ver.id, {}, "hash-bm25-cfg")

    pipeline_cfg = PipelineConfig(
        dataset=DatasetConfig(dataset_id=ds.id, dataset_version_id=ver.id),
        retrieval=RetrievalConfig(mode="bm25", top_k=5),
        reranker=RerankerConfig(enabled=False),
        query_transform=QueryTransformConfig(strategy="none"),
        evaluation=EvaluationConfig(metrics=["recall", "precision"], k_values=[1]),
    )

    query = EvaluationQuery(
        query_id="q-qc",
        query_text="What does quantum computing utilize?",
        ground_truth_chunks=["chunk-qc-1"],
    )

    executor = SingleRunExecutor()
    result = await executor.execute(
        experiment_id=exp.id,
        pipeline_config=pipeline_cfg,
        queries=[query],
        corpus_chunks=chunks,
        session=session,
    )

    assert result.status == "COMPLETED"
    assert result.total_queries == 1
    assert result.completed_queries == 1
    assert result.failed_queries == 0
    assert result.mean_metrics["recall@1"] == 1.0

    trails = await trace_repo.get_run_evidence_trails(result.experiment_run_id)
    assert len(trails) == 1
    assert trails[0].reranked_chunks == []
    assert trails[0].retrieved_chunks[0]["retriever_type"] == "bm25"


@pytest.mark.anyio
async def test_single_run_executor_partial_failure(session: AsyncSession) -> None:
    """Verify executor handles a partial failure when one query errors out."""
    dataset_repo = DatasetRepository(session)
    exp_repo = ExperimentRepository(session)

    ds = await dataset_repo.create_dataset(name="Partial DS")
    ver = await dataset_repo.create_version(
        dataset_id=ds.id, version_number=1, content_hash="hash-partial-v1"
    )
    docs = await dataset_repo.add_documents(
        ver.id,
        [{"filename": "doc.txt", "content": "Content.", "content_hash": "h-p"}],
    )
    chunks = [
        DocumentChunk.create(
            doc_id=docs[0].id,
            chunk_index=0,
            content="Content for valid query.",
            token_count=5,
            start_char=0,
            end_char=24,
            strategy="fixed",
            chunk_id="chunk-valid-1",
        )
    ]
    await dataset_repo.add_chunks(
        docs[0].id,
        [
            {
                "id": "chunk-valid-1",
                "chunk_index": 0,
                "content": chunks[0].content,
                "content_hash": "h-valid-c",
                "token_count": 5,
                "strategy": "fixed",
                "chunking_config_hash": "cfg-p",
            }
        ],
    )
    exp = await exp_repo.create_experiment("Partial Exp", ver.id, {}, "hash-partial-cfg")

    pipeline_cfg = PipelineConfig(
        dataset=DatasetConfig(dataset_id=ds.id, dataset_version_id=ver.id),
        retrieval=RetrievalConfig(mode="bm25"),
    )

    class CrashingMockLLM(MockLLMClient):
        def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: object) -> str:
            if "CRASH" in prompt:
                raise RuntimeError("Simulated model generation failure")
            return "Normal answer"

    queries = [
        EvaluationQuery(
            query_id="q-ok",
            query_text="Valid question?",
            ground_truth_chunks=["chunk-valid-1"],
        ),
        EvaluationQuery(
            query_id="q-bad",
            query_text="CRASH please",
            ground_truth_chunks=["chunk-valid-1"],
        ),
    ]

    executor = SingleRunExecutor()
    result = await executor.execute(
        experiment_id=exp.id,
        pipeline_config=pipeline_cfg,
        queries=queries,
        corpus_chunks=chunks,
        session=session,
        llm_client=CrashingMockLLM(),
    )

    assert result.status == "PARTIAL"
    assert result.total_queries == 2
    assert result.completed_queries == 1
    assert result.failed_queries == 1


@pytest.mark.anyio
async def test_single_run_executor_empty_queries(session: AsyncSession) -> None:
    """Verify executor gracefully handles empty query suite."""
    dataset_repo = DatasetRepository(session)
    exp_repo = ExperimentRepository(session)

    ds = await dataset_repo.create_dataset(name="Empty DS")
    ver = await dataset_repo.create_version(
        dataset_id=ds.id, version_number=1, content_hash="hash-empty"
    )
    exp = await exp_repo.create_experiment(
        name="Empty Exp",
        dataset_version_id=ver.id,
        configuration={"test": True},
        configuration_hash="hash-empty-exp",
    )

    pipeline_cfg = PipelineConfig(
        dataset=DatasetConfig(dataset_id=ds.id, dataset_version_id=ver.id),
        retrieval=RetrievalConfig(mode="bm25"),
    )

    executor = SingleRunExecutor()
    result = await executor.execute(
        experiment_id=exp.id,
        pipeline_config=pipeline_cfg,
        queries=[],
        corpus_chunks=[],
        session=session,
    )

    assert result.status == "COMPLETED"
    assert result.total_queries == 0
    assert result.completed_queries == 0
    assert result.failed_queries == 0
    assert result.mean_metrics == {}


@pytest.mark.anyio
async def test_single_run_executor_invalid_chunk_reference_resilience(
    session: AsyncSession,
) -> None:
    """Verify unindexed/invalid chunk references are captured as FAILED queries under validation."""
    dataset_repo = DatasetRepository(session)
    exp_repo = ExperimentRepository(session)
    trace_repo = QueryTraceRepository(session)

    ds = await dataset_repo.create_dataset(name="Strict Integrity DS")
    ver = await dataset_repo.create_version(
        dataset_id=ds.id, version_number=1, content_hash="hash-integrity"
    )
    exp = await exp_repo.create_experiment(
        name="Integrity Exp",
        dataset_version_id=ver.id,
        configuration={"test": True},
        configuration_hash="hash-integrity-exp",
    )

    # Provide corpus chunks whose chunk IDs are NOT added to the database DocumentChunk table
    unregistered_chunks = [
        DocumentChunk.create(
            doc_id="unregistered-doc",
            chunk_index=0,
            content="Ghost content that does not exist in relational persistence.",
            token_count=10,
            start_char=0,
            end_char=60,
            strategy="fixed",
            chunk_id="chunk-ghost-999",
        )
    ]

    pipeline_cfg = PipelineConfig(
        dataset=DatasetConfig(dataset_id=ds.id, dataset_version_id=ver.id),
        retrieval=RetrievalConfig(mode="bm25"),
    )

    queries = [
        EvaluationQuery(
            query_id="q-ghost",
            query_text="Ghost query finding unregistered chunk?",
            ground_truth_chunks=["chunk-ghost-999"],
        )
    ]

    executor = SingleRunExecutor()
    result = await executor.execute(
        experiment_id=exp.id,
        pipeline_config=pipeline_cfg,
        queries=queries,
        corpus_chunks=unregistered_chunks,
        session=session,
        validate_chunk_references=True,
    )

    # With validation enabled and ghost chunk retrieved, the query should fail
    assert result.status == "FAILED"
    assert result.failed_queries == 1
    assert result.completed_queries == 0

    trails = await trace_repo.get_run_evidence_trails(result.experiment_run_id)
    assert len(trails) == 1
    assert trails[0].status == "FAILED"
