"""Unit and integration tests for Phase F4: Single-Run Pipeline Executor and Audit Invariants."""

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
from app.engine.query_transforms.hyde import HyDETransformer
from app.engine.query_transforms.multi_query import MultiQueryExpander
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
            ground_truth_docs=["doc-1"],
            metadata={"domain": "botany", "difficulty": "easy"},
        ),
        EvaluationQuery(
            query_id="q-2",
            query_text="Where is ATP produced in cellular respiration?",
            expected_answer="ATP is produced in the mitochondria.",
            ground_truth_chunks=["chunk-resp-003"],
            ground_truth_docs=["doc-2"],
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
    assert "doc_recall@1" in result.mean_metrics

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
        assert s.metadata_json.get("stddev_type") == "sample"

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
async def test_audit_chunk_id_vs_doc_id_separation(session: AsyncSession) -> None:
    """Audit Invariant 1: Prove chunk_id and doc_id are distinct and mapped explicitly."""
    dataset_repo = DatasetRepository(session)
    exp_repo = ExperimentRepository(session)
    trace_repo = QueryTraceRepository(session)

    ds = await dataset_repo.create_dataset(name="Mapping Audit DS")
    ver = await dataset_repo.create_version(
        dataset_id=ds.id, version_number=1, content_hash="hash-map-v1"
    )
    docs = await dataset_repo.add_documents(
        ver.id,
        [
            {
                "external_id": "doc-alpha",
                "filename": "alpha.txt",
                "content": "Alpha.",
                "content_hash": "h-a",
            },
            {
                "external_id": "doc-beta",
                "filename": "beta.txt",
                "content": "Beta.",
                "content_hash": "h-b",
            },
        ],
    )

    # Note explicit distinct IDs: chunk_id != doc_id
    chunks = [
        DocumentChunk.create(
            doc_id="doc-alpha",
            chunk_index=0,
            content="Alpha content passage.",
            token_count=3,
            start_char=0,
            end_char=21,
            strategy="fixed",
            chunk_id="chunk-alpha-001",
        ),
        DocumentChunk.create(
            doc_id="doc-beta",
            chunk_index=0,
            content="Beta content passage.",
            token_count=3,
            start_char=0,
            end_char=20,
            strategy="fixed",
            chunk_id="chunk-beta-002",
        ),
    ]

    # Explicit assertion: chunk_id must NEVER equal doc_id
    for c in chunks:
        assert c.chunk_id != c.doc_id

    await dataset_repo.add_chunks(
        docs[0].id,
        [
            {
                "id": "chunk-alpha-001",
                "chunk_index": 0,
                "content": chunks[0].content,
                "content_hash": "h-ca1",
                "token_count": 3,
                "strategy": "fixed",
                "chunking_config_hash": "cfg-1",
            }
        ],
    )
    await dataset_repo.add_chunks(
        docs[1].id,
        [
            {
                "id": "chunk-beta-002",
                "chunk_index": 0,
                "content": chunks[1].content,
                "content_hash": "h-cb2",
                "token_count": 3,
                "strategy": "fixed",
                "chunking_config_hash": "cfg-1",
            }
        ],
    )
    exp = await exp_repo.create_experiment("Mapping Exp", ver.id, {}, "hash-map-cfg")

    pipeline_cfg = PipelineConfig(
        dataset=DatasetConfig(dataset_id=ds.id, dataset_version_id=ver.id),
        retrieval=RetrievalConfig(mode="bm25", top_k=2),
        evaluation=EvaluationConfig(metrics=["recall", "precision"], k_values=[1]),
    )

    query = EvaluationQuery(
        query_id="q-map",
        query_text="Alpha content",
        ground_truth_chunks=["chunk-alpha-001"],
        ground_truth_docs=["doc-alpha"],
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
    assert result.mean_metrics["recall@1"] == 1.0
    assert result.mean_metrics["doc_recall@1"] == 1.0

    trail = await trace_repo.get_query_evidence_trail(
        (await trace_repo.get_run_evidence_trails(result.experiment_run_id))[0].query_run_id
    )
    assert trail is not None
    assert trail.retrieved_chunks[0]["chunk_id"] == "chunk-alpha-001"
    assert trail.retrieved_chunks[0]["chunk_id"] != "doc-alpha"


@pytest.mark.anyio
async def test_audit_per_query_transaction_atomicity(session: AsyncSession) -> None:
    """Audit Invariant 2: Prove each query trace is committed atomically."""
    dataset_repo = DatasetRepository(session)
    exp_repo = ExperimentRepository(session)
    trace_repo = QueryTraceRepository(session)

    ds = await dataset_repo.create_dataset(name="Atomicity DS")
    ver = await dataset_repo.create_version(
        dataset_id=ds.id, version_number=1, content_hash="hash-atom-v1"
    )
    docs = await dataset_repo.add_documents(
        ver.id,
        [{"filename": "doc.txt", "content": "Atomic content.", "content_hash": "h-atom"}],
    )
    chunks = [
        DocumentChunk.create(
            doc_id=docs[0].id,
            chunk_index=0,
            content="Atomic chunk content.",
            token_count=3,
            start_char=0,
            end_char=20,
            strategy="fixed",
            chunk_id="chunk-atom-1",
        )
    ]
    await dataset_repo.add_chunks(
        docs[0].id,
        [
            {
                "id": "chunk-atom-1",
                "chunk_index": 0,
                "content": chunks[0].content,
                "content_hash": "h-ca-atom",
                "token_count": 3,
                "strategy": "fixed",
                "chunking_config_hash": "cfg-atom",
            }
        ],
    )
    exp = await exp_repo.create_experiment("Atomicity Exp", ver.id, {}, "hash-atom-cfg")

    pipeline_cfg = PipelineConfig(
        dataset=DatasetConfig(dataset_id=ds.id, dataset_version_id=ver.id),
        retrieval=RetrievalConfig(mode="bm25"),
    )

    class FailingSecondQueryLLM(MockLLMClient):
        def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: object) -> str:
            if "FAIL_ME" in prompt:
                raise RuntimeError("Query 2 generation exploded!")
            return "Valid answer"

    queries = [
        EvaluationQuery(
            query_id="q-1-ok", query_text="Question 1", ground_truth_chunks=["chunk-atom-1"]
        ),
        EvaluationQuery(
            query_id="q-2-fail", query_text="FAIL_ME please", ground_truth_chunks=["chunk-atom-1"]
        ),
        EvaluationQuery(
            query_id="q-3-ok", query_text="Question 3", ground_truth_chunks=["chunk-atom-1"]
        ),
    ]

    executor = SingleRunExecutor()
    result = await executor.execute(
        experiment_id=exp.id,
        pipeline_config=pipeline_cfg,
        queries=queries,
        corpus_chunks=chunks,
        session=session,
        llm_client=FailingSecondQueryLLM(),
    )

    # Overall run is PARTIAL: 2 of 3 succeeded
    assert result.status == "PARTIAL"
    assert result.total_queries == 3
    assert result.completed_queries == 2
    assert result.failed_queries == 1

    trails = await trace_repo.get_run_evidence_trails(result.experiment_run_id)
    assert len(trails) == 3

    # Query 1: completely committed and successful
    t1 = next(t for t in trails if t.query_id == "q-1-ok")
    assert t1.status == "SUCCESS"
    assert t1.generation_result is not None
    assert len(t1.metric_results) > 0

    # Query 2: captured as FAILED with zero corrupt partial entities
    t2 = next(t for t in trails if t.query_id == "q-2-fail")
    assert t2.status == "FAILED"
    assert t2.generation_result is None
    assert t2.retrieved_chunks == []
    assert t2.metric_results == {}

    # Query 3: completely committed and successful
    t3 = next(t for t in trails if t.query_id == "q-3-ok")
    assert t3.status == "SUCCESS"
    assert t3.generation_result is not None

    # Summary metrics: count MUST be 2 (number of valid observations), not 3!
    summaries = await exp_repo.get_metric_summaries(result.experiment_run_id)
    for s in summaries:
        assert s.count == 2


@pytest.mark.anyio
async def test_audit_metric_aggregation_no_zero_injection(session: AsyncSession) -> None:
    """Audit Invariant 3: Failed queries must not inject zeros into aggregate metrics."""
    dataset_repo = DatasetRepository(session)
    exp_repo = ExperimentRepository(session)

    ds = await dataset_repo.create_dataset(name="No Zero DS")
    ver = await dataset_repo.create_version(
        dataset_id=ds.id, version_number=1, content_hash="hash-nozero-v1"
    )
    docs = await dataset_repo.add_documents(
        ver.id,
        [{"filename": "doc.txt", "content": "Content.", "content_hash": "h-nz"}],
    )
    chunks = [
        DocumentChunk.create(
            doc_id=docs[0].id,
            chunk_index=0,
            content="Sample passage text.",
            token_count=3,
            start_char=0,
            end_char=19,
            strategy="fixed",
            chunk_id="chunk-nz-1",
        )
    ]
    await dataset_repo.add_chunks(
        docs[0].id,
        [
            {
                "id": "chunk-nz-1",
                "chunk_index": 0,
                "content": chunks[0].content,
                "content_hash": "h-ca-nz",
                "token_count": 3,
                "strategy": "fixed",
                "chunking_config_hash": "cfg-nz",
            }
        ],
    )
    exp = await exp_repo.create_experiment("No Zero Exp", ver.id, {}, "hash-nz-cfg")

    pipeline_cfg = PipelineConfig(
        dataset=DatasetConfig(dataset_id=ds.id, dataset_version_id=ver.id),
        retrieval=RetrievalConfig(mode="bm25"),
        evaluation=EvaluationConfig(metrics=["recall"], k_values=[1]),
    )

    class CrashingLLM(MockLLMClient):
        def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: object) -> str:
            if "CRASH" in prompt:
                raise RuntimeError("Boom")
            return "Answer"

    # Query 1 has recall@1 = 1.0, Query 2 crashes
    queries = [
        EvaluationQuery(query_id="q-ok", query_text="Valid", ground_truth_chunks=["chunk-nz-1"]),
        EvaluationQuery(query_id="q-crash", query_text="CRASH", ground_truth_chunks=["chunk-nz-1"]),
    ]

    executor = SingleRunExecutor()
    result = await executor.execute(
        experiment_id=exp.id,
        pipeline_config=pipeline_cfg,
        queries=queries,
        corpus_chunks=chunks,
        session=session,
        llm_client=CrashingLLM(),
    )

    # If zero were injected for q-crash, recall@1 would be (1.0 + 0.0) / 2 = 0.5.
    # The true mean of valid observations is 1.0 / 1 = 1.0.
    assert result.mean_metrics["recall@1"] == 1.0

    summaries = await exp_repo.get_metric_summaries(result.experiment_run_id)
    recall_summary = next(s for s in summaries if s.metric_name == "recall@1")
    assert recall_summary.count == 1
    assert recall_summary.mean == 1.0
    assert recall_summary.min == 1.0
    assert recall_summary.max == 1.0


@pytest.mark.anyio
async def test_audit_empty_queries_produces_no_fake_zeros(session: AsyncSession) -> None:
    """Audit Invariant 8: Empty queries must not produce fake zero-valued summaries."""
    dataset_repo = DatasetRepository(session)
    exp_repo = ExperimentRepository(session)

    ds = await dataset_repo.create_dataset(name="Empty Audit DS")
    ver = await dataset_repo.create_version(
        dataset_id=ds.id, version_number=1, content_hash="hash-empty-audit"
    )
    exp = await exp_repo.create_experiment("Empty Exp", ver.id, {}, "hash-empty-cfg")

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
    assert result.mean_metrics == {}

    summaries = await exp_repo.get_metric_summaries(result.experiment_run_id)
    assert summaries == []


def test_audit_topology_and_fusion_method_fidelity() -> None:
    """Audit Invariant 5: Prove hybrid_fusion (RRF vs RSN) and top_k flow correctly."""
    chunks = create_sample_chunks()
    mock_emb = DeterministicMockEmbeddingProvider(dimension=64)

    # 1. RSN fusion with alpha=0.85 and top_k=7
    cfg_rsn = PipelineConfig(
        dataset=DatasetConfig(dataset_id="ds", dataset_version_id="v1"),
        embedding=EmbeddingConfig(provider="mock", dimension=64),
        retrieval=RetrievalConfig(mode="hybrid", top_k=7, hybrid_fusion="rsn", dense_weight=0.85),
    )
    comp_rsn = build_pipeline_components(cfg_rsn, corpus_chunks=chunks, embedding_provider=mock_emb)
    assert isinstance(comp_rsn.retriever, HybridRetriever)
    assert comp_rsn.retriever.fusion_method == "rsn"
    assert comp_rsn.retriever.alpha == 0.85

    # 2. RRF fusion with rrf_k=45 and top_k=15
    cfg_rrf = PipelineConfig(
        dataset=DatasetConfig(dataset_id="ds", dataset_version_id="v1"),
        embedding=EmbeddingConfig(provider="mock", dimension=64),
        retrieval=RetrievalConfig(mode="hybrid", top_k=15, hybrid_fusion="rrf", rrf_k=45),
    )
    comp_rrf = build_pipeline_components(cfg_rrf, corpus_chunks=chunks, embedding_provider=mock_emb)
    assert isinstance(comp_rrf.retriever, HybridRetriever)
    assert comp_rrf.retriever.fusion_method == "rrf"
    assert comp_rrf.retriever.rrf_k == 45


def test_audit_transformation_lineage_all_strategies() -> None:
    """Audit Invariant 6: Verify all query transformation strategies produce verifiable lineage."""
    mock_llm = MockLLMClient()

    # Identity
    cfg_id = PipelineConfig(
        dataset=DatasetConfig(dataset_id="ds", dataset_version_id="v1"),
        query_transform=QueryTransformConfig(strategy="identity"),
    )
    comp_id = build_pipeline_components(cfg_id, corpus_chunks=[], llm_client=mock_llm)
    t_id = comp_id.transformer.transform("What is RAG?")
    assert t_id.strategy in ("none", "identity")
    assert t_id.transformed_queries == ["What is RAG?"]

    # StepBack
    cfg_sb = PipelineConfig(
        dataset=DatasetConfig(dataset_id="ds", dataset_version_id="v1"),
        query_transform=QueryTransformConfig(strategy="step_back", include_original=True),
    )
    comp_sb = build_pipeline_components(cfg_sb, corpus_chunks=[], llm_client=mock_llm)
    assert isinstance(comp_sb.transformer, StepBackTransformer)
    t_sb = comp_sb.transformer.transform("What is RAG?")
    assert t_sb.strategy == "step_back"
    assert len(t_sb.transformed_queries) == 2
    assert t_sb.transformed_queries[1] == "What is RAG?"

    # MultiQuery
    cfg_mq = PipelineConfig(
        dataset=DatasetConfig(dataset_id="ds", dataset_version_id="v1"),
        query_transform=QueryTransformConfig(strategy="multi_query", num_queries=3),
    )
    comp_mq = build_pipeline_components(cfg_mq, corpus_chunks=[], llm_client=mock_llm)
    assert isinstance(comp_mq.transformer, MultiQueryExpander)
    t_mq = comp_mq.transformer.transform("What is RAG?")
    assert t_mq.strategy == "multi_query"
    assert len(t_mq.transformed_queries) >= 2

    # HyDE
    cfg_hyde = PipelineConfig(
        dataset=DatasetConfig(dataset_id="ds", dataset_version_id="v1"),
        query_transform=QueryTransformConfig(strategy="hyde"),
    )
    comp_hyde = build_pipeline_components(cfg_hyde, corpus_chunks=[], llm_client=mock_llm)
    assert isinstance(comp_hyde.transformer, HyDETransformer)
    t_hyde = comp_hyde.transformer.transform("What is RAG?")
    assert t_hyde.strategy == "hyde"
    assert len(t_hyde.transformed_queries) == 1


@pytest.mark.anyio
async def test_audit_reranking_lineage_rank_preservation(session: AsyncSession) -> None:
    """Audit Invariant 7: Reranking lineage preserves original retrieval rank vs reranked rank."""
    dataset_repo = DatasetRepository(session)
    exp_repo = ExperimentRepository(session)
    trace_repo = QueryTraceRepository(session)

    ds = await dataset_repo.create_dataset(name="Rerank Lineage DS")
    ver = await dataset_repo.create_version(
        dataset_id=ds.id, version_number=1, content_hash="hash-rerank-v1"
    )
    docs = await dataset_repo.add_documents(
        ver.id,
        [{"filename": "doc.txt", "content": "Rerank content.", "content_hash": "h-rrk"}],
    )
    chunks = create_sample_chunks()
    await dataset_repo.add_chunks(
        docs[0].id,
        [
            {
                "id": c.chunk_id,
                "chunk_index": i,
                "content": c.content,
                "content_hash": f"h-{i}",
                "token_count": c.token_count,
                "strategy": "fixed",
                "chunking_config_hash": "cfg",
            }
            for i, c in enumerate(chunks)
        ],
    )
    exp = await exp_repo.create_experiment("Rerank Exp", ver.id, {}, "hash-rrk-cfg")

    pipeline_cfg = PipelineConfig(
        dataset=DatasetConfig(dataset_id=ds.id, dataset_version_id=ver.id),
        retrieval=RetrievalConfig(mode="bm25", top_k=3),
        reranker=RerankerConfig(enabled=True, strategy="mock", top_n=3),
    )

    query = EvaluationQuery(
        query_id="q-rrk",
        query_text="Photosynthesis sunlight",
        ground_truth_chunks=["chunk-photo-001"],
    )

    executor = SingleRunExecutor()
    result = await executor.execute(
        experiment_id=exp.id,
        pipeline_config=pipeline_cfg,
        queries=[query],
        corpus_chunks=chunks,
        session=session,
    )

    trails = await trace_repo.get_run_evidence_trails(result.experiment_run_id)
    assert len(trails) == 1
    reranked = trails[0].reranked_chunks
    assert len(reranked) > 0

    for r in reranked:
        assert r["original_rank"] >= 1  # 1-based rank from BM25 retrieval
        assert r["reranked_rank"] >= 1  # 1-based rank from Mock reranker
        assert isinstance(r["original_score"], float)  # BM25 score
        assert isinstance(r["reranker_score"], float)  # Reranker score


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

    assert result.status == "FAILED"
    assert result.failed_queries == 1
    assert result.completed_queries == 0

    trails = await trace_repo.get_run_evidence_trails(result.experiment_run_id)
    assert len(trails) == 1
    assert trails[0].status == "FAILED"
