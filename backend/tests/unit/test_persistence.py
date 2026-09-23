"""Unit and integration tests for Phase F3: Relational Persistence, Repositories, and Lineage."""

import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.db import drop_db, get_async_engine, get_session_factory, init_db
from app.db.repositories import (
    DatasetRepository,
    ExperimentRepository,
    QueryTraceRepository,
)
from app.models.entities import Document, QueryRun


@pytest.fixture
async def test_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create isolated SQLite database engine with WAL & foreign keys."""
    db_file = Path("test_persistence.db")
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


@pytest.mark.anyio
async def test_sqlite_wal_and_foreign_keys_enforcement(session: AsyncSession) -> None:
    """Verify that SQLite engine initializes with WAL journal mode and enforces foreign keys."""
    # Check foreign keys pragma
    fk_result = await session.execute(text("PRAGMA foreign_keys;"))
    assert fk_result.scalar() == 1

    # Check journal mode (file-based databases report 'wal')
    wal_result = await session.execute(text("PRAGMA journal_mode;"))
    assert str(wal_result.scalar()).lower() == "wal"

    # Enforce foreign key rejection on orphan row insertion
    orphan_doc = Document(
        dataset_version_id="non-existent-version-id",
        filename="orphan.txt",
        content="some orphan text",
        content_hash="abc123hash",
        mime_type="text/plain",
    )
    session.add(orphan_doc)
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


@pytest.mark.anyio
async def test_dataset_version_document_chunk_hierarchy(session: AsyncSession) -> None:
    """Verify creation, relationships, and cascade operations on Dataset hierarchy."""
    repo = DatasetRepository(session)

    # 1. Create Dataset
    dataset = await repo.create_dataset(
        name="TechCorp Docs",
        description="Internal technical knowledge base",
        metadata={"category": "engineering"},
    )
    assert dataset.id is not None
    assert dataset.name == "TechCorp Docs"

    # 2. Create DatasetVersion
    version = await repo.create_version(
        dataset_id=dataset.id,
        version_number=1,
        content_hash="version1_merkle_sha256",
        metadata={"release": "v1.0.0"},
    )
    assert version.id is not None
    assert dataset.current_version_id == version.id

    # 3. Add Documents
    docs = await repo.add_documents(
        version_id=version.id,
        documents_data=[
            {
                "filename": "arch.md",
                "content": "Architecture overview of RAGBench system.",
                "content_hash": "doc1_hash",
                "mime_type": "text/markdown",
            },
            {
                "filename": "deploy.md",
                "content": "Deployment instructions for Docker.",
                "content_hash": "doc2_hash",
                "mime_type": "text/markdown",
            },
        ],
    )
    assert len(docs) == 2
    assert version.document_count == 2
    assert version.total_bytes > 0

    # 4. Add Chunks
    doc1 = docs[0]
    chunks = await repo.add_chunks(
        document_id=doc1.id,
        chunks_data=[
            {
                "id": "chunk_doc1_0",
                "chunk_index": 0,
                "content": "Architecture overview",
                "content_hash": "ch1_hash",
                "token_count": 2,
                "strategy": "fixed",
                "chunking_config_hash": "cfg_hash_fixed",
            },
            {
                "id": "chunk_doc1_1",
                "chunk_index": 1,
                "content": "of RAGBench system.",
                "content_hash": "ch2_hash",
                "token_count": 3,
                "strategy": "fixed",
                "chunking_config_hash": "cfg_hash_fixed",
            },
        ],
    )
    assert len(chunks) == 2

    # Query back chunks
    queried_chunks = await repo.get_document_chunks(doc1.id)
    assert len(queried_chunks) == 2
    assert [c.chunk_index for c in queried_chunks] == [0, 1]


@pytest.mark.anyio
async def test_experiment_and_run_lifecycle(session: AsyncSession) -> None:
    """Verify Experiment, Run, caching lookup, and aggregate summary metrics."""
    ds_repo = DatasetRepository(session)
    exp_repo = ExperimentRepository(session)

    # Setup parent dataset
    dataset = await ds_repo.create_dataset(name="Benchmark Corpus")
    version = await ds_repo.create_version(
        dataset_id=dataset.id,
        version_number=1,
        content_hash="corpus_v1_hash",
    )

    # 1. Create Experiment
    config_dict = {"retrieval": {"mode": "hybrid", "top_k": 10}}
    experiment = await exp_repo.create_experiment(
        name="Hybrid Retrieval Benchmark",
        dataset_version_id=version.id,
        configuration=config_dict,
        configuration_hash="exp_cfg_sha256_hash",
        created_by="Research Engineer",
    )
    assert experiment.id is not None

    # 2. Create ExperimentRun
    run = await exp_repo.create_run(
        experiment_id=experiment.id,
        pipeline_config_hash="pipe_cfg_hash_1",
        cache_key="Benchmark Corpus@corpus_v1_hash:pipe_cfg_hash_1",
        cache_hash="cache_identity_sha256_hash",
        environment="test-runner-01",
        random_seed=42,
    )
    assert run.id is not None
    assert run.status == "RUNNING"

    # 3. Complete Run with summary metrics
    completed_run = await exp_repo.complete_run(
        run_id=run.id,
        summary_metrics={"mrr@10": 0.85, "ndcg@10": 0.89},
        status="COMPLETED",
    )
    assert completed_run.status == "COMPLETED"
    assert completed_run.completed_at is not None

    # 4. Lookup by Cache Hash
    cached_run = await exp_repo.get_run_by_cache_hash("cache_identity_sha256_hash")
    assert cached_run is not None
    assert cached_run.id == run.id

    # 5. Add and query RunMetricSummary statistical rollups
    summaries = await exp_repo.add_metric_summaries(
        experiment_run_id=run.id,
        summaries=[
            {
                "metric_name": "recall@5",
                "mean": 0.78,
                "median": 0.80,
                "min": 0.50,
                "max": 1.00,
                "stddev": 0.12,
                "count": 50,
            },
            {
                "metric_name": "ndcg@5",
                "mean": 0.84,
                "median": 0.86,
                "min": 0.60,
                "max": 1.00,
                "stddev": 0.09,
                "count": 50,
            },
        ],
    )
    assert len(summaries) == 2

    queried_summaries = await exp_repo.get_metric_summaries(run.id)
    assert len(queried_summaries) == 2
    assert [s.metric_name for s in queried_summaries] == ["ndcg@5", "recall@5"]


@pytest.mark.anyio
async def test_reconstruct_complete_query_evidence_trail(session: AsyncSession) -> None:
    """Crucial Integration Test: Full reconstruction of query evidence trail
    without aggregate metrics.

    Verifies the lineage path:
    ExperimentRun -> QueryRun -> TransformedQuery -> RetrievedChunk -> RerankedChunk
                  -> PackedContext -> GenerationResult -> MetricResult
    """
    ds_repo = DatasetRepository(session)
    exp_repo = ExperimentRepository(session)
    trace_repo = QueryTraceRepository(session)

    # 1. Setup Dataset, Version, Documents, and Chunks
    dataset = await ds_repo.create_dataset(name="Evidence Test Corpus")
    version = await ds_repo.create_version(
        dataset_id=dataset.id,
        version_number=1,
        content_hash="evidence_v1_hash",
    )
    docs = await ds_repo.add_documents(
        version_id=version.id,
        documents_data=[
            {
                "filename": "benefits.md",
                "content": "Full passage on hybrid vs dense retrieval benefits.",
                "content_hash": "hash_benefits_doc",
            }
        ],
    )
    # Add chunks so chunk reference integrity succeeds
    await ds_repo.add_chunks(
        document_id=docs[0].id,
        chunks_data=[
            {
                "id": "chunk_benefits_01",
                "chunk_index": 0,
                "content": "Dense captures semantic context.",
                "content_hash": "h_c1",
                "token_count": 5,
                "strategy": "fixed",
                "chunking_config_hash": "cfg_h",
            },
            {
                "id": "chunk_benefits_02",
                "chunk_index": 1,
                "content": "Lexical match handles exact terms.",
                "content_hash": "h_c2",
                "token_count": 5,
                "strategy": "fixed",
                "chunking_config_hash": "cfg_h",
            },
            {
                "id": "chunk_other_99",
                "chunk_index": 2,
                "content": "Other passage.",
                "content_hash": "h_c3",
                "token_count": 2,
                "strategy": "fixed",
                "chunking_config_hash": "cfg_h",
            },
            {
                "id": "chunk_unrelated_55",
                "chunk_index": 3,
                "content": "Unrelated passage.",
                "content_hash": "h_c4",
                "token_count": 2,
                "strategy": "fixed",
                "chunking_config_hash": "cfg_h",
            },
        ],
    )

    experiment = await exp_repo.create_experiment(
        name="Lineage Evidence Verification",
        dataset_version_id=version.id,
        configuration={"test": True},
        configuration_hash="cfg_hash_evidence",
    )
    run = await exp_repo.create_run(
        experiment_id=experiment.id,
        pipeline_config_hash="pipe_evidence_hash",
        cache_key="key_evidence",
        cache_hash="hash_evidence",
    )

    # 2. Record full execution trace for a single benchmark query
    recorded_query = await trace_repo.record_query_trace(
        experiment_run_id=run.id,
        query_id="q_1001",
        original_query="What are the key benefits of hybrid retrieval over dense retrieval?",
        expected_answer=(
            "Hybrid retrieval combines semantic and lexical matching, "
            "improving out-of-domain robustness."
        ),
        ground_truth_chunks=["chunk_benefits_01", "chunk_benefits_02"],
        latency_ms=142.5,
        status="SUCCESS",
        metadata={"domain": "information_retrieval", "difficulty": "hard"},
        transformed_queries=[
            {
                "transformation_type": "hyde",
                "query_text": (
                    "Hypothetical passage discussing how lexical search handles rare terms "
                    "while dense search captures synonyms."
                ),
                "sequence_index": 0,
                "metadata": {"generator": "mock-hyde"},
            },
            {
                "transformation_type": "step_back",
                "query_text": "How do sparse and dense retrieval systems differ in IR benchmarks?",
                "sequence_index": 1,
                "metadata": {"generator": "mock-stepback"},
            },
        ],
        retrieved_chunks=[
            {
                "retriever_type": "dense",
                "chunk_id": "chunk_benefits_01",
                "rank": 1,
                "score": 0.88,
                "metadata": {"source": "vector_index"},
            },
            {
                "retriever_type": "dense",
                "chunk_id": "chunk_other_99",
                "rank": 2,
                "score": 0.74,
                "metadata": {"source": "vector_index"},
            },
            {
                "retriever_type": "bm25",
                "chunk_id": "chunk_benefits_02",
                "rank": 1,
                "score": 14.2,
                "metadata": {"source": "bm25_index"},
            },
            {
                "retriever_type": "bm25",
                "chunk_id": "chunk_unrelated_55",
                "rank": 2,
                "score": 8.1,
                "metadata": {"source": "bm25_index"},
            },
        ],
        reranked_chunks=[
            {
                "chunk_id": "chunk_benefits_02",
                "original_rank": 3,
                "reranked_rank": 1,
                "original_score": 14.2,
                "reranker_score": 0.96,
                "metadata": {"cross_encoder": "flashrank"},
            },
            {
                "chunk_id": "chunk_benefits_01",
                "original_rank": 1,
                "reranked_rank": 2,
                "original_score": 0.88,
                "reranker_score": 0.91,
                "metadata": {"cross_encoder": "flashrank"},
            },
        ],
        packed_context={
            "context_text": (
                "[1] (chunk_benefits_02) Lexical match handles exact terms.\n"
                "[2] (chunk_benefits_01) Dense captures semantic context."
            ),
            "token_count": 35,
            "token_budget": 2048,
            "ordering_strategy": "lost_in_the_middle",
            "chunk_ids": ["chunk_benefits_02", "chunk_benefits_01"],
            "metadata": {"reordered": True},
        },
        generation_result={
            "model": "gpt-4o-mini",
            "answer": (
                "Hybrid retrieval leverages both lexical precision for rare terms "
                "and semantic understanding for paraphrases."
            ),
            "prompt_version": "ragbench_qa_v1",
            "input_tokens": 120,
            "output_tokens": 25,
            "total_tokens": 145,
            "latency_ms": 95.0,
            "estimated_cost": 0.00015,
            "metadata": {"finish_reason": "stop"},
        },
        metric_results=[
            {
                "metric_name": "precision@5",
                "metric_value": 0.8,
                "evaluator": "ir_evaluator",
            },
            {
                "metric_name": "recall@5",
                "metric_value": 1.0,
                "evaluator": "ir_evaluator",
            },
            {
                "metric_name": "faithfulness",
                "metric_value": 0.95,
                "evaluator": "llm_evaluator",
            },
        ],
    )
    assert recorded_query.id is not None

    # 3. Authoritative Evidence Trail Reconstruction
    trail = await trace_repo.get_query_evidence_trail(recorded_query.id)
    assert trail is not None

    # Verify Base Query Information
    assert trail.query_id == "q_1001"
    assert (
        trail.original_query
        == "What are the key benefits of hybrid retrieval over dense retrieval?"
    )
    assert trail.ground_truth_chunks == ["chunk_benefits_01", "chunk_benefits_02"]
    assert trail.status == "SUCCESS"
    assert trail.latency_ms == 142.5

    # Verify Transformed Queries Evidence
    assert len(trail.transformed_queries) == 2
    assert trail.transformed_queries[0]["transformation_type"] == "hyde"
    assert "rare terms while dense search" in trail.transformed_queries[0]["query_text"]
    assert trail.transformed_queries[1]["transformation_type"] == "step_back"

    # Verify Retrieved Chunks Evidence
    assert len(trail.retrieved_chunks) == 4
    retrieved_chunk_ids = [rc["chunk_id"] for rc in trail.retrieved_chunks]
    assert "chunk_benefits_01" in retrieved_chunk_ids
    assert "chunk_benefits_02" in retrieved_chunk_ids

    # Verify Reranked Chunks Evidence & Rank Shifts
    assert len(trail.reranked_chunks) == 2
    assert trail.reranked_chunks[0]["chunk_id"] == "chunk_benefits_02"
    assert trail.reranked_chunks[0]["reranked_rank"] == 1
    assert trail.reranked_chunks[0]["reranker_score"] == 0.96
    assert trail.reranked_chunks[1]["chunk_id"] == "chunk_benefits_01"
    assert trail.reranked_chunks[1]["reranked_rank"] == 2

    # Verify Packed Context Evidence
    assert trail.packed_context is not None
    assert trail.packed_context["token_count"] == 35
    assert trail.packed_context["token_budget"] == 2048
    assert trail.packed_context["ordering_strategy"] == "lost_in_the_middle"
    assert trail.packed_context["chunk_ids"] == ["chunk_benefits_02", "chunk_benefits_01"]

    # Verify Generation Evidence
    assert trail.generation_result is not None
    assert trail.generation_result["model"] == "gpt-4o-mini"
    assert "Hybrid retrieval leverages both lexical precision" in trail.generation_result["answer"]
    assert trail.generation_result["total_tokens"] == 145

    # Verify Individual Metric Results (not just aggregate summaries)
    assert trail.metric_results["precision@5"] == 0.8
    assert trail.metric_results["recall@5"] == 1.0
    assert trail.metric_results["faithfulness"] == 0.95

    # Verify Run-Level Query Evidence Collection
    run_trails = await trace_repo.get_run_evidence_trails(run.id)
    assert len(run_trails) == 1
    assert run_trails[0].query_run_id == recorded_query.id


@pytest.mark.anyio
async def test_historical_dataset_protection_and_immutability(session: AsyncSession) -> None:
    """Test A: Verify that a DatasetVersion referenced by experiments cannot be deleted.

    Guarantees that historical scientific evidence cannot disappear merely because an
    upstream organizational entity or snapshot is targeted for deletion.
    """
    ds_repo = DatasetRepository(session)
    exp_repo = ExperimentRepository(session)

    # 1. Create dataset, version, and experiment referencing the version
    dataset = await ds_repo.create_dataset(name="Historical Corpus")
    version = await ds_repo.create_version(
        dataset_id=dataset.id,
        version_number=1,
        content_hash="hist_v1_hash",
    )
    experiment = await exp_repo.create_experiment(
        name="Study on Historical Corpus",
        dataset_version_id=version.id,
        configuration={"model": "test"},
        configuration_hash="hist_exp_hash",
    )
    run = await exp_repo.create_run(
        experiment_id=experiment.id,
        pipeline_config_hash="p_hash",
        cache_key="ckey",
        cache_hash="chash",
    )
    exp_id = experiment.id
    run_id = run.id
    version_id = version.id

    # Commit baseline state so rollback only affects the failed delete attempt
    await session.commit()

    # 2. Attempt to delete the DatasetVersion -> MUST be rejected by RESTRICT constraint
    await session.delete(version)
    with pytest.raises(IntegrityError):
        await session.flush()

    # 3. Rollback the rejected deletion attempt
    await session.rollback()

    # 4. Verify that the experiment, run, and dataset version remain fully intact
    persisted_exp = await exp_repo.get_experiment(exp_id)
    assert persisted_exp is not None
    assert persisted_exp.dataset_version_id == version_id

    persisted_run = await exp_repo.get_run(run_id)
    assert persisted_run is not None


@pytest.mark.anyio
async def test_invalid_chunk_reference_integrity_and_atomic_rollback(
    session: AsyncSession,
) -> None:
    """Test B: Verify repository-level integrity validation of chunk references.

    Guarantees:
    - Valid chunk IDs belonging to the DatasetVersion succeed.
    - Unknown retrieved chunk IDs are rejected.
    - Unknown reranked chunk IDs are rejected.
    - Rejection leaves NO partial QueryRun or orphan child records.
    """
    ds_repo = DatasetRepository(session)
    exp_repo = ExperimentRepository(session)
    trace_repo = QueryTraceRepository(session)

    # 1. Setup Dataset, Version, Document, and valid Chunk C1
    dataset = await ds_repo.create_dataset(name="Chunk Integrity Corpus")
    version = await ds_repo.create_version(
        dataset_id=dataset.id,
        version_number=1,
        content_hash="chk_v1_hash",
    )
    docs = await ds_repo.add_documents(
        version_id=version.id,
        documents_data=[
            {
                "filename": "doc1.txt",
                "content": "Valid chunk content for C1.",
                "content_hash": "doc1_chk_hash",
            }
        ],
    )
    await ds_repo.add_chunks(
        document_id=docs[0].id,
        chunks_data=[
            {
                "id": "C1",
                "chunk_index": 0,
                "content": "Valid chunk content for C1.",
                "content_hash": "c1_hash",
                "token_count": 6,
                "strategy": "fixed",
                "chunking_config_hash": "cfg_h",
            }
        ],
    )

    experiment = await exp_repo.create_experiment(
        name="Chunk Ref Test Experiment",
        dataset_version_id=version.id,
        configuration={"test": True},
        configuration_hash="cfg_h_chk",
    )
    run = await exp_repo.create_run(
        experiment_id=experiment.id,
        pipeline_config_hash="pipe_h_chk",
        cache_key="ckey_chk",
        cache_hash="chash_chk",
    )

    # 2. Case A: Known chunk C1 -> MUST succeed
    valid_query = await trace_repo.record_query_trace(
        experiment_run_id=run.id,
        query_id="q_valid_c1",
        original_query="Valid query referencing C1",
        retrieved_chunks=[
            {
                "retriever_type": "dense",
                "chunk_id": "C1",
                "rank": 1,
                "score": 0.95,
            }
        ],
        reranked_chunks=[
            {
                "chunk_id": "C1",
                "original_rank": 1,
                "reranked_rank": 1,
                "original_score": 0.95,
                "reranker_score": 0.99,
            }
        ],
    )
    assert valid_query.id is not None

    # 3. Case B: Unknown retrieved chunk ID C999 -> MUST be rejected
    with pytest.raises(
        ValueError, match="unknown chunk ID 'C999' does not exist in dataset version"
    ):
        await trace_repo.record_query_trace(
            experiment_run_id=run.id,
            query_id="q_invalid_retrieved",
            original_query="Query referencing non-existent retrieved chunk",
            retrieved_chunks=[
                {
                    "retriever_type": "dense",
                    "chunk_id": "C999",
                    "rank": 1,
                    "score": 0.50,
                }
            ],
        )

    # Assert that no partial QueryRun was created for q_invalid_retrieved
    check_stmt_1 = select(QueryRun).where(QueryRun.query_id == "q_invalid_retrieved")
    res_1 = await session.execute(check_stmt_1)
    assert res_1.scalar_one_or_none() is None

    # 4. Case C: Unknown reranked chunk ID C999 -> MUST be rejected
    with pytest.raises(
        ValueError, match="unknown chunk ID 'C999' does not exist in dataset version"
    ):
        await trace_repo.record_query_trace(
            experiment_run_id=run.id,
            query_id="q_invalid_reranked",
            original_query="Query referencing non-existent reranked chunk",
            retrieved_chunks=[
                {
                    "retriever_type": "dense",
                    "chunk_id": "C1",
                    "rank": 1,
                    "score": 0.90,
                }
            ],
            reranked_chunks=[
                {
                    "chunk_id": "C999",
                    "original_rank": 1,
                    "reranked_rank": 1,
                    "original_score": 0.90,
                    "reranker_score": 0.92,
                }
            ],
        )

    # Assert that no partial QueryRun was created for q_invalid_reranked
    check_stmt_2 = select(QueryRun).where(QueryRun.query_id == "q_invalid_reranked")
    res_2 = await session.execute(check_stmt_2)
    assert res_2.scalar_one_or_none() is None
