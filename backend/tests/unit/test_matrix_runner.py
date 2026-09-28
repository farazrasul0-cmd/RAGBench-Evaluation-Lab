"""Unit and integration tests for Phase F5: Matrix Runner, Execution Caching, and Concurrency.

Verifies:
1. Cartesian sweep execution across multidimensional parameter sweeps.
2. Cache hit reuse returning existing completed runs without fake DB executions.
3. Cache only reuses COMPLETED runs (never RUNNING, PARTIAL, or FAILED).
4. Bypass cache forces full execution while preserving historical runs.
5. Deterministic result ordering matching CartesianExpander sequence.
6. Sequential compatibility with max_concurrency=1.
7. Concurrency bounds strictly enforced (max_observed <= max_concurrency).
8. Partial configuration failure isolation without aborting the matrix.
9. Successful configurations survive concurrent failures.
10. In-flight duplicate configurations do not execute twice.
11. Cache identity strictly matches F2 compute_cache_identity semantics.
12. Cached run ID points to original run and duration_ms=0 on cached result.
13. Single session fallback execution.
"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.db import drop_db, get_async_engine, get_session_factory, init_db
from app.db.repositories.dataset import DatasetRepository
from app.db.repositories.experiment import ExperimentRepository
from app.engine.embeddings.mock_provider import DeterministicMockEmbeddingProvider
from app.engine.orchestrator import CartesianExpander, compute_cache_identity
from app.engine.query_transforms.base import MockLLMClient
from app.engine.runner import (
    EvaluationQuery,
    MatrixRunner,
    MatrixRunResult,
    SingleRunExecutor,
    SingleRunResult,
)
from app.models.entities import ExperimentRun
from app.schemas.chunk import DocumentChunk
from app.schemas.experiment import (
    DatasetConfig,
    ExperimentConfig,
    PipelineConfig,
    RetrievalConfig,
    SweepParameters,
)


@pytest.fixture
async def test_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create isolated SQLite database engine with WAL & foreign keys."""
    db_file = Path("test_matrix_runner.db")
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
def session_factory(test_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return async sessionmaker for concurrent runs."""
    return get_session_factory(test_engine)


def create_sample_chunks() -> list[DocumentChunk]:
    """Helper to create sample DocumentChunk instances."""
    return [
        DocumentChunk.create(
            doc_id="doc-1",
            chunk_index=0,
            content="Photosynthesis converts sunlight into energy.",
            token_count=6,
            start_char=0,
            end_char=45,
            strategy="fixed",
            chunk_id="chunk-photo-001",
        ),
        DocumentChunk.create(
            doc_id="doc-2",
            chunk_index=0,
            content="Mitochondria produces cellular ATP.",
            token_count=5,
            start_char=0,
            end_char=35,
            strategy="fixed",
            chunk_id="chunk-resp-002",
        ),
    ]


async def populate_dataset_version(
    dataset_repo: DatasetRepository,
    version_id: str,
    chunks: list[DocumentChunk],
) -> None:
    """Register all chunks in DatasetVersion to satisfy reference validation."""
    doc_map: dict[str, str] = {}
    for c in chunks:
        if c.doc_id not in doc_map:
            docs = await dataset_repo.add_documents(
                version_id,
                [
                    {
                        "filename": f"{c.doc_id}.txt",
                        "content": c.content,
                        "content_hash": f"h-{c.doc_id}",
                    }
                ],
            )
            doc_map[c.doc_id] = docs[0].id
        await dataset_repo.add_chunks(
            doc_map[c.doc_id],
            [
                {
                    "id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "content_hash": f"ch-{c.chunk_id}",
                    "token_count": c.token_count,
                    "strategy": c.strategy,
                    "chunking_config_hash": "cfg",
                }
            ],
        )


@pytest.mark.anyio
async def test_matrix_runner_cartesian_expansion_and_execution(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Verify matrix runner expands combinatorial parameter sweeps and executes all points."""
    chunks = create_sample_chunks()
    mock_emb = DeterministicMockEmbeddingProvider(dimension=64)
    mock_llm = MockLLMClient()

    async with session_factory() as session:
        dataset_repo = DatasetRepository(session)
        exp_repo = ExperimentRepository(session)

        ds = await dataset_repo.create_dataset(name="Matrix DS")
        ver = await dataset_repo.create_version(
            dataset_id=ds.id, version_number=1, content_hash="hash-matrix-v1"
        )
        await populate_dataset_version(dataset_repo, ver.id, chunks)

        exp = await exp_repo.create_experiment(
            name="Matrix Exp",
            dataset_version_id=ver.id,
            configuration={"sweep": True},
            configuration_hash="hash-matrix-cfg",
        )
        await session.commit()
        exp_id = exp.id
        ds_id = ds.id
        ver_id = ver.id

    # 2 x 2 parameter sweep: 2 retrieval modes x 2 token budgets = 4 points
    exp_config = ExperimentConfig(
        dataset=DatasetConfig(dataset_id=ds_id, dataset_version_id=ver_id),
        parameters=SweepParameters(
            embedding={"provider": "mock", "dimension": 64},
            retrieval={"mode": ["bm25", "dense"], "top_k": 3},
            context={"token_budget": [512, 1024]},
            generation={"model_name": "mock"},
            evaluation={"metrics": ["recall", "precision"], "k_values": [1, 2]},
        ),
    )

    queries = [
        EvaluationQuery(
            query_id="q-1",
            query_text="photosynthesis sunlight",
            ground_truth_chunks=["chunk-photo-001"],
        ),
        EvaluationQuery(
            query_id="q-2",
            query_text="mitochondria ATP",
            ground_truth_chunks=["chunk-resp-002"],
        ),
    ]

    runner = MatrixRunner()
    result: MatrixRunResult = await runner.execute_matrix(
        experiment_id=exp_id,
        experiment_config=exp_config,
        queries=queries,
        corpus_chunks=chunks,
        session_factory=session_factory,
        max_concurrency=2,
        bypass_cache=False,
        llm_client=mock_llm,
        embedding_provider=mock_emb,
    )

    assert result.status == "COMPLETED"
    assert result.total_configurations == 4
    assert result.executed_runs == 4
    assert result.cached_runs == 0
    assert result.failed_runs == 0
    assert len(result.results) == 4
    for r in result.results:
        assert isinstance(r, SingleRunResult)
        assert r.status == "COMPLETED"
        assert not r.cached
        assert "recall@1" in r.mean_metrics


@pytest.mark.anyio
async def test_matrix_runner_caching_engine_hit_and_reuse(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Verify execution caching engine detects cache hits and reuses completed runs."""
    chunks = create_sample_chunks()
    mock_emb = DeterministicMockEmbeddingProvider(dimension=64)
    mock_llm = MockLLMClient()

    async with session_factory() as session:
        dataset_repo = DatasetRepository(session)
        exp_repo = ExperimentRepository(session)

        ds = await dataset_repo.create_dataset(name="Cache DS")
        ver = await dataset_repo.create_version(
            dataset_id=ds.id, version_number=1, content_hash="hash-cache-v1"
        )
        await populate_dataset_version(dataset_repo, ver.id, chunks)

        exp = await exp_repo.create_experiment(
            name="Cache Exp",
            dataset_version_id=ver.id,
            configuration={"cache_test": True},
            configuration_hash="hash-cache-cfg",
        )
        await session.commit()
        exp_id = exp.id
        ds_id = ds.id
        ver_id = ver.id

    queries = [
        EvaluationQuery(
            query_id="q-1",
            query_text="photosynthesis",
            ground_truth_chunks=["chunk-photo-001"],
        )
    ]

    # Initial run: 1 configuration (bm25)
    cfg_initial = ExperimentConfig(
        dataset=DatasetConfig(dataset_id=ds_id, dataset_version_id=ver_id),
        parameters=SweepParameters(
            embedding={"provider": "mock", "dimension": 64},
            retrieval={"mode": "bm25", "top_k": 3},
            evaluation={"metrics": ["recall"], "k_values": [1]},
        ),
    )

    runner = MatrixRunner()
    res1 = await runner.execute_matrix(
        experiment_id=exp_id,
        experiment_config=cfg_initial,
        queries=queries,
        corpus_chunks=chunks,
        session_factory=session_factory,
        bypass_cache=False,
        llm_client=mock_llm,
    )
    assert res1.executed_runs == 1
    assert res1.cached_runs == 0
    original_run_id = res1.results[0].experiment_run_id
    assert not res1.results[0].cached

    # Second run: sweep with bm25 AND dense
    cfg_expanded = ExperimentConfig(
        dataset=DatasetConfig(dataset_id=ds_id, dataset_version_id=ver_id),
        parameters=SweepParameters(
            embedding={"provider": "mock", "dimension": 64},
            retrieval={"mode": ["bm25", "dense"], "top_k": 3},
            evaluation={"metrics": ["recall"], "k_values": [1]},
        ),
    )

    res2 = await runner.execute_matrix(
        experiment_id=exp_id,
        experiment_config=cfg_expanded,
        queries=queries,
        corpus_chunks=chunks,
        session_factory=session_factory,
        bypass_cache=False,
        llm_client=mock_llm,
        embedding_provider=mock_emb,
    )

    assert res2.total_configurations == 2
    assert res2.cached_runs == 1  # bm25 reused from previous run
    assert res2.executed_runs == 1  # dense executed freshly
    assert res2.failed_runs == 0
    assert res2.status == "COMPLETED"

    # Verify cached run points to original run and duration_ms=0
    cached_result = next(r for r in res2.results if r.cached)
    assert cached_result.experiment_run_id == original_run_id
    assert cached_result.duration_ms == 0.0

    # Verify original run in DB was not duplicated
    async with session_factory() as session:
        all_runs = (await session.execute(select(ExperimentRun))).scalars().all()
        assert len(all_runs) == 2  # exactly 2 runs in DB (1 from initial, 1 from expanded)


@pytest.mark.anyio
async def test_matrix_runner_cache_only_reuses_completed(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Verify caching engine NEVER reuses FAILED, PARTIAL, or RUNNING runs (Amendment 2)."""
    chunks = create_sample_chunks()
    mock_llm = MockLLMClient()

    async with session_factory() as session:
        dataset_repo = DatasetRepository(session)
        exp_repo = ExperimentRepository(session)

        ds = await dataset_repo.create_dataset(name="Incomplete DS")
        ver = await dataset_repo.create_version(
            dataset_id=ds.id, version_number=1, content_hash="hash-inc-v1"
        )
        await populate_dataset_version(dataset_repo, ver.id, chunks)

        exp = await exp_repo.create_experiment(
            name="Incomplete Exp",
            dataset_version_id=ver.id,
            configuration={"inc": True},
            configuration_hash="hash-inc-cfg",
        )

        # Pre-seed a FAILED run with the same cache hash
        pipeline_cfg = PipelineConfig(
            dataset=DatasetConfig(dataset_id=ds.id, dataset_version_id=ver.id),
            retrieval=RetrievalConfig(mode="bm25", top_k=3),
        )
        cfg_hash = pipeline_cfg.compute_configuration_hash()
        cache_id = compute_cache_identity(
            dataset_id=ds.id,
            dataset_version_hash=ver.content_hash,
            pipeline_config_hash=cfg_hash,
        )

        failed_run = await exp_repo.create_run(
            experiment_id=exp.id,
            pipeline_config_hash=cfg_hash,
            cache_key=cache_id.cache_key,
            cache_hash=cache_id.cache_hash,
        )
        await exp_repo.complete_run(failed_run.id, status="FAILED", error="Simulated failure")
        await session.commit()
        exp_id = exp.id
        ds_id = ds.id
        ver_id = ver.id

    queries = [
        EvaluationQuery(
            query_id="q-1",
            query_text="photosynthesis",
            ground_truth_chunks=["chunk-photo-001"],
        )
    ]

    exp_config = ExperimentConfig(
        dataset=DatasetConfig(dataset_id=ds_id, dataset_version_id=ver_id),
        parameters=SweepParameters(
            retrieval={"mode": "bm25", "top_k": 3},
            evaluation={"metrics": ["recall"], "k_values": [1]},
        ),
    )

    runner = MatrixRunner()
    result = await runner.execute_matrix(
        experiment_id=exp_id,
        experiment_config=exp_config,
        queries=queries,
        corpus_chunks=chunks,
        session_factory=session_factory,
        bypass_cache=False,
        llm_client=mock_llm,
    )

    # Must NOT reuse the FAILED run -> must execute freshly and succeed
    assert result.cached_runs == 0
    assert result.executed_runs == 1
    assert result.status == "COMPLETED"
    assert result.results[0].status == "COMPLETED"
    assert not result.results[0].cached
    assert result.results[0].experiment_run_id != failed_run.id


@pytest.mark.anyio
async def test_matrix_runner_bypass_cache_forces_reexecution(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Verify bypass_cache=True executes fresh run and preserves historical runs."""
    chunks = create_sample_chunks()
    mock_llm = MockLLMClient()

    async with session_factory() as session:
        dataset_repo = DatasetRepository(session)
        exp_repo = ExperimentRepository(session)

        ds = await dataset_repo.create_dataset(name="Bypass DS")
        ver = await dataset_repo.create_version(
            dataset_id=ds.id, version_number=1, content_hash="hash-byp-v1"
        )
        await populate_dataset_version(dataset_repo, ver.id, chunks)

        exp = await exp_repo.create_experiment(
            name="Bypass Exp",
            dataset_version_id=ver.id,
            configuration={"bypass": True},
            configuration_hash="hash-byp-cfg",
        )
        await session.commit()
        exp_id = exp.id
        ds_id = ds.id
        ver_id = ver.id

    queries = [
        EvaluationQuery(
            query_id="q-1",
            query_text="photosynthesis",
            ground_truth_chunks=["chunk-photo-001"],
        )
    ]

    exp_config = ExperimentConfig(
        dataset=DatasetConfig(dataset_id=ds_id, dataset_version_id=ver_id),
        parameters=SweepParameters(
            retrieval={"mode": "bm25", "top_k": 3},
            evaluation={"metrics": ["recall"], "k_values": [1]},
        ),
    )

    runner = MatrixRunner()

    # Run 1: Normal execution
    res1 = await runner.execute_matrix(
        experiment_id=exp_id,
        experiment_config=exp_config,
        queries=queries,
        corpus_chunks=chunks,
        session_factory=session_factory,
        bypass_cache=False,
        llm_client=mock_llm,
    )
    assert res1.executed_runs == 1
    assert res1.cached_runs == 0
    first_run_id = res1.results[0].experiment_run_id

    # Run 2: Bypass cache -> forces fresh execution
    res2 = await runner.execute_matrix(
        experiment_id=exp_id,
        experiment_config=exp_config,
        queries=queries,
        corpus_chunks=chunks,
        session_factory=session_factory,
        bypass_cache=True,
        llm_client=mock_llm,
    )
    assert res2.executed_runs == 1
    assert res2.cached_runs == 0
    second_run_id = res2.results[0].experiment_run_id
    assert second_run_id != first_run_id

    # Verify both runs are preserved in the DB
    async with session_factory() as session:
        all_runs = (await session.execute(select(ExperimentRun))).scalars().all()
        assert len(all_runs) == 2

    # Run 3: Normal execution without bypass -> deterministically selects latest completed run
    res3 = await runner.execute_matrix(
        experiment_id=exp_id,
        experiment_config=exp_config,
        queries=queries,
        corpus_chunks=chunks,
        session_factory=session_factory,
        bypass_cache=False,
        llm_client=mock_llm,
    )
    assert res3.cached_runs == 1
    assert res3.executed_runs == 0
    assert res3.results[0].experiment_run_id == second_run_id


@pytest.mark.anyio
async def test_matrix_runner_deterministic_result_ordering(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Verify MatrixRunResult.results strictly preserves CartesianExpander order (Amendment 5)."""
    chunks = create_sample_chunks()
    mock_emb = DeterministicMockEmbeddingProvider(dimension=64)

    async with session_factory() as session:
        dataset_repo = DatasetRepository(session)
        exp_repo = ExperimentRepository(session)

        ds = await dataset_repo.create_dataset(name="Order DS")
        ver = await dataset_repo.create_version(
            dataset_id=ds.id, version_number=1, content_hash="hash-order-v1"
        )
        await populate_dataset_version(dataset_repo, ver.id, chunks)

        exp = await exp_repo.create_experiment(
            name="Order Exp",
            dataset_version_id=ver.id,
            configuration={"order": True},
            configuration_hash="hash-order-cfg",
        )
        await session.commit()
        exp_id = exp.id
        ds_id = ds.id
        ver_id = ver.id

    exp_config = ExperimentConfig(
        dataset=DatasetConfig(dataset_id=ds_id, dataset_version_id=ver_id),
        parameters=SweepParameters(
            embedding={"provider": "mock", "dimension": 64},
            retrieval={"mode": ["bm25", "dense"], "top_k": [3, 5]},
            evaluation={"metrics": ["recall"], "k_values": [1]},
        ),
    )

    # Establish canonical order from CartesianExpander
    expander = CartesianExpander()
    expected_configs = expander.expand(exp_config)
    expected_hashes = [c.compute_configuration_hash() for c in expected_configs]

    queries = [
        EvaluationQuery(
            query_id="q-1",
            query_text="photosynthesis",
            ground_truth_chunks=["chunk-photo-001"],
        )
    ]

    runner = MatrixRunner()
    result = await runner.execute_matrix(
        experiment_id=exp_id,
        experiment_config=exp_config,
        queries=queries,
        corpus_chunks=chunks,
        session_factory=session_factory,
        max_concurrency=4,
        llm_client=MockLLMClient(),
        embedding_provider=mock_emb,
    )

    actual_hashes = [r.pipeline_config_hash for r in result.results]
    assert actual_hashes == expected_hashes


@pytest.mark.anyio
async def test_matrix_runner_concurrency_bound_measurement(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Instrument executor to prove active executions never exceed max_concurrency (Amendment 9)."""
    chunks = create_sample_chunks()

    async with session_factory() as session:
        dataset_repo = DatasetRepository(session)
        exp_repo = ExperimentRepository(session)

        ds = await dataset_repo.create_dataset(name="Conc DS")
        ver = await dataset_repo.create_version(
            dataset_id=ds.id, version_number=1, content_hash="hash-conc-v1"
        )
        await populate_dataset_version(dataset_repo, ver.id, chunks)

        exp = await exp_repo.create_experiment(
            name="Conc Exp",
            dataset_version_id=ver.id,
            configuration={"conc": True},
            configuration_hash="hash-conc-cfg",
        )
        await session.commit()
        exp_id = exp.id
        ds_id = ds.id
        ver_id = ver.id

    # 4 distinct pipeline configurations
    exp_config = ExperimentConfig(
        dataset=DatasetConfig(dataset_id=ds_id, dataset_version_id=ver_id),
        parameters=SweepParameters(
            retrieval={"mode": "bm25", "top_k": [1, 2, 3, 4]},
            evaluation={"metrics": ["recall"], "k_values": [1]},
        ),
    )

    queries = [
        EvaluationQuery(
            query_id="q-1",
            query_text="photosynthesis",
            ground_truth_chunks=["chunk-photo-001"],
        )
    ]

    # Instrument SingleRunExecutor to monitor concurrency
    class InstrumentingExecutor(SingleRunExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.current_active = 0
            self.max_observed = 0
            self.lock = asyncio.Lock()

        async def execute(self, *args: Any, **kwargs: Any) -> SingleRunResult:
            async with self.lock:
                self.current_active += 1
                if self.current_active > self.max_observed:
                    self.max_observed = self.current_active
            try:
                # Small non-zero sleep to induce concurrency overlap
                await asyncio.sleep(0.05)
                return await super().execute(*args, **kwargs)
            finally:
                async with self.lock:
                    self.current_active -= 1

    # Test 1: max_concurrency = 2
    instrumented_exec = InstrumentingExecutor()
    runner = MatrixRunner(single_run_executor=instrumented_exec)

    result = await runner.execute_matrix(
        experiment_id=exp_id,
        experiment_config=exp_config,
        queries=queries,
        corpus_chunks=chunks,
        session_factory=session_factory,
        max_concurrency=2,
        bypass_cache=True,
        llm_client=MockLLMClient(),
    )

    assert result.status == "COMPLETED"
    assert result.executed_runs == 4
    assert instrumented_exec.max_observed <= 2
    assert instrumented_exec.max_observed > 0

    # Test 2: max_concurrency = 1 (strict sequential compatibility)
    instrumented_seq = InstrumentingExecutor()
    runner_seq = MatrixRunner(single_run_executor=instrumented_seq)

    res_seq = await runner_seq.execute_matrix(
        experiment_id=exp_id,
        experiment_config=exp_config,
        queries=queries,
        corpus_chunks=chunks,
        session_factory=session_factory,
        max_concurrency=1,
        bypass_cache=True,
        llm_client=MockLLMClient(),
    )

    assert res_seq.status == "COMPLETED"
    assert instrumented_seq.max_observed == 1


@pytest.mark.anyio
async def test_matrix_runner_partial_failure_isolation(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Verify that a failure in one configuration isolates cleanly (Amendment 8)."""
    chunks = create_sample_chunks()

    async with session_factory() as session:
        dataset_repo = DatasetRepository(session)
        exp_repo = ExperimentRepository(session)

        ds = await dataset_repo.create_dataset(name="Iso DS")
        ver = await dataset_repo.create_version(
            dataset_id=ds.id, version_number=1, content_hash="hash-iso-v1"
        )
        await populate_dataset_version(dataset_repo, ver.id, chunks)

        exp = await exp_repo.create_experiment(
            name="Iso Exp",
            dataset_version_id=ver.id,
            configuration={"iso": True},
            configuration_hash="hash-iso-cfg",
        )
        await session.commit()
        exp_id = exp.id
        ds_id = ds.id
        ver_id = ver.id

    class CrashingOnBudgetMockLLM(MockLLMClient):
        def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: object) -> str:
            if "CRASH_ME" in prompt:
                raise RuntimeError("Targeted prompt crash")
            return "Valid answer"

    # 2 queries, where second query crashes on all runs
    queries = [
        EvaluationQuery(
            query_id="q-ok", query_text="Normal", ground_truth_chunks=["chunk-photo-001"]
        ),
        EvaluationQuery(
            query_id="q-fail", query_text="CRASH_ME", ground_truth_chunks=["chunk-photo-001"]
        ),
    ]

    exp_config = ExperimentConfig(
        dataset=DatasetConfig(dataset_id=ds_id, dataset_version_id=ver_id),
        parameters=SweepParameters(
            retrieval={"mode": "bm25", "top_k": 3},
            evaluation={"metrics": ["recall"], "k_values": [1]},
        ),
    )

    runner = MatrixRunner()
    result = await runner.execute_matrix(
        experiment_id=exp_id,
        experiment_config=exp_config,
        queries=queries,
        corpus_chunks=chunks,
        session_factory=session_factory,
        llm_client=CrashingOnBudgetMockLLM(),
    )

    # 1 of 2 queries failed in the single configuration -> SingleRunResult is PARTIAL
    assert result.total_configurations == 1
    assert result.results[0].status == "PARTIAL"
    assert result.status == "FAILED"  # all attempted executions had failures
    assert result.failed_runs == 1
    assert result.executed_runs == 0


@pytest.mark.anyio
async def test_matrix_runner_successful_survives_failure(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Verify successful configurations survive and persist while failing configurations isolate."""
    chunks = create_sample_chunks()

    async with session_factory() as session:
        dataset_repo = DatasetRepository(session)
        exp_repo = ExperimentRepository(session)

        ds = await dataset_repo.create_dataset(name="Survive DS")
        ver = await dataset_repo.create_version(
            dataset_id=ds.id, version_number=1, content_hash="hash-survive-v1"
        )
        await populate_dataset_version(dataset_repo, ver.id, chunks)

        exp = await exp_repo.create_experiment(
            name="Survive Exp",
            dataset_version_id=ver.id,
            configuration={"survive": True},
            configuration_hash="hash-survive-cfg",
        )
        await session.commit()
        exp_id = exp.id
        ds_id = ds.id
        ver_id = ver.id

    # 2 configs: top_k=1 and top_k=2
    exp_config = ExperimentConfig(
        dataset=DatasetConfig(dataset_id=ds_id, dataset_version_id=ver_id),
        parameters=SweepParameters(
            retrieval={"mode": "bm25", "top_k": [1, 2]},
            evaluation={"metrics": ["recall"], "k_values": [1]},
        ),
    )

    queries = [
        EvaluationQuery(
            query_id="q-1",
            query_text="photosynthesis",
            ground_truth_chunks=["chunk-photo-001"],
        )
    ]

    class CrashingConfigExecutor(SingleRunExecutor):
        async def execute(self, *args: Any, **kwargs: Any) -> SingleRunResult:
            pipeline_cfg: PipelineConfig = kwargs["pipeline_config"]
            if pipeline_cfg.retrieval.top_k == 2:
                raise RuntimeError("Top_k 2 crashed intentionally")
            return await super().execute(*args, **kwargs)

    runner = MatrixRunner(single_run_executor=CrashingConfigExecutor())
    result = await runner.execute_matrix(
        experiment_id=exp_id,
        experiment_config=exp_config,
        queries=queries,
        corpus_chunks=chunks,
        session_factory=session_factory,
        bypass_cache=True,
        llm_client=MockLLMClient(),
    )

    assert result.total_configurations == 2
    assert result.status == "PARTIAL"  # 1 succeeded, 1 failed
    assert result.executed_runs == 1
    assert result.failed_runs == 1
    assert result.results[0].status == "COMPLETED"
    assert result.results[1].status == "FAILED"


@pytest.mark.anyio
async def test_matrix_runner_duplicate_configurations_deduplication(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Verify duplicate matrix configurations execute only once via in-flight dedup."""
    chunks = create_sample_chunks()

    async with session_factory() as session:
        dataset_repo = DatasetRepository(session)
        exp_repo = ExperimentRepository(session)

        ds = await dataset_repo.create_dataset(name="Dedup DS")
        ver = await dataset_repo.create_version(
            dataset_id=ds.id, version_number=1, content_hash="hash-dedup-v1"
        )
        await populate_dataset_version(dataset_repo, ver.id, chunks)

        exp = await exp_repo.create_experiment(
            name="Dedup Exp",
            dataset_version_id=ver.id,
            configuration={"dedup": True},
            configuration_hash="hash-dedup-cfg",
        )
        await session.commit()
        exp_id = exp.id
        ds_id = ds.id
        ver_id = ver.id

    queries = [
        EvaluationQuery(
            query_id="q-1",
            query_text="photosynthesis",
            ground_truth_chunks=["chunk-photo-001"],
        )
    ]

    # Matrix with identical parameters duplicated across sweep
    exp_config = ExperimentConfig(
        dataset=DatasetConfig(dataset_id=ds_id, dataset_version_id=ver_id),
        parameters=SweepParameters(
            retrieval={"mode": "bm25", "top_k": 3},
            evaluation={"metrics": ["recall"], "k_values": [1]},
        ),
    )

    execution_count = 0
    lock = asyncio.Lock()

    class CountingExecutor(SingleRunExecutor):
        async def execute(self, *args: Any, **kwargs: Any) -> SingleRunResult:
            nonlocal execution_count
            async with lock:
                execution_count += 1
            await asyncio.sleep(0.05)
            return await super().execute(*args, **kwargs)

    # Subclass expander to produce duplicate identical configs
    # Manually expand into 2 identical configs
    expander = CartesianExpander()
    single_cfg = expander.expand(exp_config)[0]

    # Patch expander on runner
    class DuplicateExpander(CartesianExpander):
        @classmethod
        def expand(
            cls,
            config: ExperimentConfig,
            baseline_pipeline: PipelineConfig | None = None,
            skip_invalid: bool = False,
        ) -> list[PipelineConfig]:
            return [single_cfg, single_cfg]

    runner_with_dups = MatrixRunner(
        single_run_executor=CountingExecutor(),
        expander=DuplicateExpander(),
    )

    result = await runner_with_dups.execute_matrix(
        experiment_id=exp_id,
        experiment_config=exp_config,
        queries=queries,
        corpus_chunks=chunks,
        session_factory=session_factory,
        max_concurrency=4,
        bypass_cache=False,
        llm_client=MockLLMClient(),
    )

    assert result.total_configurations == 2
    assert result.status == "COMPLETED"
    assert result.executed_runs == 1
    assert result.cached_runs == 1
    assert result.results[0].experiment_run_id == result.results[1].experiment_run_id
    assert execution_count == 1  # Executed exactly once!


@pytest.mark.anyio
async def test_matrix_runner_cache_identity_matches_f2() -> None:
    """Verify cache identity matches F2 compute_cache_identity contract (Amendment 1)."""
    cfg1 = PipelineConfig(
        dataset=DatasetConfig(dataset_id="ds-1", dataset_version_id="v1"),
        retrieval=RetrievalConfig(mode="bm25", top_k=3),
    )
    cfg2 = PipelineConfig(
        dataset=DatasetConfig(dataset_id="ds-1", dataset_version_id="v1"),
        retrieval=RetrievalConfig(mode="bm25", top_k=3),
    )
    cfg3 = PipelineConfig(
        dataset=DatasetConfig(dataset_id="ds-1", dataset_version_id="v1"),
        retrieval=RetrievalConfig(mode="dense", top_k=3),
    )

    id1 = compute_cache_identity("ds-1", "hash-v1", cfg1.compute_configuration_hash())
    id2 = compute_cache_identity("ds-1", "hash-v1", cfg2.compute_configuration_hash())
    id3 = compute_cache_identity("ds-1", "hash-v1", cfg3.compute_configuration_hash())

    # Invariant: identical semantic config -> identical cache hash
    assert id1.cache_hash == id2.cache_hash
    assert id1.cache_key == id2.cache_key

    # Invariant: semantic modification -> different cache hash
    assert id1.cache_hash != id3.cache_hash


@pytest.mark.anyio
async def test_matrix_runner_single_session_fallback(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Verify MatrixRunner functions seamlessly with a single shared session."""
    chunks = create_sample_chunks()
    mock_llm = MockLLMClient()

    async with session_factory() as session:
        dataset_repo = DatasetRepository(session)
        exp_repo = ExperimentRepository(session)

        ds = await dataset_repo.create_dataset(name="Single Sess DS")
        ver = await dataset_repo.create_version(
            dataset_id=ds.id, version_number=1, content_hash="hash-ss-v1"
        )
        await populate_dataset_version(dataset_repo, ver.id, chunks)

        exp = await exp_repo.create_experiment(
            name="Single Sess Exp",
            dataset_version_id=ver.id,
            configuration={"ss": True},
            configuration_hash="hash-ss-cfg",
        )
        await session.commit()

        exp_config = ExperimentConfig(
            dataset=DatasetConfig(dataset_id=ds.id, dataset_version_id=ver.id),
            parameters=SweepParameters(
                retrieval={"mode": "bm25", "top_k": 3},
                evaluation={"metrics": ["recall"], "k_values": [1]},
            ),
        )

        queries = [
            EvaluationQuery(
                query_id="q-1",
                query_text="photosynthesis",
                ground_truth_chunks=["chunk-photo-001"],
            )
        ]

        runner = MatrixRunner()
        result = await runner.execute_matrix(
            experiment_id=exp.id,
            experiment_config=exp_config,
            queries=queries,
            corpus_chunks=chunks,
            session=session,
            max_concurrency=4,  # Clamped to 1 because single session passed
            llm_client=mock_llm,
        )

        assert result.status == "COMPLETED"
        assert result.total_configurations == 1
        assert result.executed_runs == 1
        assert result.results[0].status == "COMPLETED"
