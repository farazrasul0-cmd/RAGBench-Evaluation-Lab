"""Matrix Pipeline Runner for combinatorial sweeps and execution caching (Phase F5).

Wraps the canonical F4 SingleRunExecutor to orchestrate Cartesian-expanded parameter
sweeps with bounded concurrency, execution caching, and per-configuration fault isolation.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.repositories.dataset import DatasetRepository
from app.db.repositories.experiment import ExperimentRepository
from app.engine.embeddings.base import BaseEmbeddingProvider
from app.engine.orchestrator import CartesianExpander, compute_cache_identity
from app.engine.query_transforms.base import BaseLLMClient
from app.engine.runner.models import EvaluationQuery, MatrixRunResult, SingleRunResult
from app.engine.runner.single_run import SingleRunExecutor
from app.schemas.chunk import DocumentChunk
from app.schemas.experiment import ExperimentConfig, PipelineConfig


class MatrixRunner:
    """Orchestrates parallel execution, caching, and result ordering across experiment matrices."""

    def __init__(
        self,
        single_run_executor: SingleRunExecutor | None = None,
        expander: CartesianExpander | None = None,
    ) -> None:
        self.single_run_executor = single_run_executor or SingleRunExecutor()
        self.expander = expander or CartesianExpander()

    async def execute_matrix(
        self,
        experiment_id: str,
        experiment_config: ExperimentConfig,
        queries: list[EvaluationQuery],
        corpus_chunks: list[DocumentChunk],
        session: AsyncSession | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        max_concurrency: int = 4,
        bypass_cache: bool = False,
        llm_client: BaseLLMClient | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        validate_chunk_references: bool = True,
        environment: str = "local",
        random_seed: int = 42,
        git_commit: str | None = None,
    ) -> MatrixRunResult:
        """Expand and execute an ExperimentConfig sweep across benchmark evaluation queries.

        Args:
            experiment_id: Parent Experiment entity identifier.
            experiment_config: ExperimentConfig containing parameter sweeps.
            queries: Benchmark evaluation queries with ground truth annotations.
            corpus_chunks: In-memory corpus chunks available for retrieval indexing.
            session: Optional existing SQLAlchemy async session (for single-session execution).
            session_factory: Optional sessionmaker producing fresh sessions for concurrent runs.
            max_concurrency: Maximum number of concurrent pipeline evaluations (default 4).
            bypass_cache: If True, forces execution even if completed runs exist in DB.
            llm_client: Optional custom LLM client.
            embedding_provider: Optional custom embedding provider.
            validate_chunk_references: Whether to enforce chunk reference integrity.
            environment: Execution environment descriptor.
            random_seed: Random seed for deterministic reproducibility.
            git_commit: Optional git commit SHA recording code state.

        Returns:
            MatrixRunResult with aggregate matrix status, cache counts, and run reports
            strictly ordered according to CartesianExpander configuration sequence.
        """
        start_time = time.perf_counter()

        if session is None and session_factory is None:
            raise ValueError("Either session or session_factory must be provided to MatrixRunner")

        # Bound concurrency to 1 if single shared session provided to avoid session conflicts
        effective_concurrency = max_concurrency if session_factory is not None else 1
        semaphore = asyncio.Semaphore(effective_concurrency)

        @asynccontextmanager
        async def get_session() -> AsyncGenerator[AsyncSession, None]:
            if session_factory is not None:
                async with session_factory() as s:
                    yield s
            elif session is not None:
                yield session
            else:
                raise ValueError("No active session available")

        # 1. Fetch Experiment and DatasetVersion metadata
        async with get_session() as init_session:
            exp_repo = ExperimentRepository(init_session)
            dataset_repo = DatasetRepository(init_session)
            experiment = await exp_repo.get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment '{experiment_id}' not found")
            dataset_version = await dataset_repo.get_version(experiment.dataset_version_id)
            if not dataset_version:
                raise ValueError(f"DatasetVersion '{experiment.dataset_version_id}' not found")
            dataset_id = str(dataset_version.dataset_id)
            content_hash = str(dataset_version.content_hash)

        # 2. Deterministic Cartesian expansion of combinatorial sweeps
        pipeline_configs = self.expander.expand(experiment_config)

        if not pipeline_configs:
            return MatrixRunResult(
                experiment_id=experiment_id,
                total_configurations=0,
                executed_runs=0,
                cached_runs=0,
                failed_runs=0,
                status="COMPLETED",
                results=[],
                duration_ms=0.0,
            )

        # In-flight deduplication tracking to prevent duplicate matrix entries executing redundantly
        in_flight_tasks: dict[str, asyncio.Future[SingleRunResult]] = {}
        in_flight_lock = asyncio.Lock()

        # 3. Process individual pipeline point with caching, in-flight deduplication & F4 delegation
        async def _process_point(
            index: int, pipeline_cfg: PipelineConfig
        ) -> tuple[int, SingleRunResult]:
            cfg_hash = pipeline_cfg.compute_configuration_hash()
            cache_id = compute_cache_identity(
                dataset_id=dataset_id,
                dataset_version_hash=content_hash,
                pipeline_config_hash=cfg_hash,
            )
            cache_hash = cache_id.cache_hash

            # Cache lookup performed BEFORE acquiring the execution semaphore (Amendment 6)
            if not bypass_cache:
                async with get_session() as lookup_session:
                    point_exp_repo = ExperimentRepository(lookup_session)
                    cached_run = await point_exp_repo.get_run_by_cache_hash(cache_hash)
                    if cached_run is not None:
                        # Cache hit: return existing run without fake execution
                        return index, SingleRunResult(
                            experiment_run_id=cached_run.id,
                            experiment_id=experiment_id,
                            pipeline_config_hash=cached_run.pipeline_config_hash,
                            cache_hash=cached_run.cache_hash,
                            status="COMPLETED",
                            total_queries=len(queries),
                            completed_queries=len(queries),
                            failed_queries=0,
                            mean_metrics=cached_run.summary_metrics or {},
                            duration_ms=0.0,
                            cached=True,
                        )

            # In-flight execution coordination for duplicate configs within the matrix (Amendment 4)
            is_leader = False
            leader_future: asyncio.Future[SingleRunResult] | None = None

            if not bypass_cache:
                async with in_flight_lock:
                    if cache_hash in in_flight_tasks:
                        leader_future = in_flight_tasks[cache_hash]
                    else:
                        loop = asyncio.get_running_loop()
                        new_future: asyncio.Future[SingleRunResult] = loop.create_future()
                        in_flight_tasks[cache_hash] = new_future
                        leader_future = new_future
                        is_leader = True

            if not is_leader and leader_future is not None:
                # Follower waits for the in-flight leader's execution to complete
                leader_result = await leader_future
                return index, SingleRunResult(
                    experiment_run_id=leader_result.experiment_run_id,
                    experiment_id=experiment_id,
                    pipeline_config_hash=leader_result.pipeline_config_hash,
                    cache_hash=leader_result.cache_hash,
                    status=leader_result.status,
                    total_queries=leader_result.total_queries,
                    completed_queries=leader_result.completed_queries,
                    failed_queries=leader_result.failed_queries,
                    mean_metrics=leader_result.mean_metrics,
                    duration_ms=0.0,
                    cached=True,
                )

            # Execution block: acquire semaphore only for actual execution (Amendment 6)
            try:
                async with semaphore, get_session() as run_session:
                    # Delegate exclusively to canonical SingleRunExecutor (F4)
                    exec_res = await self.single_run_executor.execute(
                        experiment_id=experiment_id,
                        pipeline_config=pipeline_cfg,
                        queries=queries,
                        corpus_chunks=corpus_chunks,
                        session=run_session,
                        llm_client=llm_client,
                        embedding_provider=embedding_provider,
                        validate_chunk_references=validate_chunk_references,
                        environment=environment,
                        random_seed=random_seed,
                        git_commit=git_commit,
                    )

                if is_leader and leader_future is not None and not leader_future.done():
                    leader_future.set_result(exec_res)

                return index, exec_res

            except Exception:
                # Partial failure isolation: capture configuration-level crashes (Amendment 8)
                fail_res = SingleRunResult(
                    experiment_run_id="",
                    experiment_id=experiment_id,
                    pipeline_config_hash=cfg_hash,
                    cache_hash=cache_hash,
                    status="FAILED",
                    total_queries=len(queries),
                    completed_queries=0,
                    failed_queries=len(queries),
                    mean_metrics={},
                    duration_ms=0.0,
                    cached=False,
                )
                if is_leader and leader_future is not None and not leader_future.done():
                    leader_future.set_result(fail_res)

                return index, fail_res

        # 4. Schedule concurrent worker execution
        tasks = [_process_point(idx, cfg) for idx, cfg in enumerate(pipeline_configs)]
        indexed_results = await asyncio.gather(*tasks)

        # 5. Deterministic Cartesian ordering guarantee (Amendment 5)
        indexed_results.sort(key=lambda x: x[0])
        ordered_results = [res for _, res in indexed_results]

        # 6. Aggregate matrix-level status and statistics (Amendment 8)
        total_configurations = len(pipeline_configs)
        cached_runs = sum(1 for r in ordered_results if r.cached)
        executed_runs = sum(1 for r in ordered_results if not r.cached and r.status == "COMPLETED")
        failed_runs = sum(1 for r in ordered_results if r.status in ("FAILED", "PARTIAL"))
        completed_or_cached = sum(1 for r in ordered_results if r.status == "COMPLETED")

        if failed_runs == 0:
            status = "COMPLETED"
        elif completed_or_cached == 0:
            status = "FAILED"
        else:
            status = "PARTIAL"

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return MatrixRunResult(
            experiment_id=experiment_id,
            total_configurations=total_configurations,
            executed_runs=executed_runs,
            cached_runs=cached_runs,
            failed_runs=failed_runs,
            status=status,
            results=ordered_results,
            duration_ms=duration_ms,
        )
