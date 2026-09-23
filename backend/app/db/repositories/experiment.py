"""Repository for Experiment, ExperimentRun, and RunMetricSummary persistence."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.base import utc_now
from app.models.entities import Experiment, ExperimentRun, RunMetricSummary


class ExperimentRepository:
    """Async repository for managing experiments, runs, and metric summaries."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_experiment(
        self,
        name: str,
        dataset_version_id: str,
        configuration: dict[str, Any],
        configuration_hash: str,
        description: str | None = None,
        created_by: str | None = None,
    ) -> Experiment:
        """Create and persist an Experiment definition."""
        experiment = Experiment(
            name=name,
            description=description,
            dataset_version_id=dataset_version_id,
            configuration=configuration,
            configuration_hash=configuration_hash,
            created_by=created_by,
        )
        self.session.add(experiment)
        await self.session.flush()
        return experiment

    async def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Fetch Experiment by ID."""
        stmt = (
            select(Experiment)
            .where(Experiment.id == experiment_id)
            .options(selectinload(Experiment.runs))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_run(
        self,
        experiment_id: str,
        pipeline_config_hash: str,
        cache_key: str,
        cache_hash: str,
        environment: str = "local",
        random_seed: int = 42,
        git_commit: str | None = None,
    ) -> ExperimentRun:
        """Create an individual physical execution run of a pipeline point."""
        run = ExperimentRun(
            experiment_id=experiment_id,
            pipeline_config_hash=pipeline_config_hash,
            cache_key=cache_key,
            cache_hash=cache_hash,
            environment=environment,
            random_seed=random_seed,
            git_commit=git_commit,
            status="RUNNING",
        )
        self.session.add(run)
        await self.session.flush()
        return run

    async def get_run(self, run_id: str) -> ExperimentRun | None:
        """Fetch ExperimentRun by ID with metric summaries."""
        stmt = (
            select(ExperimentRun)
            .where(ExperimentRun.id == run_id)
            .options(selectinload(ExperimentRun.metric_summaries))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_run_by_cache_hash(self, cache_hash: str) -> ExperimentRun | None:
        """Fetch completed ExperimentRun by dataset-version-aware cache hash."""
        stmt = (
            select(ExperimentRun)
            .where(
                ExperimentRun.cache_hash == cache_hash,
                ExperimentRun.status == "COMPLETED",
            )
            .options(selectinload(ExperimentRun.metric_summaries))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def complete_run(
        self,
        run_id: str,
        summary_metrics: dict[str, Any] | None = None,
        status: str = "COMPLETED",
        error: str | None = None,
    ) -> ExperimentRun:
        """Mark an ExperimentRun as completed or failed."""
        run = await self.get_run(run_id)
        if not run:
            raise ValueError(f"ExperimentRun '{run_id}' not found")

        run.status = status
        run.completed_at = utc_now()
        run.summary_metrics = summary_metrics or {}
        run.error = error

        await self.session.flush()
        return run

    async def add_metric_summaries(
        self,
        experiment_run_id: str,
        summaries: list[dict[str, Any]],
    ) -> list[RunMetricSummary]:
        """Persist statistical rollups for an ExperimentRun."""
        entities: list[RunMetricSummary] = []
        for s in summaries:
            summary = RunMetricSummary(
                experiment_run_id=experiment_run_id,
                metric_name=s["metric_name"],
                mean=float(s["mean"]),
                median=float(s["median"]),
                min=float(s["min"]),
                max=float(s["max"]),
                stddev=float(s.get("stddev", 0.0)),
                count=int(s["count"]),
                metadata_json=s.get("metadata", {}),
            )
            self.session.add(summary)
            entities.append(summary)

        await self.session.flush()
        return entities

    async def get_metric_summaries(
        self,
        experiment_run_id: str,
    ) -> list[RunMetricSummary]:
        """Fetch all statistical metric summaries for an ExperimentRun."""
        stmt = (
            select(RunMetricSummary)
            .where(RunMetricSummary.experiment_run_id == experiment_run_id)
            .order_by(RunMetricSummary.metric_name)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
