"""Experiment CLI commands: run, plan, show, and compare."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import click

from app.cli.formatting import format_delta, render_table
from app.db.repositories.dataset import DatasetRepository
from app.db.repositories.experiment import ExperimentRepository
from app.db.repositories.query_trace import QueryTraceRepository
from app.db.session import get_async_engine, get_session_factory, init_db
from app.engine.orchestrator.dry_run import DryRunPlanner
from app.engine.runner.matrix import MatrixRunner
from app.engine.runner.models import EvaluationQuery, MatrixRunResult
from app.schemas.chunk import DocumentChunk
from app.schemas.experiment import ExperimentConfig


@click.group(name="experiment")
def experiment_group() -> None:
    """Manage, plan, execute, and analyze benchmark experiments."""


@experiment_group.command(name="plan")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--db-url", default=None, help="Database connection URL override.")
def plan_command(config_path: Path, db_url: str | None) -> None:
    """Dry-run preview of combinatorial sweeps, component diffs, and cache hits."""
    try:
        config = ExperimentConfig.from_yaml(config_path)
    except Exception as exc:
        click.echo(f"Error loading configuration: {exc}", err=True)
        sys.exit(1)

    async def _plan() -> int:
        engine = get_async_engine(db_url)
        await init_db(engine)
        session_factory = get_session_factory(engine)

        planner = DryRunPlanner()
        plan_res = planner.plan(config)

        click.echo(f"\nExperiment Plan: {config.metadata.name}")
        click.echo(f"Total Combinatorial Configurations: {plan_res.total_planned_pipelines}\n")

        headers = ["#", "Config Hash", "Cache Hash", "Cache Status", "Diff vs Baseline"]
        rows: list[list[Any]] = []

        async with session_factory() as session:
            exp_repo = ExperimentRepository(session)
            for idx, pipe in enumerate(plan_res.planned_runs, start=1):
                cache_status = "MISS"
                diff_summary = "Baseline"
                # Check DB cache
                cached_run = await exp_repo.get_run_by_cache_hash(pipe.cache_hash)
                if cached_run is not None:
                    cache_status = "HIT"

                if pipe.parameter_diff:
                    diff_parts = []
                    for comp, fields in pipe.parameter_diff.items():
                        diff_parts.append(f"{comp}: {fields}")
                    diff_summary = "; ".join(diff_parts)

                rows.append(
                    [
                        idx,
                        pipe.pipeline_config_hash[:12],
                        pipe.cache_hash[:12],
                        cache_status,
                        diff_summary,
                    ]
                )

        click.echo(render_table(headers, rows, title="Planned Execution Matrix"))
        await engine.dispose()
        return 0

    code = asyncio.run(_plan())
    sys.exit(code)


@experiment_group.command(name="run")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--concurrency", "-c", default=4, type=int, help="Maximum concurrent pipeline executions."
)
@click.option(
    "--bypass-cache", is_flag=True, default=False, help="Bypass cache and force re-execution."
)
@click.option(
    "--fail-on-partial",
    is_flag=True,
    default=False,
    help="Return exit code 1 if matrix status is PARTIAL.",
)
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Export execution summary to JSON.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress detailed tables, show summary only.",
)
@click.option("--db-url", default=None, help="Database connection URL override.")
def run_command(
    config_path: Path,
    concurrency: int,
    bypass_cache: bool,
    fail_on_partial: bool,
    output_json: Path | None,
    quiet: bool,
    db_url: str | None,
) -> None:
    """Execute an experiment matrix sweep across benchmark evaluation queries."""
    try:
        config = ExperimentConfig.from_yaml(config_path)
    except Exception as exc:
        click.echo(f"Error loading configuration: {exc}", err=True)
        sys.exit(1)

    async def _run() -> int:
        engine = get_async_engine(db_url)
        await init_db(engine)
        session_factory = get_session_factory(engine)

        async with session_factory() as session:
            exp_repo = ExperimentRepository(session)
            dataset_repo = DatasetRepository(session)

            # Resolve target dataset version
            ver_id = config.dataset.dataset_version_id
            ver = await dataset_repo.get_version(ver_id)
            if not ver:
                # If not found by version ID, attempt to find latest by dataset_id or dataset_name
                ds = await dataset_repo.get_dataset(config.dataset.dataset_id)
                if not ds:
                    ds = await dataset_repo.get_dataset_by_name(config.dataset.dataset_id)
                if ds and ds.versions:
                    ver = ds.versions[-1]
                else:
                    click.echo(
                        f"Error: Dataset version '{ver_id}' not found in database.", err=True
                    )
                    await engine.dispose()
                    return 1

            # Fetch or create parent experiment
            exp = await exp_repo.create_experiment(
                name=config.metadata.name,
                dataset_version_id=ver.id,
                description=config.metadata.description,
                configuration=config.model_dump(),
                configuration_hash=config.compute_configuration_hash(),
            )
            await session.commit()
            exp_id = exp.id

            # Load corpus chunks from dataset documents
            docs = await dataset_repo.get_documents(ver.id)
            corpus_chunks: list[DocumentChunk] = []
            for d in docs:
                for c in d.chunks:
                    corpus_chunks.append(
                        DocumentChunk.create(
                            doc_id=d.id,
                            chunk_index=c.chunk_index,
                            content=c.content,
                            token_count=c.token_count,
                            start_char=0,
                            end_char=len(c.content),
                            strategy=c.strategy,
                            chunk_id=c.id,
                        )
                    )

            # Formulate sample queries if none in dataset metadata
            queries: list[EvaluationQuery] = []
            if ver.metadata_json and "queries" in ver.metadata_json:
                for q_data in ver.metadata_json["queries"]:
                    queries.append(EvaluationQuery(**q_data))
            elif corpus_chunks:
                # Default query against first available chunk
                queries.append(
                    EvaluationQuery(
                        query_id="q-default-1",
                        query_text=corpus_chunks[0].content[:60],
                        ground_truth_chunks=[corpus_chunks[0].chunk_id],
                    )
                )

        runner = MatrixRunner()
        matrix_result: MatrixRunResult = await runner.execute_matrix(
            experiment_id=exp_id,
            experiment_config=config,
            queries=queries,
            corpus_chunks=corpus_chunks,
            session_factory=session_factory,
            max_concurrency=concurrency,
            bypass_cache=bypass_cache,
        )

        # Output results
        if not quiet:
            headers = [
                "#",
                "Run ID",
                "Config Hash",
                "Status",
                "Cached",
                "Queries",
                "Mean Recall",
                "Duration (ms)",
            ]
            rows: list[list[Any]] = []
            for idx, r in enumerate(matrix_result.results, start=1):
                recall_val = r.mean_metrics.get("recall@1", r.mean_metrics.get("recall", 0.0))
                rows.append(
                    [
                        idx,
                        r.experiment_run_id[:8] if r.experiment_run_id else "N/A",
                        r.pipeline_config_hash[:10],
                        r.status,
                        "YES" if r.cached else "NO",
                        f"{r.completed_queries}/{r.total_queries}",
                        f"{recall_val:.4f}" if recall_val else "N/A",
                        f"{r.duration_ms:.1f}",
                    ]
                )

            click.echo(
                render_table(
                    headers, rows, title=f"Matrix Execution Results: {config.metadata.name}"
                )
            )

        click.echo(f"\nMatrix Status: {matrix_result.status}")
        click.echo(
            f"Total Configurations: {matrix_result.total_configurations} | "
            f"Executed: {matrix_result.executed_runs} | "
            f"Cached: {matrix_result.cached_runs} | "
            f"Failed: {matrix_result.failed_runs} | "
            f"Duration: {matrix_result.duration_ms:.1f}ms"
        )

        if output_json:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(matrix_result.model_dump(), f, indent=2)
            click.echo(f"Results exported to {output_json}")

        await engine.dispose()

        if matrix_result.status == "FAILED":
            return 1
        if matrix_result.status == "PARTIAL" and fail_on_partial:
            return 1
        return 0

    code = asyncio.run(_run())
    sys.exit(code)


@experiment_group.command(name="show")
@click.argument("run_id")
@click.option("--db-url", default=None, help="Database connection URL override.")
def show_command(run_id: str, db_url: str | None) -> None:
    """Inspect the full query evidence trail for an experiment run (RQ3 auditability)."""

    async def _show() -> int:
        engine = get_async_engine(db_url)
        session_factory = get_session_factory(engine)

        async with session_factory() as session:
            exp_repo = ExperimentRepository(session)
            trace_repo = QueryTraceRepository(session)

            run = await exp_repo.get_run(run_id)
            if not run:
                click.echo(f"ExperimentRun '{run_id}' not found.", err=True)
                await engine.dispose()
                return 1

            trails = await trace_repo.get_run_evidence_trails(run_id)

            click.echo(f"\nExperiment Run: {run.id}")
            click.echo(f"Status: {run.status} | Cache Hash: {run.cache_hash}")
            click.echo(f"Started: {run.started_at} | Completed: {run.completed_at}")

            # Summary Metrics Table
            if run.metric_summaries:
                headers = ["Metric", "Mean", "Median", "Min", "Max", "StdDev", "Count"]
                rows = [
                    [
                        s.metric_name,
                        f"{s.mean:.4f}",
                        f"{s.median:.4f}",
                        f"{s.min:.4f}",
                        f"{s.max:.4f}",
                        f"{s.stddev:.4f}",
                        s.count,
                    ]
                    for s in run.metric_summaries
                ]
                click.echo(render_table(headers, rows, title="Aggregate Metric Summaries"))

            # Evidence Trails Table
            click.echo(f"\nTotal Query Evidence Trails: {len(trails)}")
            for idx, trail in enumerate(trails, start=1):
                click.echo(f"\n[{idx}] Query ID: {trail.query_id} (Status: {trail.status})")
                click.echo(f"    Original Query: {trail.original_query}")
                click.echo(f"    Retrieved Chunks: {len(trail.retrieved_chunks)} chunks")
                if trail.generation_result:
                    ans_snippet = trail.generation_result.get("answer", "")[:80]
                    click.echo(f"    Generated Answer: {ans_snippet}...")
                if trail.metric_results:
                    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in trail.metric_results.items())
                    click.echo(f"    Metrics: {metrics_str}")

        await engine.dispose()
        return 0

    code = asyncio.run(_show())
    sys.exit(code)


@experiment_group.command(name="compare")
@click.argument("run_id_1")
@click.argument("run_id_2")
@click.option("--db-url", default=None, help="Database connection URL override.")
def compare_command(run_id_1: str, run_id_2: str, db_url: str | None) -> None:
    """Side-by-side comparative analysis of two experiment runs."""

    async def _compare() -> int:
        engine = get_async_engine(db_url)
        session_factory = get_session_factory(engine)

        async with session_factory() as session:
            exp_repo = ExperimentRepository(session)

            r1 = await exp_repo.get_run(run_id_1)
            r2 = await exp_repo.get_run(run_id_2)

            if not r1:
                click.echo(f"Run 1 '{run_id_1}' not found.", err=True)
                await engine.dispose()
                return 1
            if not r2:
                click.echo(f"Run 2 '{run_id_2}' not found.", err=True)
                await engine.dispose()
                return 1

            click.echo("\nComparing Runs:")
            click.echo(f"Run 1: {r1.id} (status: {r1.status})")
            click.echo(f"Run 2: {r2.id} (status: {r2.status})\n")

            metrics_1 = {s.metric_name: s.mean for s in r1.metric_summaries}
            metrics_2 = {s.metric_name: s.mean for s in r2.metric_summaries}
            all_metric_names = sorted(set(metrics_1.keys()) | set(metrics_2.keys()))

            headers = ["Metric", f"Run 1 ({r1.id[:8]})", f"Run 2 ({r2.id[:8]})", "Delta (Δ)"]
            rows: list[list[Any]] = []

            for m in all_metric_names:
                v1 = metrics_1.get(m)
                v2 = metrics_2.get(m)
                str_v1 = f"{v1:.4f}" if v1 is not None else "N/A"
                str_v2 = f"{v2:.4f}" if v2 is not None else "N/A"
                str_delta = format_delta(v1, v2) if (v1 is not None and v2 is not None) else "N/A"
                rows.append([m, str_v1, str_v2, str_delta])

            click.echo(render_table(headers, rows, title="Metric Comparison Table"))

        await engine.dispose()
        return 0

    code = asyncio.run(_compare())
    sys.exit(code)
