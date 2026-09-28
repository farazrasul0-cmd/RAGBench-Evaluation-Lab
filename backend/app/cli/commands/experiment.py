"""Experiment CLI commands: run, plan, show, and compare."""

import asyncio
import csv
import io
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
from app.engine.analysis.alignment import IncompatibleRunsError, QuerySetMismatchError
from app.engine.analysis.export import (
    AcademicLatexExporter,
    CSVExporter,
    ReplicationArchiveExporter,
    escape_latex,
)
from app.engine.analysis.statistics import StatisticalComparator
from app.engine.orchestrator.cartesian import CartesianExpander
from app.engine.orchestrator.dry_run import DryRunPlanner
from app.engine.runner.matrix import MatrixRunner
from app.engine.runner.models import EvaluationQuery, MatrixRunResult
from app.schemas.chunk import DocumentChunk
from app.schemas.experiment import ExperimentConfig, PipelineConfig


@click.group(name="experiment")
def experiment_group() -> None:
    """Manage, plan, execute, and analyze benchmark experiments."""


@experiment_group.command(name="plan")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--db-url", default=None, help="Database connection URL override.")
def plan_command(config_path: Path, db_url: str | None) -> None:
    """Dry-run preview of sweeps, diffs, and cache hits without DB writes."""
    try:
        config = ExperimentConfig.from_yaml(config_path)
    except Exception as exc:
        click.echo(f"Error loading configuration: {exc}", err=True)
        sys.exit(1)

    # 1. Purely in-memory dry-run plan generation
    planner = DryRunPlanner()
    plan_res = planner.plan(config)

    click.echo(f"\nExperiment Plan: {config.metadata.name}")
    click.echo(f"Total Combinatorial Configurations: {plan_res.total_planned_pipelines}\n")

    # 2. Check if DB is available for cache inspection WITHOUT mutating or initializing schema
    can_check_cache = True
    if db_url and db_url.startswith("sqlite"):
        db_path_str = db_url.split("///")[-1]
        if db_path_str != ":memory:" and not Path(db_path_str).exists():
            can_check_cache = False

    cached_hashes: set[str] = set()

    if can_check_cache:

        async def _check_cache() -> None:
            nonlocal cached_hashes
            try:
                engine = get_async_engine(db_url)
                session_factory = get_session_factory(engine)
                async with session_factory() as session:
                    exp_repo = ExperimentRepository(session)
                    for pipe in plan_res.planned_runs:
                        try:
                            cached_run = await exp_repo.get_run_by_cache_hash(pipe.cache_hash)
                            if cached_run is not None:
                                cached_hashes.add(pipe.cache_hash)
                        except Exception:
                            # DB exists but table might not exist yet
                            break
                await engine.dispose()
            except Exception:
                pass

        import contextlib

        with contextlib.suppress(Exception):
            asyncio.run(_check_cache())

    headers = ["#", "Config Hash", "Cache Hash", "Cache Status", "Diff vs Baseline"]
    rows: list[list[Any]] = []

    for idx, pipe in enumerate(plan_res.planned_runs, start=1):
        if not can_check_cache:
            cache_status = "UNKNOWN (no DB)"
        elif pipe.cache_hash in cached_hashes:
            cache_status = "HIT"
        else:
            cache_status = "MISS"

        diff_summary = "Baseline"
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
    sys.exit(0)


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

            # Strict dataset version resolution — zero fallback to preserve immutable lineage
            ver_id = config.dataset.dataset_version_id
            ver = await dataset_repo.get_version(ver_id)
            if not ver:
                click.echo(
                    f"Error: Dataset version '{ver_id}' not found in database. "
                    "Automatic fallback is disabled to preserve dataset-version immutability.",
                    err=True,
                )
                await engine.dispose()
                return 1

            # Validate dataset ID alignment if specified
            if config.dataset.dataset_id:
                ds = await dataset_repo.get_dataset(ver.dataset_id)
                if (
                    ds
                    and ds.id != config.dataset.dataset_id
                    and ds.name != config.dataset.dataset_id
                ):
                    click.echo(
                        f"Error: Dataset version '{ver_id}' belongs to '{ds.name}', "
                        f"not requested dataset '{config.dataset.dataset_id}'.",
                        err=True,
                    )
                    await engine.dispose()
                    return 1

            # Load benchmark evaluation queries from dataset version metadata — zero fabrication
            queries: list[EvaluationQuery] = []
            if (
                ver.metadata_json
                and "queries" in ver.metadata_json
                and ver.metadata_json["queries"]
            ):
                for q_data in ver.metadata_json["queries"]:
                    queries.append(EvaluationQuery(**q_data))
            else:
                click.echo(
                    f"Error: Dataset version '{ver.id}' contains no benchmark evaluation queries. "
                    "Ground truth queries must be provided during registration (--queries).",
                    err=True,
                )
                await engine.dispose()
                return 1

            # Fetch or create parent experiment record
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

        # Execute matrix sweep via F5 MatrixRunner
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
@click.option("--query-id", "-q", default=None, help="Filter for a specific query ID.")
@click.option(
    "--full", is_flag=True, default=False, help="Display full untruncated answers and chunk text."
)
@click.option("--db-url", default=None, help="Database connection URL override.")
def show_command(run_id: str, query_id: str | None, full: bool, db_url: str | None) -> None:
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
            click.echo(f"Status: {run.status} | Cache Hash: {run.cache_hash[:16]}...")
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

            # Filter trails if requested
            selected_trails = trails
            if query_id:
                selected_trails = [t for t in trails if t.query_id == query_id]
                if not selected_trails:
                    click.echo(f"Query ID '{query_id}' not found in run traces.")

            click.echo(f"\nQuery Evidence Trails ({len(selected_trails)} displayed):")
            click.echo("=" * 80)

            for idx, trail in enumerate(selected_trails, start=1):
                header = (
                    f"\n[{idx}] Query: {trail.query_id} ({trail.status}, {trail.latency_ms:.1f}ms)"
                )
                click.echo(header)
                click.echo(f"    Original Query: {trail.original_query}")

                # Transformed Queries
                if trail.transformed_queries:
                    t_desc = [
                        f"[{t.get('transformation_type', 'transform')}] {t.get('query_text', '')}"
                        for t in trail.transformed_queries
                    ]
                    click.echo(f"    Transformations: {'; '.join(t_desc)}")

                # Retrieved Chunks
                if trail.retrieved_chunks:
                    click.echo("    Retrieved Chunks:")
                    for rc in trail.retrieved_chunks:
                        r_type = rc.get("retriever_type", "retriever")
                        r_line = (
                            f"      Rank {rc.get('rank', 0)} | {rc.get('chunk_id', '')} | "
                            f"Score: {rc.get('score', 0.0):.4f} ({r_type})"
                        )
                        click.echo(r_line)

                # Reranked Chunks
                if trail.reranked_chunks:
                    click.echo("    Reranked Chunks:")
                    for rk in trail.reranked_chunks:
                        o_r = rk.get("original_rank", 0)
                        n_r = rk.get("reranked_rank", 0)
                        cid = rk.get("chunk_id", "")
                        s1 = f"{rk.get('original_score', 0.0):.4f}"
                        s2 = f"{rk.get('reranker_score', 0.0):.4f}"
                        rk_line = f"      Orig {o_r} -> Rerank {n_r} | {cid} | Scores: {s1} -> {s2}"
                        click.echo(rk_line)

                # Context Packing
                if trail.packed_context:
                    p = trail.packed_context
                    strat = p.get("ordering_strategy")
                    c_cnt = len(p.get("chunk_ids", []))
                    t_str = f"{p.get('token_count')}/{p.get('token_budget')}"
                    click.echo(f"    Context: strategy={strat} | chunks={c_cnt} | tokens={t_str}")

                # Generation
                if trail.generation_result:
                    g = trail.generation_result
                    ans = g.get("answer", "")
                    if not full and len(ans) > 150:
                        ans = ans[:150] + "..."
                    gen_hdr = (
                        f"    Generation: model={g.get('model')} | "
                        f"latency={g.get('latency_ms', 0.0):.1f}ms | tokens={g.get('total_tokens')}"
                    )
                    click.echo(gen_hdr)
                    click.echo(f"    Answer: {ans}")

                # Metrics
                if trail.metric_results:
                    m_parts = [f"{k}={v:.4f}" for k, v in trail.metric_results.items()]
                    click.echo(f"    Metrics: {', '.join(m_parts)}")

        await engine.dispose()
        return 0

    code = asyncio.run(_show())
    sys.exit(code)


@experiment_group.command(name="compare")
@click.argument("run_id_1")
@click.argument("run_id_2")
@click.option(
    "--stats",
    is_flag=True,
    default=False,
    help="Perform paired statistical hypothesis testing (t-test, Wilcoxon, Cohen's dz, 95% CI).",
)
@click.option(
    "--correction",
    type=click.Choice(["holm", "bh_fdr", "none"], case_sensitive=False),
    default="holm",
    help="Multiple testing correction method: holm (default), bh_fdr, or none.",
)
@click.option(
    "--warmup",
    default=0,
    type=int,
    help="Number of initial queries to drop from latency metrics for warm-up.",
)
@click.option(
    "--allow-partial-query-overlap",
    is_flag=True,
    default=False,
    help="Allow partial query set overlap between runs.",
)
@click.option("--db-url", default=None, help="Database connection URL override.")
def compare_command(
    run_id_1: str,
    run_id_2: str,
    stats: bool,
    correction: str,
    warmup: int,
    allow_partial_query_overlap: bool,
    db_url: str | None,
) -> None:
    """Side-by-side comparative analysis of pipeline parameters and metrics for two runs."""

    async def _compare() -> int:
        engine = get_async_engine(db_url)
        session_factory = get_session_factory(engine)

        async with session_factory() as session:
            exp_repo = ExperimentRepository(session)
            trace_repo = QueryTraceRepository(session)

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

            # 1. Pipeline Parameter Differences
            exp1 = await exp_repo.get_experiment(r1.experiment_id)
            exp2 = await exp_repo.get_experiment(r2.experiment_id)
            if exp1 is not None:
                r1.experiment = exp1
            if exp2 is not None:
                r2.experiment = exp2

            pipe_cfg_1: PipelineConfig | None = None
            pipe_cfg_2: PipelineConfig | None = None

            if exp1 and exp1.configuration:
                try:
                    c1 = ExperimentConfig(**exp1.configuration)
                    for p in CartesianExpander.expand(c1):
                        if p.compute_configuration_hash() == r1.pipeline_config_hash:
                            pipe_cfg_1 = p
                            break
                except Exception:
                    pass

            if exp2 and exp2.configuration:
                try:
                    c2 = ExperimentConfig(**exp2.configuration)
                    for p in CartesianExpander.expand(c2):
                        if p.compute_configuration_hash() == r2.pipeline_config_hash:
                            pipe_cfg_2 = p
                            break
                except Exception:
                    pass

            if pipe_cfg_1 and pipe_cfg_2:
                param_rows: list[list[Any]] = []
                d1 = pipe_cfg_1.model_dump()
                d2 = pipe_cfg_2.model_dump()

                for component in sorted(set(d1.keys()) | set(d2.keys())):
                    sub1 = d1.get(component, {})
                    sub2 = d2.get(component, {})
                    if isinstance(sub1, dict) and isinstance(sub2, dict):
                        for param in sorted(set(sub1.keys()) | set(sub2.keys())):
                            val1 = sub1.get(param)
                            val2 = sub2.get(param)
                            if val1 != val2:
                                param_rows.append([component, param, str(val1), str(val2)])

                if param_rows:
                    p_headers = [
                        "Component",
                        "Parameter",
                        f"Run 1 ({r1.id[:8]})",
                        f"Run 2 ({r2.id[:8]})",
                    ]
                    click.echo(
                        render_table(p_headers, param_rows, title="Pipeline Parameter Differences")
                    )
                else:
                    click.echo("Pipeline Configurations: Identical parameters.\n")

            # 2. Metric Summaries Comparison Table
            metrics_1 = {s.metric_name: s.mean for s in r1.metric_summaries}
            metrics_2 = {s.metric_name: s.mean for s in r2.metric_summaries}
            all_metric_names = sorted(set(metrics_1.keys()) | set(metrics_2.keys()))

            headers = [
                "Metric",
                f"Run 1 ({r1.id[:8]})",
                f"Run 2 ({r2.id[:8]})",
                "Delta (Run2 - Run1)",
            ]
            rows: list[list[Any]] = []

            for m in all_metric_names:
                v1 = metrics_1.get(m)
                v2 = metrics_2.get(m)
                str_v1 = f"{v1:.4f}" if v1 is not None else "N/A"
                str_v2 = f"{v2:.4f}" if v2 is not None else "N/A"
                str_delta = format_delta(v1, v2) if (v1 is not None and v2 is not None) else "N/A"
                rows.append([m, str_v1, str_v2, str_delta])

            click.echo(render_table(headers, rows, title="Metric Comparison Table"))

            # 3. Execution Latency Comparison
            dur_1 = (
                (r1.completed_at - r1.started_at).total_seconds() * 1000.0
                if (r1.completed_at and r1.started_at)
                else None
            )
            dur_2 = (
                (r2.completed_at - r2.started_at).total_seconds() * 1000.0
                if (r2.completed_at and r2.started_at)
                else None
            )
            if dur_1 is not None and dur_2 is not None:
                click.echo(f"Duration Run 1: {dur_1:.1f}ms | Duration Run 2: {dur_2:.1f}ms\n")

            # 4. Statistical Hypothesis Testing
            if stats:
                trails_1 = await trace_repo.get_run_evidence_trails(r1.id)
                trails_2 = await trace_repo.get_run_evidence_trails(r2.id)

                if not trails_1 or not trails_2:
                    click.echo(
                        "Warning: Query evidence trails are empty for one or both runs.", err=True
                    )
                else:
                    metrics_by_query_a: dict[str, dict[str, float]] = {}
                    for t in trails_1:
                        m_map = dict(t.metric_results)
                        m_map["latency_ms"] = t.latency_ms
                        metrics_by_query_a[t.query_id] = m_map

                    metrics_by_query_b: dict[str, dict[str, float]] = {}
                    for t in trails_2:
                        m_map = dict(t.metric_results)
                        m_map["latency_ms"] = t.latency_ms
                        metrics_by_query_b[t.query_id] = m_map

                    try:
                        report = StatisticalComparator.compare_runs(
                            run_a=r1,
                            run_b=r2,
                            metrics_by_query_a=metrics_by_query_a,
                            metrics_by_query_b=metrics_by_query_b,
                            correction_method=correction.lower(),
                            warmup_queries=warmup,
                            allow_partial_query_overlap=allow_partial_query_overlap,
                        )

                        stat_headers = [
                            "Metric",
                            f"Run 1 ({r1.id[:8]})",
                            f"Run 2 ({r2.id[:8]})",
                            "Mean Δ",
                            "95% CI (Δ)",
                            "t-stat",
                            "Wilcoxon W",
                            f"Adj. p ({report.adjustment_method})",
                            "Sig",
                            "Cohen's dz",
                        ]
                        stat_rows: list[list[Any]] = []

                        for m_name, res in sorted(report.metrics.items()):
                            mean_1 = f"{res.mean_a:.4f}"
                            mean_2 = f"{res.mean_b:.4f}"
                            mean_delta = f"{res.mean_difference:+.4f}"
                            ci_str = (
                                f"[{res.ci95_lower:+.4f}, {res.ci95_upper:+.4f}]"
                                if (res.ci95_lower is not None and res.ci95_upper is not None)
                                else "N/A"
                            )
                            t_str = (
                                f"{res.t_statistic:.3f}" if res.t_statistic is not None else "N/A"
                            )
                            w_str = (
                                f"{res.wilcoxon_statistic:.1f}"
                                if res.wilcoxon_statistic is not None
                                else "N/A"
                            )
                            p_str = (
                                f"{res.adjusted_p_value:.4f}"
                                if res.adjusted_p_value is not None
                                else "N/A"
                            )
                            sig_str = res.significance_level
                            dz_str = f"{res.cohens_dz:+.2f}" if res.cohens_dz is not None else "N/A"

                            stat_rows.append(
                                [
                                    m_name,
                                    mean_1,
                                    mean_2,
                                    mean_delta,
                                    ci_str,
                                    t_str,
                                    w_str,
                                    p_str,
                                    sig_str,
                                    dz_str,
                                ]
                            )

                        click.echo(
                            render_table(
                                stat_headers,
                                stat_rows,
                                title=(
                                    "Statistical Significance Analysis "
                                    f"({report.adjustment_method.upper()})"
                                ),
                            )
                        )
                        click.echo(
                            "Significance: * p < 0.05, ** p < 0.01, *** p < 0.001 "
                            "(Two-tailed paired t-test & Wilcoxon signed-rank test)\n"
                        )

                    except (IncompatibleRunsError, QuerySetMismatchError) as exc:
                        click.echo(f"Statistical Analysis Incompatible: {exc}", err=True)
                        await engine.dispose()
                        return 1

        await engine.dispose()
        return 0

    code = asyncio.run(_compare())
    sys.exit(code)


@experiment_group.command(name="export")
@click.argument("run_id_1")
@click.argument("run_id_2", required=False, default=None)
@click.option(
    "--format",
    "-f",
    "export_format",
    type=click.Choice(["latex", "csv", "replication"], case_sensitive=False),
    default="latex",
    help="Export format: publication latex table, csv summary, or replication bundle.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file or directory path.",
)
@click.option(
    "--correction",
    type=click.Choice(["holm", "bh_fdr", "none"], case_sensitive=False),
    default="holm",
    help="Multiple testing correction method: holm (default), bh_fdr, or none.",
)
@click.option(
    "--warmup",
    default=0,
    type=int,
    help="Number of initial queries to drop from latency metrics for warm-up.",
)
@click.option(
    "--allow-partial-query-overlap",
    is_flag=True,
    default=False,
    help="Allow partial query set overlap between runs.",
)
@click.option("--db-url", default=None, help="Database connection URL override.")
def export_command(
    run_id_1: str,
    run_id_2: str | None,
    export_format: str,
    output: Path | None,
    correction: str,
    warmup: int,
    allow_partial_query_overlap: bool,
    db_url: str | None,
) -> None:
    """Export academic LaTeX tables, CSV summaries, or replication archives for experiment runs."""

    async def _export() -> int:
        engine = get_async_engine(db_url)
        session_factory = get_session_factory(engine)

        async with session_factory() as session:
            exp_repo = ExperimentRepository(session)
            trace_repo = QueryTraceRepository(session)

            r1 = await exp_repo.get_run(run_id_1)
            if not r1:
                click.echo(f"Run 1 '{run_id_1}' not found.", err=True)
                await engine.dispose()
                return 1

            exp1 = await exp_repo.get_experiment(r1.experiment_id)
            if exp1 is not None:
                r1.experiment = exp1

            if run_id_2 is not None:
                r2 = await exp_repo.get_run(run_id_2)
                if not r2:
                    click.echo(f"Run 2 '{run_id_2}' not found.", err=True)
                    await engine.dispose()
                    return 1

                exp2 = await exp_repo.get_experiment(r2.experiment_id)
                if exp2 is not None:
                    r2.experiment = exp2

                trails_1 = await trace_repo.get_run_evidence_trails(r1.id)
                trails_2 = await trace_repo.get_run_evidence_trails(r2.id)

                metrics_by_query_a: dict[str, dict[str, float]] = {}
                for t in trails_1:
                    m_map = dict(t.metric_results)
                    m_map["latency_ms"] = t.latency_ms
                    metrics_by_query_a[t.query_id] = m_map

                metrics_by_query_b: dict[str, dict[str, float]] = {}
                for t in trails_2:
                    m_map = dict(t.metric_results)
                    m_map["latency_ms"] = t.latency_ms
                    metrics_by_query_b[t.query_id] = m_map

                try:
                    report = StatisticalComparator.compare_runs(
                        run_a=r1,
                        run_b=r2,
                        metrics_by_query_a=metrics_by_query_a,
                        metrics_by_query_b=metrics_by_query_b,
                        correction_method=correction.lower(),
                        warmup_queries=warmup,
                        allow_partial_query_overlap=allow_partial_query_overlap,
                    )
                except (IncompatibleRunsError, QuerySetMismatchError) as exc:
                    click.echo(f"Export Incompatible: {exc}", err=True)
                    await engine.dispose()
                    return 1

                # Calculate config diff if available
                config_diff: dict[str, Any] | None = None
                pipe_cfg_1: PipelineConfig | None = None
                pipe_cfg_2: PipelineConfig | None = None
                if exp1 and exp1.configuration:
                    try:
                        c1 = ExperimentConfig(**exp1.configuration)
                        for p in CartesianExpander.expand(c1):
                            if p.compute_configuration_hash() == r1.pipeline_config_hash:
                                pipe_cfg_1 = p
                                break
                    except Exception:
                        pass
                if exp2 and exp2.configuration:
                    try:
                        c2 = ExperimentConfig(**exp2.configuration)
                        for p in CartesianExpander.expand(c2):
                            if p.compute_configuration_hash() == r2.pipeline_config_hash:
                                pipe_cfg_2 = p
                                break
                    except Exception:
                        pass

                if pipe_cfg_1 and pipe_cfg_2:
                    d1 = pipe_cfg_1.model_dump()
                    d2 = pipe_cfg_2.model_dump()
                    diffs: dict[str, Any] = {}
                    for comp in sorted(set(d1.keys()) | set(d2.keys())):
                        sub1 = d1.get(comp, {})
                        sub2 = d2.get(comp, {})
                        if isinstance(sub1, dict) and isinstance(sub2, dict):
                            for param in sorted(set(sub1.keys()) | set(sub2.keys())):
                                if sub1.get(param) != sub2.get(param):
                                    diffs[f"{comp}.{param}"] = {
                                        "run_1": sub1.get(param),
                                        "run_2": sub2.get(param),
                                    }
                    if diffs:
                        config_diff = diffs

                fmt = export_format.lower()
                if fmt == "latex":
                    latex_str = AcademicLatexExporter.export_comparison_table(report)
                    if output:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_text(latex_str, encoding="utf-8")
                        click.echo(f"Exported LaTeX comparison table to: {output}")
                    else:
                        click.echo(latex_str)

                elif fmt == "csv":
                    csv_str = CSVExporter.export_summary_csv(report)
                    if output:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_text(csv_str, encoding="utf-8")
                        click.echo(f"Exported CSV summary to: {output}")
                    else:
                        click.echo(csv_str)

                elif fmt == "replication":
                    dest_dir = output or Path(f"replication_{r1.id[:8]}_{r2.id[:8]}")
                    ReplicationArchiveExporter.export_package(
                        report=report,
                        target_dir=dest_dir,
                        config_diff=config_diff,
                    )
                    click.echo(f"Exported complete replication archive package to: {dest_dir}")

            else:
                fmt = export_format.lower()
                if fmt == "latex":
                    lines = [
                        "% Auto-generated by RAGBench Evaluation Lab",
                        "\\begin{table}[t]",
                        "\\centering",
                        "\\small",
                        "\\begin{tabular}{lccccc}",
                        "\\toprule",
                        "\\textbf{Metric} & \\textbf{Mean} & \\textbf{Median} & "
                        "\\textbf{Min} & \\textbf{Max} & \\textbf{StdDev} \\\\",
                        "\\midrule",
                    ]
                    for s in r1.metric_summaries:
                        m_safe = escape_latex(s.metric_name)
                        line_tex = (
                            f"{m_safe} & {s.mean:.4f} & {s.median:.4f} & "
                            f"{s.min:.4f} & {s.max:.4f} & {s.stddev:.4f} \\\\"
                        )
                        lines.append(line_tex)
                    lines.extend(
                        [
                            "\\bottomrule",
                            "\\end{tabular}",
                            f"\\caption{{Metric Summaries for Run {r1.id[:8]}}}",
                            f"\\label{{tab:ragbench_run_{r1.id[:8]}}}",
                            "\\end{table}",
                        ]
                    )
                    latex_str = "\n".join(lines)
                    if output:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_text(latex_str, encoding="utf-8")
                        click.echo(f"Exported single-run LaTeX table to: {output}")
                    else:
                        click.echo(latex_str)

                elif fmt == "csv":
                    out_io = io.StringIO()
                    writer = csv.writer(out_io)
                    writer.writerow(["metric", "mean", "median", "min", "max", "stddev", "count"])
                    for s in r1.metric_summaries:
                        writer.writerow(
                            [
                                s.metric_name,
                                f"{s.mean:.6f}",
                                f"{s.median:.6f}",
                                f"{s.min:.6f}",
                                f"{s.max:.6f}",
                                f"{s.stddev:.6f}",
                                s.count,
                            ]
                        )
                    csv_str = out_io.getvalue()
                    if output:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_text(csv_str, encoding="utf-8")
                        click.echo(f"Exported single-run CSV summary to: {output}")
                    else:
                        click.echo(csv_str)

                elif fmt == "replication":
                    dest_dir = output or Path(f"replication_{r1.id[:8]}")
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    meta_path = dest_dir / "run_summary.json"
                    meta_path.write_text(
                        json.dumps(
                            {
                                "run_id": r1.id,
                                "status": r1.status,
                                "pipeline_config_hash": r1.pipeline_config_hash,
                                "cache_hash": r1.cache_hash,
                                "metrics": {s.metric_name: s.mean for s in r1.metric_summaries},
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    click.echo(f"Exported single-run archive to: {dest_dir}")

        await engine.dispose()
        return 0

    code = asyncio.run(_export())
    sys.exit(code)
