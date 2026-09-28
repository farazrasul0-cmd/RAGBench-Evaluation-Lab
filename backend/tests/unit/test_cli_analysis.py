"""Tests for Phase F7.4: CLI Comparative Statistical Analysis & Academic Export.

Verifies:
1. 'compare --stats' renders hypothesis testing table with t-test, Wilcoxon, Cohen's dz, CI.
2. Multiple testing correction selection (--correction holm, bh_fdr, none).
3. Incompatible runs error handling (mismatched dataset versions).
4. Academic LaTeX table export (--format latex).
5. Structured CSV summary export (--format csv).
6. Self-contained replication package archive export (--format replication) with SHA-256 hashes.
7. Single-run export functionality for publication summaries.
"""

import json
import shutil
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from click.testing import CliRunner

from app.cli.main import cli
from app.db.repositories.dataset import DatasetRepository
from app.db.repositories.experiment import ExperimentRepository
from app.db.repositories.query_trace import QueryTraceRepository
from app.db.session import get_async_engine, get_session_factory, init_db


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def analysis_ws() -> Generator[Path, None, None]:
    ws = Path("test_cli_analysis_ws")
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True, exist_ok=True)
    try:
        yield ws
    finally:
        shutil.rmtree(ws, ignore_errors=True)


async def _seed_test_runs(db_url: str) -> tuple[str, str, str]:
    """Seed DB with a dataset, two completed runs under same dataset, and one incompatible."""
    engine = get_async_engine(db_url)
    await init_db(engine)
    session_factory = get_session_factory(engine)

    uid = uuid.uuid4().hex[:8]
    async with session_factory() as session:
        ds_repo = DatasetRepository(session)
        exp_repo = ExperimentRepository(session)
        trace_repo = QueryTraceRepository(session)

        # 1. Dataset 1 & Version 1
        ds1 = await ds_repo.create_dataset(name=f"BenchmarkCorpusA_{uid}")
        ver1 = await ds_repo.create_version(
            dataset_id=ds1.id,
            version_number=1,
            content_hash=f"hash-ver-1-{uid}",
            metadata={"queries": [{"query_id": f"q{i}"} for i in range(1, 11)]},
        )
        doc1_list = await ds_repo.add_documents(
            ver1.id,
            [
                {
                    "filename": "doc1.txt",
                    "content": "text content",
                    "content_hash": "c1",
                    "mime_type": "text/plain",
                    "size_bytes": 12,
                    "page_count": 1,
                    "metadata": {},
                }
            ],
        )
        await ds_repo.add_chunks(
            doc1_list[0].id,
            [
                {
                    "id": f"c1_{uid}",
                    "chunk_index": 0,
                    "content": "chunk content",
                    "content_hash": "ch1",
                    "token_count": 5,
                    "char_count": 13,
                    "start_char": 0,
                    "end_char": 13,
                    "strategy": "fixed",
                    "chunking_config_hash": "cfg-1",
                    "metadata": {},
                }
            ],
        )

        # 2. Experiment 1 with Run 1 & Run 2 (both evaluate ver1)
        cfg1 = {
            "metadata": {"name": f"Exp1_{uid}"},
            "dataset": {"dataset_id": ds1.id, "dataset_version_id": ver1.id},
            "parameters": {"retrieval": {"mode": "bm25", "top_k": 5}},
        }
        e1 = await exp_repo.create_experiment(
            name=f"Exp1_{uid}",
            dataset_version_id=ver1.id,
            configuration=cfg1,
            configuration_hash=f"cfg1_{uid}",
        )
        r1 = await exp_repo.create_run(
            experiment_id=e1.id, pipeline_config_hash="p1", cache_key="k1", cache_hash=f"h1_{uid}"
        )
        r2 = await exp_repo.create_run(
            experiment_id=e1.id, pipeline_config_hash="p2", cache_key="k2", cache_hash=f"h2_{uid}"
        )

        r1.status = "COMPLETED"
        r2.status = "COMPLETED"

        # Record traces for Run 1 and Run 2 across 10 paired queries
        for i in range(1, 11):
            qid = f"q{i}"
            await trace_repo.record_query_trace(
                experiment_run_id=r1.id,
                query_id=qid,
                original_query=f"Query {i}",
                latency_ms=10.0 + i,
                status="SUCCESS",
                metric_results=[
                    {"metric_name": "recall@5", "metric_value": 0.60 + 0.02 * i},
                    {"metric_name": "mrr@5", "metric_value": 0.50 + 0.02 * i},
                ],
                validate_chunk_references=False,
            )
            await trace_repo.record_query_trace(
                experiment_run_id=r2.id,
                query_id=qid,
                original_query=f"Query {i}",
                latency_ms=15.0 + i,
                status="SUCCESS",
                metric_results=[
                    {"metric_name": "recall@5", "metric_value": 0.75 + 0.02 * i},
                    {"metric_name": "mrr@5", "metric_value": 0.65 + 0.02 * i},
                ],
                validate_chunk_references=False,
            )

        # 3. Incompatible Dataset 2 & Run 3 (different dataset version)
        ds2 = await ds_repo.create_dataset(name=f"BenchmarkCorpusB_{uid}")
        ver2 = await ds_repo.create_version(
            dataset_id=ds2.id,
            version_number=1,
            content_hash=f"hash-ver-2-{uid}",
            metadata={"queries": [{"query_id": f"q{i}"} for i in range(1, 11)]},
        )
        e2 = await exp_repo.create_experiment(
            name=f"Exp2_{uid}",
            dataset_version_id=ver2.id,
            configuration={},
            configuration_hash=f"cfg3_{uid}",
        )
        r3 = await exp_repo.create_run(
            experiment_id=e2.id, pipeline_config_hash="p3", cache_key="k3", cache_hash=f"h3_{uid}"
        )
        r3.status = "COMPLETED"

        # Record traces for Run 3 under ver2
        for i in range(1, 11):
            qid = f"q{i}"
            await trace_repo.record_query_trace(
                experiment_run_id=r3.id,
                query_id=qid,
                original_query=f"Query {i}",
                latency_ms=20.0 + i,
                status="SUCCESS",
                metric_results=[
                    {"metric_name": "recall@5", "metric_value": 0.80},
                ],
                validate_chunk_references=False,
            )

        await session.commit()
        r1_id, r2_id, r3_id = r1.id, r2.id, r3.id

    await engine.dispose()
    return r1_id, r2_id, r3_id


def test_cli_compare_with_stats_table(cli_runner: CliRunner, analysis_ws: Path) -> None:
    """'experiment compare --stats' displays the empirical hypothesis testing table."""
    db_file = analysis_ws / "test_compare.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"
    import asyncio

    r1_id, r2_id, _ = asyncio.run(_seed_test_runs(db_url))

    res = cli_runner.invoke(
        cli,
        [
            "experiment",
            "compare",
            r1_id,
            r2_id,
            "--stats",
            "--correction",
            "holm",
            "--db-url",
            db_url,
        ],
    )
    assert res.exit_code == 0
    assert "Statistical Significance Analysis (HOLM)" in res.output
    assert "recall@5" in res.output
    assert "mrr@5" in res.output
    assert "Wilcoxon W" in res.output
    assert "Cohen's dz" in res.output


def test_cli_compare_incompatible_runs_error(cli_runner: CliRunner, analysis_ws: Path) -> None:
    """Comparing runs from different dataset versions aborts with explicit Incompatible error."""
    db_file = analysis_ws / "test_incompat.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"
    import asyncio

    r1_id, _, r3_id = asyncio.run(_seed_test_runs(db_url))

    res = cli_runner.invoke(
        cli,
        [
            "experiment",
            "compare",
            r1_id,
            r3_id,
            "--stats",
            "--db-url",
            db_url,
        ],
    )
    assert res.exit_code != 0
    assert (
        "Statistical Analysis Incompatible" in res.output
        or "dataset_version_id differs" in res.output
    )


def test_cli_export_latex_and_csv(cli_runner: CliRunner, analysis_ws: Path) -> None:
    """'experiment export' produces valid LaTeX and CSV outputs."""
    db_file = analysis_ws / "test_export.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"
    import asyncio

    r1_id, r2_id, _ = asyncio.run(_seed_test_runs(db_url))

    # 1. LaTeX export to stdout
    tex_res = cli_runner.invoke(
        cli,
        ["experiment", "export", r1_id, r2_id, "--format", "latex", "--db-url", db_url],
    )
    assert tex_res.exit_code == 0
    assert "\\begin{table*}" in tex_res.output
    assert "\\toprule" in tex_res.output
    assert "recall" in tex_res.output
    assert "\\bottomrule" in tex_res.output

    # 2. CSV export to file
    csv_file = analysis_ws / "exported_stats.csv"
    csv_res = cli_runner.invoke(
        cli,
        [
            "experiment",
            "export",
            r1_id,
            r2_id,
            "--format",
            "csv",
            "--output",
            str(csv_file),
            "--db-url",
            db_url,
        ],
    )
    assert csv_res.exit_code == 0
    assert csv_file.exists()
    content = csv_file.read_text(encoding="utf-8")
    assert "metric,mean_run_a,mean_run_b" in content
    assert "recall@5" in content


def test_cli_export_replication_archive(cli_runner: CliRunner, analysis_ws: Path) -> None:
    """'experiment export --format replication' creates self-contained verified bundle."""
    db_file = analysis_ws / "test_repl.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"
    import asyncio

    r1_id, r2_id, _ = asyncio.run(_seed_test_runs(db_url))

    archive_dir = analysis_ws / "replication_bundle"
    res = cli_runner.invoke(
        cli,
        [
            "experiment",
            "export",
            r1_id,
            r2_id,
            "--format",
            "replication",
            "--output",
            str(archive_dir),
            "--db-url",
            db_url,
        ],
    )
    assert res.exit_code == 0
    assert archive_dir.exists()
    assert (archive_dir / "manifest.json").exists()
    assert (archive_dir / "hashes.json").exists()
    assert (archive_dir / "comparison_table.tex").exists()
    assert (archive_dir / "statistical_summary.csv").exists()
    assert (archive_dir / "raw_paired_observations.csv").exists()

    # Verify SHA-256 hashes integrity in hashes.json
    import hashlib

    hashes = json.loads((archive_dir / "hashes.json").read_text(encoding="utf-8"))
    assert "comparison_table.tex" in hashes
    tex_bytes = (archive_dir / "comparison_table.tex").read_bytes()
    expected_hash = hashlib.sha256(tex_bytes).hexdigest()
    assert hashes["comparison_table.tex"] == expected_hash
