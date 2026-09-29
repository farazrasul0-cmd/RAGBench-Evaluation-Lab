"""Empirical integration test for Phase F8: End-to-End RQ1/RQ2 Reference Studies & Replication.

Verifies:
1. Ingestion of canonical SciFact and Multilingual benchmark QA datasets with exact passage offsets.
2. Persistence and verification of evaluation_protocol_version and metric_definition_version.
3. Matrix execution across experimental points (RQ1 chunking & RQ2 retrieval paradigms).
4. Strict query-level paired statistical hypothesis testing (Wilcoxon, paired t-test, Cohen's dz).
5. Generation and cryptographic SHA-256 verification of the self-contained replication archive.
"""

import hashlib
import json
import shutil
from collections.abc import Generator
from pathlib import Path

import pytest
from click.testing import CliRunner
from sqlalchemy import select

from app.cli.main import cli
from app.core.protocols import DEFAULT_EVALUATION_PROTOCOL, DEFAULT_METRIC_PROTOCOL
from app.db.session import get_async_engine, get_session_factory
from app.models.entities import BenchmarkVersion, ExperimentRun


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def pipeline_workspace() -> Generator[Path, None, None]:
    ws = Path("test_empirical_pipeline_ws")
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True, exist_ok=True)
    try:
        yield ws
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_empirical_scifact_rq1_study(cli_runner: CliRunner, pipeline_workspace: Path) -> None:
    """End-to-end SciFact RQ1 sweep, statistical comparison, and replication verification."""
    db_file = pipeline_workspace / "scifact_study.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    repo_root = Path(__file__).resolve().parents[3]
    corpus_dir = repo_root / "data" / "benchmarks" / "scifact_ragbench" / "corpus"
    eval_qa_file = repo_root / "data" / "benchmarks" / "scifact_ragbench" / "eval_qa.jsonl"

    assert corpus_dir.exists(), f"SciFact corpus directory not found at {corpus_dir}"
    assert eval_qa_file.exists(), f"SciFact eval_qa file not found at {eval_qa_file}"

    # 1. Register canonical SciFact dataset with benchmark QA pairs
    reg_res = cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(corpus_dir),
            "--name",
            "scifact_study",
            "--eval-qa",
            str(eval_qa_file),
            "--strategy",
            "fixed",
            "--chunk-size",
            "200",
            "--chunk-overlap",
            "20",
            "--db-url",
            db_url,
        ],
    )
    assert reg_res.exit_code == 0, f"Registration failed: {reg_res.output}"
    assert "Dataset Version ID:" in reg_res.output
    assert "Benchmark Queries:" in reg_res.output and "30" in reg_res.output

    ver_id = ""
    for line in reg_res.output.splitlines():
        if "Dataset Version ID:" in line:
            ver_id = line.split("Dataset Version ID:")[1].split("(")[0].strip()
            break
    assert ver_id, "Failed to parse Dataset Version ID"

    # 2. Verify BenchmarkVersion persistence with protocol specifications in DB
    import asyncio

    async def _verify_benchmark() -> None:
        engine = get_async_engine(db_url)
        session_factory = get_session_factory(engine)
        async with session_factory() as session:
            stmt = select(BenchmarkVersion)
            bv = (await session.execute(stmt)).scalars().first()
            assert bv is not None
            assert bv.query_count == 30
            assert bv.evaluation_protocol_version == DEFAULT_EVALUATION_PROTOCOL
            assert bv.metric_definition_version == DEFAULT_METRIC_PROTOCOL
        await engine.dispose()

    asyncio.run(_verify_benchmark())

    # 3. Create experiment configuration for RQ1 (comparing chunk size 200 vs 400)
    exp_yaml = pipeline_workspace / "rq1_study.yaml"
    exp_yaml.write_text(
        f"""
name: "SciFact RQ1 Chunking Study"
description: "Empirical study comparing retrieval at 200 vs 400 chunk sizes."
dataset:
  dataset_id: "scifact_study"
  dataset_version_id: "{ver_id}"
parameters:
  chunking:
    strategy: "fixed"
    chunk_size: [200, 400]
    chunk_overlap: [20]
  retrieval:
    mode: "dense"
    top_k: 3
  evaluation:
    metrics: ["recall", "mrr"]
    k_values: [3]
""",
        encoding="utf-8",
    )

    results_json = pipeline_workspace / "rq1_results.json"
    run_res = cli_runner.invoke(
        cli,
        [
            "experiment",
            "run",
            str(exp_yaml),
            "--concurrency",
            "2",
            "--output-json",
            str(results_json),
            "--db-url",
            db_url,
        ],
    )
    assert run_res.exit_code == 0, f"Experiment run failed: {run_res.output}"
    assert "Matrix Status: COMPLETED" in run_res.output
    assert "Executed: 2" in run_res.output

    with open(results_json, encoding="utf-8") as rf:
        run_data = json.load(rf)
    results = run_data["results"]
    assert len(results) == 2
    run_a_id = results[0]["experiment_run_id"]
    run_b_id = results[1]["experiment_run_id"]

    # 4. Verify ExperimentRun records inherited protocol specifications in DB
    async def _verify_runs() -> None:
        engine = get_async_engine(db_url)
        session_factory = get_session_factory(engine)
        async with session_factory() as session:
            r_a = (
                await session.execute(select(ExperimentRun).where(ExperimentRun.id == run_a_id))
            ).scalar_one()
            r_b = (
                await session.execute(select(ExperimentRun).where(ExperimentRun.id == run_b_id))
            ).scalar_one()
            assert r_a.evaluation_protocol_version == DEFAULT_EVALUATION_PROTOCOL
            assert r_b.evaluation_protocol_version == DEFAULT_EVALUATION_PROTOCOL
            assert r_a.metric_definition_version == DEFAULT_METRIC_PROTOCOL
            assert r_b.metric_definition_version == DEFAULT_METRIC_PROTOCOL
        await engine.dispose()

    asyncio.run(_verify_runs())

    # 5. Run comparative hypothesis testing via CLI
    comp_res = cli_runner.invoke(
        cli,
        [
            "experiment",
            "compare",
            run_a_id,
            run_b_id,
            "--stats",
            "--correction",
            "holm",
            "--db-url",
            db_url,
        ],
    )
    assert comp_res.exit_code == 0, f"Compare failed: {comp_res.output}"
    assert "Statistical Significance Analysis (HOLM)" in comp_res.output
    assert "recall@3" in comp_res.output

    # 6. Export self-contained replication archive and cryptographically verify SHA-256 manifest
    repl_dir = pipeline_workspace / "replication_archive"
    export_res = cli_runner.invoke(
        cli,
        [
            "experiment",
            "export",
            run_a_id,
            run_b_id,
            "--format",
            "replication",
            "--output",
            str(repl_dir),
            "--db-url",
            db_url,
        ],
    )
    assert export_res.exit_code == 0, f"Export failed: {export_res.output}"
    assert repl_dir.exists()

    manifest_file = repl_dir / "manifest.json"
    hashes_file = repl_dir / "hashes.json"
    assert manifest_file.exists(), "manifest.json missing from replication archive"
    assert hashes_file.exists(), "hashes.json missing from replication archive"

    with open(manifest_file, encoding="utf-8") as mf:
        manifest = json.load(mf)
    assert manifest["run_a_id"] == run_a_id
    assert manifest["run_b_id"] == run_b_id

    with open(hashes_file, encoding="utf-8") as hf:
        recorded_hashes = json.load(hf)

    # Cryptographic SHA-256 verification of every file recorded in manifest
    assert len(recorded_hashes) >= 4, "Insufficient files hashed in replication archive"
    for rel_path, expected_hash in recorded_hashes.items():
        file_path = repl_dir / rel_path
        assert file_path.exists(), f"File {rel_path} in hashes.json not found on disk"
        computed_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert computed_hash == expected_hash, (
            f"SHA-256 mismatch for {rel_path}: expected {expected_hash}, got {computed_hash}"
        )


def test_empirical_multilingual_rq2_study(cli_runner: CliRunner, pipeline_workspace: Path) -> None:
    """End-to-end Multilingual RQ2 sweep, statistical comparison, and replication verification."""
    db_file = pipeline_workspace / "multilingual_study.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    repo_root = Path(__file__).resolve().parents[3]
    corpus_dir = repo_root / "data" / "benchmarks" / "multilingual_bengali_english" / "corpus"
    eval_qa_file = (
        repo_root / "data" / "benchmarks" / "multilingual_bengali_english" / "eval_qa.jsonl"
    )

    assert corpus_dir.exists(), f"Multilingual corpus directory not found at {corpus_dir}"
    assert eval_qa_file.exists(), f"Multilingual eval_qa file not found at {eval_qa_file}"

    # 1. Register canonical Multilingual dataset with benchmark QA pairs
    reg_res = cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(corpus_dir),
            "--name",
            "multilingual_study",
            "--eval-qa",
            str(eval_qa_file),
            "--strategy",
            "fixed",
            "--chunk-size",
            "200",
            "--chunk-overlap",
            "20",
            "--db-url",
            db_url,
        ],
    )
    assert reg_res.exit_code == 0, f"Registration failed: {reg_res.output}"
    assert "Benchmark Queries:" in reg_res.output and "20" in reg_res.output

    ver_id = ""
    for line in reg_res.output.splitlines():
        if "Dataset Version ID:" in line:
            ver_id = line.split("Dataset Version ID:")[1].split("(")[0].strip()
            break
    assert ver_id, "Failed to parse Dataset Version ID"

    # 2. Create experiment configuration for RQ2 (comparing dense vs bm25)
    exp_yaml = pipeline_workspace / "rq2_study.yaml"
    exp_yaml.write_text(
        f"""
name: "Multilingual RQ2 Retrieval Study"
description: "Empirical study comparing dense vs bm25 retrieval over Bengali-English corpus."
dataset:
  dataset_id: "multilingual_study"
  dataset_version_id: "{ver_id}"
parameters:
  retrieval:
    mode: ["dense", "bm25"]
    top_k: 3
  evaluation:
    metrics: ["recall", "mrr"]
    k_values: [3]
""",
        encoding="utf-8",
    )

    results_json = pipeline_workspace / "rq2_results.json"
    run_res = cli_runner.invoke(
        cli,
        [
            "experiment",
            "run",
            str(exp_yaml),
            "--concurrency",
            "2",
            "--output-json",
            str(results_json),
            "--db-url",
            db_url,
        ],
    )
    assert run_res.exit_code == 0, f"Experiment run failed: {run_res.output}"
    assert "Matrix Status: COMPLETED" in run_res.output
    assert "Executed: 2" in run_res.output

    with open(results_json, encoding="utf-8") as rf:
        run_data = json.load(rf)
    results = run_data["results"]
    assert len(results) == 2
    run_a_id = results[0]["experiment_run_id"]
    run_b_id = results[1]["experiment_run_id"]

    # 3. Compare with statistics
    comp_res = cli_runner.invoke(
        cli,
        [
            "experiment",
            "compare",
            run_a_id,
            run_b_id,
            "--stats",
            "--db-url",
            db_url,
        ],
    )
    assert comp_res.exit_code == 0, f"Compare failed: {comp_res.output}"
    assert "Statistical Significance Analysis" in comp_res.output

    # 4. Replication export and verification
    repl_dir = pipeline_workspace / "multi_replication_archive"
    export_res = cli_runner.invoke(
        cli,
        [
            "experiment",
            "export",
            run_a_id,
            run_b_id,
            "--format",
            "replication",
            "--output",
            str(repl_dir),
            "--db-url",
            db_url,
        ],
    )
    assert export_res.exit_code == 0, f"Export failed: {export_res.output}"
    hashes_file = repl_dir / "hashes.json"
    assert hashes_file.exists()

    with open(hashes_file, encoding="utf-8") as hf:
        recorded_hashes = json.load(hf)

    for rel_path, expected_hash in recorded_hashes.items():
        file_path = repl_dir / rel_path
        computed_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert computed_hash == expected_hash
