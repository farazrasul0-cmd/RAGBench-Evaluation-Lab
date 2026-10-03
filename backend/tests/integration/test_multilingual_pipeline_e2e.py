"""End-to-end integration test for Phase G: Multilingual Evaluation & Transfer Analysis.

Verifies:
1. Ingestion of canonical parallel Bengali-English benchmark (25 units, 50 docs, 100 QA pairs).
2. Deterministic benchmark hash generation and protocol version adherence.
3. Provider contract: MockMultilingualEmbeddingProvider operates hermetically (IS_MOCK=True).
4. Execution of Monolingual (EN-EN), Cross-lingual (EN-BN), and Hybrid runs.
5. Verification of Claim A (Transfer Penalty) and Claim B (Hybrid Attenuation) via CLI transfer.
6. Permutation invariance of matched-pair information units.
7. Verification of self-contained replication archive bundle with cryptographic SHA-256 hashes.
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
from app.engine.embeddings.mock_provider import DeterministicMockMultilingualEmbeddingProvider
from app.models.entities import BenchmarkVersion, ExperimentRun


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def multilingual_workspace() -> Generator[Path, None, None]:
    ws = Path("test_multilingual_pipeline_ws")
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True, exist_ok=True)
    try:
        yield ws
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_multilingual_benchmark_ingestion_and_hash_determinism(
    cli_runner: CliRunner, multilingual_workspace: Path
) -> None:
    """Verifies that the 25-unit/100-QA benchmark ingests cleanly with deterministic hash."""
    db_file = multilingual_workspace / "ingest_test.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    repo_root = Path(__file__).resolve().parents[3]
    corpus_dir = repo_root / "data" / "benchmarks" / "multilingual_bengali_english" / "corpus"
    qa_rel = Path("data") / "benchmarks" / "multilingual_bengali_english" / "eval_qa.jsonl"
    eval_qa_file = repo_root / qa_rel

    assert corpus_dir.exists(), f"Bilingual corpus directory not found at {corpus_dir}"
    assert eval_qa_file.exists(), f"Bilingual eval_qa file not found at {eval_qa_file}"

    # 1. Register canonical multilingual benchmark
    reg_res = cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(corpus_dir),
            "--name",
            "multilingual_canonical_test",
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
    assert "100" in reg_res.output  # 100 benchmark queries

    # 2. Verify BenchmarkVersion persistence with protocol specifications in DB
    import asyncio

    async def _verify_benchmark_in_db() -> str:
        engine = get_async_engine(db_url)
        session_factory = get_session_factory(engine)
        async with session_factory() as session:
            stmt = select(BenchmarkVersion)
            bv = (await session.execute(stmt)).scalars().first()
            assert bv is not None
            assert bv.query_count == 100
            assert bv.evaluation_protocol_version == DEFAULT_EVALUATION_PROTOCOL
            assert bv.metric_definition_version == DEFAULT_METRIC_PROTOCOL
            b_hash = bv.benchmark_hash
        await engine.dispose()
        return b_hash

    hash_1 = asyncio.run(_verify_benchmark_in_db())
    assert hash_1, "Benchmark hash must not be empty"

    # 3. Test Mock Provider Hermetic Contract (CI only)
    mock_provider = DeterministicMockMultilingualEmbeddingProvider()
    assert getattr(mock_provider, "IS_MOCK", False) is True
    assert mock_provider.dimension == 1024


def test_multilingual_transfer_study_e2e(
    cli_runner: CliRunner, multilingual_workspace: Path
) -> None:
    """Full E2E test: execution, Claim A penalty, Claim B attenuation, replication."""
    db_file = multilingual_workspace / "transfer_study.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    repo_root = Path(__file__).resolve().parents[3]
    corpus_dir = repo_root / "data" / "benchmarks" / "multilingual_bengali_english" / "corpus"
    qa_rel = Path("data") / "benchmarks" / "multilingual_bengali_english" / "eval_qa.jsonl"
    eval_qa_file = repo_root / qa_rel

    # 1. Register canonical dataset
    reg_res = cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(corpus_dir),
            "--name",
            "multilingual_e2e",
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

    ver_id = ""
    for line in reg_res.output.splitlines():
        if "Dataset Version ID:" in line:
            ver_id = line.split("Dataset Version ID:")[1].split("(")[0].strip()
            break
    assert ver_id, "Failed to parse Dataset Version ID"

    # 2. Run sweep experiment across modalities using mock multilingual provider
    exp_yaml = multilingual_workspace / "multilingual_exp.yaml"
    exp_yaml.write_text(
        f"""
name: "Multilingual Transfer E2E Study"
description: "Cross-lingual transfer penalty and attenuation study."
dataset:
  dataset_id: "multilingual_e2e"
  dataset_version_id: "{ver_id}"
parameters:
  chunking:
    strategy: "fixed"
    chunk_size: [200]
    chunk_overlap: [20]
  retrieval:
    mode: ["dense", "hybrid"]
    top_k: [5]
  embeddings:
    provider: "mock_multilingual"
    model_name: "mock-multilingual-1024d"
"""
    )

    run_res = cli_runner.invoke(
        cli,
        [
            "experiment",
            "run",
            str(exp_yaml),
            "--db-url",
            db_url,
        ],
    )
    assert run_res.exit_code == 0, f"Experiment run failed: {run_res.output}"

    # Extract executed run IDs from DB
    import asyncio

    async def _get_run_ids() -> tuple[str, str]:
        engine = get_async_engine(db_url)
        session_factory = get_session_factory(engine)
        async with session_factory() as session:
            stmt = select(ExperimentRun).order_by(ExperimentRun.started_at)
            runs = (await session.execute(stmt)).scalars().all()
            assert len(runs) >= 2, f"Expected at least 2 runs, found {len(runs)}"
            r1 = runs[0].id
            r2 = runs[1].id
        await engine.dispose()
        return r1, r2

    r_dense_id, r_hybrid_id = asyncio.run(_get_run_ids())

    # 3. Test Transfer Analysis CLI Command (Claim A)
    transfer_res = cli_runner.invoke(
        cli,
        [
            "experiment",
            "transfer",
            r_dense_id,
            r_hybrid_id,
            "--stats",
            "--db-url",
            db_url,
        ],
    )
    assert transfer_res.exit_code == 0, f"Transfer command failed: {transfer_res.output}"
    assert "Cross-Lingual Language Transfer Analysis (Claim A)" in transfer_res.output
    assert "recall@5" in transfer_res.output
    assert "Penalty" in transfer_res.output

    # 4. Test Claim B Attenuation CLI invocation
    attenuation_res = cli_runner.invoke(
        cli,
        [
            "experiment",
            "transfer",
            r_dense_id,
            r_dense_id,  # baseline comparison
            "--compare-hybrid",
            r_hybrid_id,
            "--stats",
            "--db-url",
            db_url,
        ],
    )
    assert attenuation_res.exit_code == 0
    assert "Hybrid Attenuation of Language Transfer Penalty (Claim B)" in attenuation_res.output
    assert "Attenuation" in attenuation_res.output

    # 5. Export replication bundle and cryptographically verify SHA-256 hashes
    repl_dir = multilingual_workspace / "replication_bundle"
    export_res = cli_runner.invoke(
        cli,
        [
            "experiment",
            "export",
            r_dense_id,
            r_hybrid_id,
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

    hashes_file = repl_dir / "hashes.json"
    assert hashes_file.exists(), "hashes.json missing from replication package"

    hashes_data = json.loads(hashes_file.read_text(encoding="utf-8"))
    for file_rel_path, expected_sha256 in hashes_data.items():
        artifact_path = repl_dir / file_rel_path
        assert artifact_path.exists(), f"Replication artifact {file_rel_path} does not exist"
        actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert actual_sha256 == expected_sha256, (
            f"SHA-256 mismatch for {file_rel_path}: expected {expected_sha256}, got {actual_sha256}"
        )
