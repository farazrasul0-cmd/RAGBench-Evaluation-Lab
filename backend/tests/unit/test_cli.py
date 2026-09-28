"""Unit and integration tests for Phase F6: CLI and Local Research Workflow.

Verifies:
1. Root CLI command group and help displays.
2. Dataset registration from local files and dataset listing.
3. Strict delegation to Phase A chunkers and canonical SHA-256 chunk IDs.
4. Dry-run planning immutability: DB absent remains absent, existing DB untouched.
5. Strict dataset version resolution: non-existent version fails with 0 fallback.
6. Benchmark evaluation query integrity: missing queries rejected, zero fabricated queries.
7. Full matrix execution rendering tabular results and summary metrics.
8. Caching reuse and --bypass-cache force re-execution via CLI flags.
9. JSON report export matching MatrixRunResult schema.
10. Complete evidence trail inspection (ragbench experiment show).
11. Comparative analysis of pipeline parameters and metrics (ragbench experiment compare).
12. Error handling on invalid YAML configurations.
"""

import asyncio
import json
import shutil
from collections.abc import Generator
from pathlib import Path

import pytest
from click.testing import CliRunner
from sqlalchemy import func, select

from app.cli.main import cli
from app.db.repositories.dataset import DatasetRepository
from app.db.session import get_async_engine, get_session_factory, init_db
from app.engine.chunkers.fixed_token import FixedTokenChunker
from app.engine.parsers.text import PlainTextParser
from app.models.entities import Experiment, ExperimentRun


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return a click CliRunner for executing commands."""
    return CliRunner()


@pytest.fixture
def cli_workspace() -> Generator[Path, None, None]:
    """Create isolated local workspace directory for CLI file tests."""
    ws = Path("test_cli_workspace")
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True, exist_ok=True)
    try:
        yield ws
    finally:
        shutil.rmtree(ws, ignore_errors=True)


@pytest.fixture
def cli_db(cli_workspace: Path) -> Path:
    """Return isolated SQLite database path for CLI tests."""
    return cli_workspace / "test_cli.db"


def test_cli_root_help(cli_runner: CliRunner) -> None:
    """Verify root CLI help lists available command groups."""
    result = cli_runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "experiment" in result.output
    assert "dataset" in result.output
    assert "RAGBench" in result.output


def test_cli_invalid_yaml_error_handling(cli_runner: CliRunner, cli_workspace: Path) -> None:
    """Verify invalid YAML returns non-zero exit code with friendly error."""
    invalid_yaml = cli_workspace / "invalid.yaml"
    invalid_yaml.write_text("invalid: [unclosed list", encoding="utf-8")

    res = cli_runner.invoke(cli, ["experiment", "plan", str(invalid_yaml)])
    assert res.exit_code != 0
    assert "Error loading configuration" in res.output


def test_cli_dataset_register_and_list(
    cli_runner: CliRunner, cli_workspace: Path, cli_db: Path
) -> None:
    """Verify registering local documents into SQLite and listing datasets."""
    db_url = f"sqlite+aiosqlite:///{cli_db}"

    # Create dummy corpus files
    doc1 = cli_workspace / "doc1.txt"
    doc1.write_text("Photosynthesis converts sunlight into energy.", encoding="utf-8")
    doc2 = cli_workspace / "doc2.txt"
    doc2.write_text(
        "Cellular respiration produces ATP in eukaryotic mitochondria.",
        encoding="utf-8",
    )

    # Register dataset with benchmark queries
    queries_file = cli_workspace / "queries.json"
    queries_file.write_text(
        json.dumps(
            [
                {
                    "query_id": "q-bio-1",
                    "query_text": "How does photosynthesis produce energy?",
                    "ground_truth_chunks": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    reg_res = cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(cli_workspace),
            "--name",
            "BiologyCorpus",
            "--version",
            "1",
            "--chunk-size",
            "30",
            "--queries",
            str(queries_file),
            "--db-url",
            db_url,
        ],
    )
    assert reg_res.exit_code == 0
    assert "Successfully registered dataset 'BiologyCorpus'" in reg_res.output
    assert "Dataset ID:" in reg_res.output
    assert "Dataset Version ID:" in reg_res.output
    assert "Chunking Strategy:  fixed" in reg_res.output
    assert "Benchmark Queries:  1" in reg_res.output

    # List datasets
    list_res = cli_runner.invoke(cli, ["dataset", "list", "--db-url", db_url])
    assert list_res.exit_code == 0
    assert "BiologyCorpus" in list_res.output


def test_cli_dataset_chunking_delegates_to_phase_a(
    cli_runner: CliRunner, cli_workspace: Path
) -> None:
    """Verify CLI dataset registration uses Phase A chunker producing canonical chunk IDs."""
    db_path = cli_workspace / "canonical.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    sample_text = (
        "Cellular respiration is a set of metabolic reactions that convert chemical energy "
        "from oxygen nutrients into adenosine triphosphate (ATP), and then release waste products."
    )
    doc_path = cli_workspace / "canonical_doc.txt"
    doc_path.write_text(sample_text, encoding="utf-8")

    # 1. Register via CLI using fixed token strategy
    reg_res = cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(doc_path),
            "--name",
            "CanonicalDS",
            "--version",
            "1",
            "--chunk-size",
            "16",
            "--chunk-overlap",
            "4",
            "--db-url",
            db_url,
        ],
    )
    assert reg_res.exit_code == 0

    # 2. Directly chunk using Phase A FixedTokenChunker
    raw_doc = PlainTextParser().parse(doc_path)
    chunker = FixedTokenChunker(chunk_size=16, chunk_overlap=4)
    expected_chunks = chunker.chunk(raw_doc)

    # 3. Retrieve chunks persisted by CLI and verify exact match with Phase A canonical IDs
    async def _verify() -> None:
        engine = get_async_engine(db_url)
        session_factory = get_session_factory(engine)
        async with session_factory() as session:
            ds_repo = DatasetRepository(session)
            ds = await ds_repo.get_dataset_by_name("CanonicalDS")
            assert ds is not None
            assert len(ds.versions) == 1
            docs = await ds_repo.get_documents(ds.versions[0].id)
            assert len(docs) == 1
            cli_chunks = docs[0].chunks

            assert len(cli_chunks) == len(expected_chunks)
            for cli_c, exp_c in zip(cli_chunks, expected_chunks, strict=False):
                # Verify exact canonical SHA-256 chunk ID from generate_chunk_id
                assert cli_c.id == exp_c.chunk_id
                assert cli_c.content == exp_c.content
                assert cli_c.chunk_index == exp_c.chunk_index
                assert cli_c.strategy == exp_c.strategy

        await engine.dispose()

    asyncio.run(_verify())


def test_cli_plan_dry_run_immutability(cli_runner: CliRunner, cli_workspace: Path) -> None:
    """Verify dry-run plan is strictly read-only: does not create DB file or mutate tables."""
    absent_db = cli_workspace / "absent_db.db"
    db_url = f"sqlite+aiosqlite:///{absent_db}"

    yaml_content = """
name: "CLI Plan Immutability Test"
description: "Testing dry-run read-only invariant"
dataset:
  dataset_id: "ds-plan-1"
  dataset_version_id: "ver-plan-1"
parameters:
  retrieval:
    mode: ["bm25", "dense"]
    top_k: 3
  evaluation:
    metrics: ["recall"]
    k_values: [1]
"""
    cfg_file = cli_workspace / "plan_experiment.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    # Invariant A: DB file does NOT exist before plan -> DB file MUST NOT exist after plan
    assert not absent_db.exists()
    plan_res = cli_runner.invoke(
        cli,
        ["experiment", "plan", str(cfg_file), "--db-url", db_url],
    )
    assert plan_res.exit_code == 0
    assert "Planned Execution Matrix" in plan_res.output
    assert not absent_db.exists(), "DB file must not be created during dry-run plan!"


def test_cli_plan_existing_db_zero_records_added(
    cli_runner: CliRunner, cli_workspace: Path
) -> None:
    """Verify dry-run against existing database creates zero experiments or experiment runs."""
    existing_db = cli_workspace / "existing_db.db"
    db_url = f"sqlite+aiosqlite:///{existing_db}"

    # Initialize empty DB schema
    async def _init_and_count() -> tuple[int, int]:
        engine = get_async_engine(db_url)
        await init_db(engine)
        session_factory = get_session_factory(engine)
        async with session_factory() as session:
            exp_count = (await session.execute(select(func.count(Experiment.id)))).scalar_one()
            run_count = (await session.execute(select(func.count(ExperimentRun.id)))).scalar_one()
        await engine.dispose()
        return exp_count, run_count

    exp_before, run_before = asyncio.run(_init_and_count())
    assert exp_before == 0
    assert run_before == 0

    yaml_content = """
name: "CLI Plan Existing DB Test"
description: "Testing dry-run on existing DB"
dataset:
  dataset_id: "ds-plan-2"
  dataset_version_id: "ver-plan-2"
parameters:
  retrieval:
    mode: ["bm25"]
    top_k: 2
  evaluation:
    metrics: ["recall"]
    k_values: [1]
"""
    cfg_file = cli_workspace / "plan_existing.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    plan_res = cli_runner.invoke(
        cli,
        ["experiment", "plan", str(cfg_file), "--db-url", db_url],
    )
    assert plan_res.exit_code == 0

    # Invariant B: Zero rows added
    async def _count_after() -> tuple[int, int]:
        engine = get_async_engine(db_url)
        session_factory = get_session_factory(engine)
        async with session_factory() as session:
            exp_count = (await session.execute(select(func.count(Experiment.id)))).scalar_one()
            run_count = (await session.execute(select(func.count(ExperimentRun.id)))).scalar_one()
        await engine.dispose()
        return exp_count, run_count

    exp_after, run_after = asyncio.run(_count_after())
    assert exp_after == exp_before == 0
    assert run_after == run_before == 0


def test_cli_dataset_version_strictness(
    cli_runner: CliRunner, cli_workspace: Path, cli_db: Path
) -> None:
    """Verify strict dataset_version_id resolution: missing version errors without fallback."""
    db_url = f"sqlite+aiosqlite:///{cli_db}"

    # 1. Register V1 and V2
    doc = cli_workspace / "sample.txt"
    doc.write_text("Sample content for version testing.", encoding="utf-8")

    cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(doc),
            "--name",
            "StrictDS",
            "--version",
            "1",
            "--db-url",
            db_url,
        ],
    )
    cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(doc),
            "--name",
            "StrictDS",
            "--version",
            "2",
            "--db-url",
            db_url,
        ],
    )

    # 2. Request non-existent version ID
    yaml_content = """
name: "Strict Version Test"
description: "Must fail when requested version does not exist"
dataset:
  dataset_id: "StrictDS"
  dataset_version_id: "non-existent-version-uuid"
parameters:
  retrieval:
    mode: ["bm25"]
    top_k: 2
  evaluation:
    metrics: ["recall"]
    k_values: [1]
"""
    cfg_file = cli_workspace / "strict_version.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    res = cli_runner.invoke(
        cli,
        ["experiment", "run", str(cfg_file), "--db-url", db_url],
    )
    assert res.exit_code != 0
    assert "Error: Dataset version 'non-existent-version-uuid' not found" in res.output
    assert "Automatic fallback is disabled" in res.output


def test_cli_missing_benchmark_queries_rejection(
    cli_runner: CliRunner, cli_workspace: Path, cli_db: Path
) -> None:
    """Verify experiment run fails when dataset has no benchmark evaluation queries."""
    db_url = f"sqlite+aiosqlite:///{cli_db}"

    doc = cli_workspace / "no_queries.txt"
    doc.write_text("Photosynthesis converts solar light into carbohydrates.", encoding="utf-8")

    # Register WITHOUT queries
    reg_res = cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(doc),
            "--name",
            "NoQueriesDS",
            "--version",
            "1",
            "--db-url",
            db_url,
        ],
    )
    assert reg_res.exit_code == 0

    ver_id = ""
    for line in reg_res.output.splitlines():
        if "Dataset Version ID:" in line:
            ver_id = line.split("Dataset Version ID:")[1].split("(")[0].strip()
            break
    assert ver_id

    yaml_content = f"""
name: "No Queries Rejection Test"
description: "Must reject dataset without benchmark queries"
dataset:
  dataset_id: "NoQueriesDS"
  dataset_version_id: "{ver_id}"
parameters:
  retrieval:
    mode: ["bm25"]
    top_k: 2
  evaluation:
    metrics: ["recall"]
    k_values: [1]
"""
    cfg_file = cli_workspace / "no_queries.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    res = cli_runner.invoke(
        cli,
        ["experiment", "run", str(cfg_file), "--db-url", db_url],
    )
    assert res.exit_code != 0
    assert "contains no benchmark evaluation queries" in res.output
    # Assert zero fabricated query
    assert "q-default-1" not in res.output


def test_cli_run_full_matrix_caching_and_inspection(
    cli_runner: CliRunner, cli_workspace: Path, cli_db: Path
) -> None:
    """Verify full matrix execution, caching hits, --bypass-cache, show trace, and compare diffs."""
    db_url = f"sqlite+aiosqlite:///{cli_db}"

    # 1. Register dataset with benchmark queries
    doc = cli_workspace / "biology.txt"
    doc.write_text(
        "Photosynthesis converts sunlight into energy. Chlorophyll absorbs light in chloroplasts.",
        encoding="utf-8",
    )

    # First get chunk_id to form valid ground truth
    raw_doc = PlainTextParser().parse(doc)
    chunker = FixedTokenChunker(chunk_size=40, chunk_overlap=0)
    expected_chunks = chunker.chunk(raw_doc)
    gt_chunk_id = expected_chunks[0].chunk_id

    q_file = cli_workspace / "bio_queries.json"
    q_file.write_text(
        json.dumps(
            [
                {
                    "query_id": "q-bio-01",
                    "query_text": "What absorbs light in chloroplasts?",
                    "ground_truth_chunks": [gt_chunk_id],
                }
            ]
        ),
        encoding="utf-8",
    )

    reg_res = cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(doc),
            "--name",
            "BioDS",
            "--version",
            "1",
            "--chunk-size",
            "40",
            "--chunk-overlap",
            "0",
            "--queries",
            str(q_file),
            "--db-url",
            db_url,
        ],
    )
    assert reg_res.exit_code == 0

    ver_id = ""
    for line in reg_res.output.splitlines():
        if "Dataset Version ID:" in line:
            ver_id = line.split("Dataset Version ID:")[1].split("(")[0].strip()
            break
    assert ver_id

    # 2. Write experiment config with a parameter sweep: 2 retrieval modes
    yaml_content = f"""
name: "CLI Full Study"
description: "Full matrix execution with parameter diffs"
dataset:
  dataset_id: "BioDS"
  dataset_version_id: "{ver_id}"
parameters:
  retrieval:
    mode: ["bm25", "dense"]
    top_k: 2
  evaluation:
    metrics: ["recall"]
    k_values: [1]
"""
    cfg_file = cli_workspace / "study.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")
    json_out = cli_workspace / "results.json"

    # 3. First execution: both configurations execute freshly
    res1 = cli_runner.invoke(
        cli,
        [
            "experiment",
            "run",
            str(cfg_file),
            "--concurrency",
            "2",
            "--output-json",
            str(json_out),
            "--db-url",
            db_url,
        ],
    )
    assert res1.exit_code == 0
    assert "Matrix Status: COMPLETED" in res1.output
    assert "Total Configurations: 2" in res1.output
    assert "Executed: 2" in res1.output
    assert "Cached: 0" in res1.output
    assert json_out.exists()

    with open(json_out, encoding="utf-8") as f:
        data = json.load(f)
    run_id_1 = data["results"][0]["experiment_run_id"]
    run_id_2 = data["results"][1]["experiment_run_id"]

    # 4. Second execution: both configurations hit cache
    res2 = cli_runner.invoke(
        cli,
        ["experiment", "run", str(cfg_file), "--db-url", db_url],
    )
    assert res2.exit_code == 0
    assert "Matrix Status: COMPLETED" in res2.output
    assert "Executed: 0" in res2.output
    assert "Cached: 2" in res2.output

    # 5. Third execution with --bypass-cache: forces re-execution
    res3 = cli_runner.invoke(
        cli,
        ["experiment", "run", str(cfg_file), "--bypass-cache", "--db-url", db_url],
    )
    assert res3.exit_code == 0
    assert "Executed: 2" in res3.output
    assert "Cached: 0" in res3.output

    # 6. Verify "show" command displays complete evidence trail
    show_res = cli_runner.invoke(
        cli,
        ["experiment", "show", run_id_1, "--db-url", db_url],
    )
    assert show_res.exit_code == 0
    assert f"Experiment Run: {run_id_1}" in show_res.output
    assert "Aggregate Metric Summaries" in show_res.output
    assert "Query Evidence Trails (1 displayed):" in show_res.output
    assert "Original Query: What absorbs light in chloroplasts?" in show_res.output
    assert "Retrieved Chunks:" in show_res.output
    assert "Rank 1" in show_res.output
    assert "Context: strategy=" in show_res.output
    assert "Generation: model=" in show_res.output
    assert "Metrics:" in show_res.output

    # 7. Verify "compare" command displays pipeline parameter differences & metrics
    comp_res = cli_runner.invoke(
        cli,
        ["experiment", "compare", run_id_1, run_id_2, "--db-url", db_url],
    )
    assert comp_res.exit_code == 0
    assert "Comparing Runs:" in comp_res.output
    assert "Pipeline Parameter Differences" in comp_res.output
    assert "retrieval" in comp_res.output
    assert "mode" in comp_res.output
    assert "bm25" in comp_res.output
    assert "dense" in comp_res.output
    assert "Metric Comparison Table" in comp_res.output
    assert "Delta (Run2 - Run1)" in comp_res.output
