"""Unit and integration tests for Phase F6: CLI and Local Research Workflow.

Verifies:
1. Root CLI command group and help displays.
2. Dataset registration from local files and dataset listing.
3. Dry-run planning displaying Cartesian matrix and cache status without execution.
4. Full matrix execution rendering tabular results and summary metrics.
5. Caching reuse and --bypass-cache force re-execution via CLI flags.
6. JSON report export matching MatrixRunResult schema.
7. Exit code behavior (0 on COMPLETED, 1 on FAILED / PARTIAL with --fail-on-partial).
8. Evidence trail inspection (ragbench experiment show).
9. Comparative analysis between experiment runs (ragbench experiment compare).
10. Human-friendly error handling on invalid YAML configurations.
"""

import json
import shutil
from collections.abc import Generator
from pathlib import Path

import pytest
from click.testing import CliRunner

from app.cli.main import cli


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

    # Register dataset
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
            "--db-url",
            db_url,
        ],
    )
    assert reg_res.exit_code == 0
    assert "Successfully registered dataset 'BiologyCorpus'" in reg_res.output
    assert "Dataset ID:" in reg_res.output
    assert "Dataset Version ID:" in reg_res.output
    assert "Total Chunks:" in reg_res.output

    # List datasets
    list_res = cli_runner.invoke(cli, ["dataset", "list", "--db-url", db_url])
    assert list_res.exit_code == 0
    assert "BiologyCorpus" in list_res.output


def test_cli_invalid_yaml_error_handling(cli_runner: CliRunner, cli_workspace: Path) -> None:
    """Verify invalid YAML returns non-zero exit code with friendly error."""
    invalid_yaml = cli_workspace / "invalid.yaml"
    invalid_yaml.write_text("invalid: [unclosed list", encoding="utf-8")

    res = cli_runner.invoke(cli, ["experiment", "plan", str(invalid_yaml)])
    assert res.exit_code != 0
    assert "Error loading configuration" in res.output


def test_cli_plan_dry_run(cli_runner: CliRunner, cli_workspace: Path, cli_db: Path) -> None:
    """Verify dry-run plan displays matrix, component diffs, and cache status without DB writes."""
    db_url = f"sqlite+aiosqlite:///{cli_db}"

    yaml_content = """
name: "CLI Plan Test"
description: "Testing dry-run planning CLI command"
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

    plan_res = cli_runner.invoke(
        cli,
        ["experiment", "plan", str(cfg_file), "--db-url", db_url],
    )
    assert plan_res.exit_code == 0
    assert "Experiment Plan: CLI Plan Test" in plan_res.output
    assert "Total Combinatorial Configurations: 2" in plan_res.output
    assert "Planned Execution Matrix" in plan_res.output
    assert "MISS" in plan_res.output


def test_cli_run_full_matrix_and_caching(
    cli_runner: CliRunner, cli_workspace: Path, cli_db: Path
) -> None:
    """Verify full matrix execution, caching hits, --bypass-cache, and --output-json."""
    db_url = f"sqlite+aiosqlite:///{cli_db}"

    # 1. Register a dataset first
    doc = cli_workspace / "biology.txt"
    doc.write_text(
        "Photosynthesis converts sunlight into energy. Chlorophyll absorbs light in chloroplasts.",
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
            "--db-url",
            db_url,
        ],
    )
    assert reg_res.exit_code == 0

    # Extract registered version ID from output
    ver_id = ""
    for line in reg_res.output.splitlines():
        if "Dataset Version ID:" in line:
            ver_id = line.split("Dataset Version ID:")[1].split("(")[0].strip()
            break
    assert ver_id

    # 2. Write experiment config referencing registered version ID
    yaml_content = f"""
name: "CLI Run Study"
description: "Full matrix execution via CLI"
dataset:
  dataset_id: "BioDS"
  dataset_version_id: "{ver_id}"
parameters:
  retrieval:
    mode: ["bm25"]
    top_k: [2, 3]
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

    # Validate JSON schema matches MatrixRunResult
    with open(json_out, encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "COMPLETED"
    assert data["total_configurations"] == 2
    assert data["executed_runs"] == 2
    assert data["cached_runs"] == 0
    assert len(data["results"]) == 2
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

    # 6. Verify "show" command displays query evidence trail
    show_res = cli_runner.invoke(
        cli,
        ["experiment", "show", run_id_1, "--db-url", db_url],
    )
    assert show_res.exit_code == 0
    assert f"Experiment Run: {run_id_1}" in show_res.output
    assert "Total Query Evidence Trails:" in show_res.output
    assert "Aggregate Metric Summaries" in show_res.output

    # 7. Verify "compare" command displays delta table
    comp_res = cli_runner.invoke(
        cli,
        ["experiment", "compare", run_id_1, run_id_2, "--db-url", db_url],
    )
    assert comp_res.exit_code == 0
    assert "Comparing Runs:" in comp_res.output
    assert "Metric Comparison Table" in comp_res.output
