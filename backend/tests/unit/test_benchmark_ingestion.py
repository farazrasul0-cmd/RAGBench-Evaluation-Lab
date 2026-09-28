"""Tests for Phase F7.3: Benchmark QA Dataset Hardening & Ingestion Validation.

Verifies:
1. Strict parser failure: corrupt files fail with non-zero exit code (zero silent fallback).
2. Clean eval_qa.jsonl registration: queries and gold passages parsed into BenchmarkQuerySet.
3. Deterministic passage-to-chunk resolution: gold snippets resolve to exact chunk IDs.
4. Ambiguous ground-truth passages matching multiple chunks fail registration.
5. Unresolved ground-truth passages fail registration unless --allow-unresolved-passages is passed.
6. Deterministic SHA-256 benchmark hash calculation and persistence in DatasetVersion.metadata.
"""

import asyncio
import json
import shutil
from collections.abc import Generator
from pathlib import Path

import pytest
from click.testing import CliRunner
from sqlalchemy import select

from app.cli.main import cli
from app.db.session import get_async_engine, get_session_factory
from app.models.entities import DatasetVersion


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def test_workspace() -> Generator[Path, None, None]:
    ws = Path("test_benchmark_ws")
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True, exist_ok=True)
    try:
        yield ws
    finally:
        shutil.rmtree(ws, ignore_errors=True)


@pytest.fixture
def test_db(test_workspace: Path) -> Path:
    return test_workspace / "test_benchmark.db"


def test_strict_parser_failure(cli_runner: CliRunner, test_workspace: Path, test_db: Path) -> None:
    """Corrupted or unparseable files must abort registration without silent fallback."""
    db_url = f"sqlite+aiosqlite:///{test_db}"

    # Create corrupted PDF file that fails PDFParser
    bad_pdf = test_workspace / "corrupt.pdf"
    bad_pdf.write_bytes(b"%PDF-1.4 completely corrupt bytes not valid pdf header or xref")

    res = cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(bad_pdf),
            "--name",
            "CorruptCorpus",
            "--db-url",
            db_url,
        ],
    )
    assert res.exit_code != 0
    assert "Error parsing file" in res.output or "Ingestion aborted" in res.output


def test_eval_qa_ingestion_and_passage_resolution(
    cli_runner: CliRunner, test_workspace: Path, test_db: Path
) -> None:
    """eval_qa.jsonl correctly resolves gold passages to chunks and persists benchmark metadata."""
    db_url = f"sqlite+aiosqlite:///{test_db}"

    # Create corpus doc
    doc = test_workspace / "doc.txt"
    doc.write_text(
        "Quantum computing leverages superposition and entanglement to perform complex "
        "computations. "
        "Classical computers use binary bits that represent either 0 or 1.",
        encoding="utf-8",
    )

    # Create eval_qa.jsonl
    eval_qa = test_workspace / "eval_qa.jsonl"
    q_data = {
        "query_id": "q-qc-1",
        "query": "What physical phenomena does quantum computing leverage?",
        "ground_truth_answer": "Superposition and entanglement.",
        "ground_truth_passages": [
            {
                "doc_id": "doc.txt",
                "text_snippet": "Quantum computing leverages superposition and entanglement",
            }
        ],
        "domain": "physics",
        "difficulty": "easy",
    }
    eval_qa.write_text(json.dumps(q_data) + "\n", encoding="utf-8")

    res = cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(doc),
            "--name",
            "QuantumBench",
            "--version",
            "1",
            "--eval-qa",
            str(eval_qa),
            "--db-url",
            db_url,
        ],
    )
    assert res.exit_code == 0
    assert "Benchmark Queries:  1" in res.output

    # Verify DB persistence of benchmark query set and resolved chunk IDs
    async def _check() -> None:
        engine = get_async_engine(db_url)
        session_factory = get_session_factory(engine)
        async with session_factory() as session:
            stmt = select(DatasetVersion)
            result = await session.execute(stmt)
            ver = result.scalar_one()

            assert "benchmark" in ver.metadata_json
            assert "benchmark_hash" in ver.metadata_json
            bench_meta = ver.metadata_json["benchmark"]
            assert len(bench_meta["queries"]) == 1
            q = bench_meta["queries"][0]
            assert q["query_id"] == "q-qc-1"
            passages = q["ground_truth_passages"]
            assert len(passages) == 1
            assert passages[0]["passage_id"] is not None
            assert "resolved_chunk_id" in passages[0]["metadata"]

            # Check queries compatibility array
            assert "queries" in ver.metadata_json
            q_compat = ver.metadata_json["queries"][0]
            assert len(q_compat["ground_truth_chunks"]) == 1
            assert q_compat["ground_truth_chunks"][0] == passages[0]["passage_id"]

        await engine.dispose()

    asyncio.run(_check())


def test_ambiguous_passage_resolution_fails(
    cli_runner: CliRunner, test_workspace: Path, test_db: Path
) -> None:
    """If a snippet matches multiple chunks, registration must fail with AMBIGUOUS error."""
    db_url = f"sqlite+aiosqlite:///{test_db}"

    # Create two docs with identical text snippet
    doc1 = test_workspace / "doc1.txt"
    doc1.write_text(
        "The mitochondria is the powerhouse of the cell. Unique fact one.", encoding="utf-8"
    )
    doc2 = test_workspace / "doc2.txt"
    doc2.write_text(
        "The mitochondria is the powerhouse of the cell. Unique fact two.", encoding="utf-8"
    )

    eval_qa = test_workspace / "eval_qa.jsonl"
    q_data = {
        "query_id": "q-cell-1",
        "query": "What is the mitochondria?",
        "ground_truth_answer": "Powerhouse of the cell.",
        "ground_truth_passages": [
            {
                "text_snippet": "The mitochondria is the powerhouse of the cell",
            }
        ],
    }
    eval_qa.write_text(json.dumps(q_data) + "\n", encoding="utf-8")

    res = cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(test_workspace),
            "--name",
            "CellBench",
            "--eval-qa",
            str(eval_qa),
            "--db-url",
            db_url,
        ],
    )
    assert res.exit_code != 0
    assert "AMBIGUOUS" in res.output


def test_unresolved_passage_resolution(
    cli_runner: CliRunner, test_workspace: Path, test_db: Path
) -> None:
    """Passage not found in corpus fails by default, but succeeds with allow flag."""
    db_url = f"sqlite+aiosqlite:///{test_db}"

    doc = test_workspace / "doc.txt"
    doc.write_text("General knowledge text without specific answer.", encoding="utf-8")

    eval_qa = test_workspace / "eval_qa.jsonl"
    q_data = {
        "query_id": "q-missing-1",
        "query": "Where is Atlantis?",
        "ground_truth_answer": "Lost underwater.",
        "ground_truth_passages": [
            {
                "text_snippet": "Atlantis is located in the Atlantic Ocean.",
            }
        ],
    }
    eval_qa.write_text(json.dumps(q_data) + "\n", encoding="utf-8")

    # 1. Without flag -> must fail
    res_fail = cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(doc),
            "--name",
            "AtlantisBenchFail",
            "--eval-qa",
            str(eval_qa),
            "--db-url",
            db_url,
        ],
    )
    assert res_fail.exit_code != 0
    assert "UNRESOLVED" in res_fail.output

    # 2. With flag -> must succeed
    res_ok = cli_runner.invoke(
        cli,
        [
            "dataset",
            "register",
            str(doc),
            "--name",
            "AtlantisBenchOk",
            "--eval-qa",
            str(eval_qa),
            "--allow-unresolved-passages",
            "--db-url",
            db_url,
        ],
    )
    assert res_ok.exit_code == 0
    assert "Successfully registered dataset 'AtlantisBenchOk'" in res_ok.output
