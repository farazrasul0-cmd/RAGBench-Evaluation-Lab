"""Unit tests for the 'experiment transfer' CLI subcommand (Phase G)."""

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
def transfer_ws() -> Generator[Path, None, None]:
    ws = Path("test_cli_transfer_ws")
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True, exist_ok=True)
    try:
        yield ws
    finally:
        shutil.rmtree(ws, ignore_errors=True)


async def _seed_transfer_test_runs(db_url: str) -> tuple[str, str, str]:
    """Seed three runs: mono (EN-EN), dense cross (EN-BN), and hybrid cross (EN-BN)."""
    engine = get_async_engine(db_url)
    await init_db(engine)
    session_factory = get_session_factory(engine)

    uid = uuid.uuid4().hex[:8]
    async with session_factory() as session:
        ds_repo = DatasetRepository(session)
        exp_repo = ExperimentRepository(session)
        trace_repo = QueryTraceRepository(session)

        ds = await ds_repo.create_dataset(name=f"MultilingualCorpus_{uid}")
        ver = await ds_repo.create_version(
            dataset_id=ds.id,
            version_number=1,
            content_hash=f"hash-multilingual-{uid}",
            metadata={"domain": "multilingual"},
        )

        # 1. Monolingual Experiment (EN-EN)
        exp_mono = await exp_repo.create_experiment(
            name=f"ExpMono_{uid}",
            dataset_version_id=ver.id,
            configuration={"retrieval": {"strategy": "dense"}, "modality": "EN-EN"},
            configuration_hash=f"cfg_mono_{uid}",
        )
        r_mono = await exp_repo.create_run(
            experiment_id=exp_mono.id,
            pipeline_config_hash="p_mono",
            cache_key="k_mono",
            cache_hash=f"h_mono_{uid}",
        )
        r_mono.status = "COMPLETED"

        # 2. Dense Cross Experiment (EN-BN)
        exp_cross = await exp_repo.create_experiment(
            name=f"ExpCross_{uid}",
            dataset_version_id=ver.id,
            configuration={"retrieval": {"strategy": "dense"}, "modality": "EN-BN"},
            configuration_hash=f"cfg_cross_{uid}",
        )
        r_cross = await exp_repo.create_run(
            experiment_id=exp_cross.id,
            pipeline_config_hash="p_cross",
            cache_key="k_cross",
            cache_hash=f"h_cross_{uid}",
        )
        r_cross.status = "COMPLETED"

        # 3. Hybrid Cross Experiment (EN-BN)
        exp_hybrid = await exp_repo.create_experiment(
            name=f"ExpHybrid_{uid}",
            dataset_version_id=ver.id,
            configuration={"retrieval": {"strategy": "hybrid"}, "modality": "EN-BN"},
            configuration_hash=f"cfg_hybrid_{uid}",
        )
        r_hybrid = await exp_repo.create_run(
            experiment_id=exp_hybrid.id,
            pipeline_config_hash="p_hybrid",
            cache_key="k_hybrid",
            cache_hash=f"h_hybrid_{uid}",
        )
        r_hybrid.status = "COMPLETED"

        # 5 matched units
        unit_ids = [f"climate_{i:02d}" for i in range(1, 6)]

        for i, u_id in enumerate(unit_ids):
            # Mono (high recall)
            await trace_repo.record_query_trace(
                experiment_run_id=r_mono.id,
                query_id=f"q_{u_id}_en_en",
                original_query=f"Question for {u_id} in English",
                latency_ms=40.0 + i,
                status="SUCCESS",
                metric_results=[
                    {"metric_name": "recall@5", "metric_value": 0.95},
                    {"metric_name": "ndcg@5", "metric_value": 0.90},
                ],
                validate_chunk_references=False,
            )

            # Dense cross (penalty: lower recall)
            await trace_repo.record_query_trace(
                experiment_run_id=r_cross.id,
                query_id=f"q_{u_id}_en_bn",
                original_query=f"Question for {u_id} cross-lingual",
                latency_ms=45.0 + i,
                status="SUCCESS",
                metric_results=[
                    {"metric_name": "recall@5", "metric_value": 0.70},
                    {"metric_name": "ndcg@5", "metric_value": 0.65},
                ],
                validate_chunk_references=False,
            )

            # Hybrid cross (attenuated penalty: higher recall than dense cross)
            await trace_repo.record_query_trace(
                experiment_run_id=r_hybrid.id,
                query_id=f"q_{u_id}_en_bn",
                original_query=f"Question for {u_id} hybrid cross-lingual",
                latency_ms=50.0 + i,
                status="SUCCESS",
                metric_results=[
                    {"metric_name": "recall@5", "metric_value": 0.85},
                    {"metric_name": "ndcg@5", "metric_value": 0.80},
                ],
                validate_chunk_references=False,
            )

        await session.commit()

    await engine.dispose()
    return r_mono.id, r_cross.id, r_hybrid.id


def test_cli_transfer_claim_a(transfer_ws: Path) -> None:
    """'experiment transfer <mono> <cross>' renders Claim A transfer penalty table."""
    import asyncio

    db_file = transfer_ws / "test_transfer.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    mono_id, cross_id, _ = asyncio.run(_seed_transfer_test_runs(db_url))

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "experiment",
            "transfer",
            mono_id,
            cross_id,
            "--stats",
            "--db-url",
            db_url,
        ],
    )
    assert res.exit_code == 0, f"Command failed: {res.output}"
    assert "Cross-Lingual Language Transfer Analysis (Claim A)" in res.output
    assert "recall@5" in res.output
    assert "ndcg@5" in res.output
    assert "Penalty" in res.output
    assert "Cohen's dz" in res.output


def test_cli_transfer_claim_b_attenuation(transfer_ws: Path) -> None:
    """'experiment transfer --compare-hybrid' computes Claim B hybrid attenuation."""
    import asyncio

    db_file = transfer_ws / "test_attenuation.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    mono_id, cross_id, hybrid_id = asyncio.run(_seed_transfer_test_runs(db_url))

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "experiment",
            "transfer",
            mono_id,
            cross_id,
            "--compare-hybrid",
            hybrid_id,
            "--stats",
            "--db-url",
            db_url,
        ],
    )
    assert res.exit_code == 0, f"Command failed: {res.output}"
    assert "Hybrid Attenuation of Language Transfer Penalty (Claim B)" in res.output
    assert "Formula: Attenuation = Penalty(Dense) - Penalty(Hybrid)" in res.output
    assert "Attenuated" in res.output


def test_cli_transfer_missing_run(transfer_ws: Path) -> None:
    """Aborts gracefully with error if run ID is not found."""
    db_file = transfer_ws / "test_missing.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "experiment",
            "transfer",
            "non-existent-mono",
            "non-existent-cross",
            "--db-url",
            db_url,
        ],
    )
    assert res.exit_code != 0
    assert "not found" in res.output
