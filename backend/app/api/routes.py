"""Core REST API Routes for Academic Research Workbench."""

import hashlib
import os
from collections.abc import AsyncGenerator
from itertools import product
from pathlib import Path
from typing import Annotated, Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.repositories.benchmark import BenchmarkRepository
from app.db.repositories.experiment import ExperimentRepository
from app.db.repositories.query_trace import QueryTraceRepository
from app.db.session import get_async_engine, get_session_factory
from app.schemas.workbench import (
    MatrixConfigurationPoint,
    MatrixPreviewRequest,
    MatrixPreviewResponse,
    MatrixYamlResponse,
)

api_router = APIRouter()


def get_active_db_url() -> str:
    """Resolve active database URL, checking environment and reference databases."""
    env_url = os.environ.get("RAGBENCH_DB_URL")
    if env_url:
        return env_url
    for cand in [
        settings.base_dir.parent / "data" / "experiments" / "g_multilingual_reference_studies.db",
        settings.base_dir / "data" / "experiments" / "g_multilingual_reference_studies.db",
        Path("data/experiments/g_multilingual_reference_studies.db"),
    ]:
        if cand.exists():
            return f"sqlite+aiosqlite:///{cand.resolve()}"
    return f"sqlite+aiosqlite:///{settings.base_dir / 'ragbench.db'}"


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency providing a scoped async database session."""
    from app.db.session import init_db

    engine = get_async_engine(get_active_db_url())
    await init_db(engine)
    session_factory = get_session_factory(engine)
    async with session_factory() as session:
        yield session
    await engine.dispose()


@api_router.get("/health", summary="Health Check")
async def health_check() -> dict[str, Any]:
    """Return health status and system version."""
    return {
        "status": "healthy",
        "project": settings.project_name,
        "version": settings.version,
    }


@api_router.get("/experiments", summary="List Experiments")
async def list_experiments(
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    """List persisted experiments with associated runs and metric rollups."""
    repo = ExperimentRepository(session)
    experiments = await repo.list_experiments(limit=limit, offset=offset)

    output: list[dict[str, Any]] = []
    for exp in experiments:
        runs_data = []
        for r in exp.runs:
            runs_data.append(
                {
                    "run_id": r.id,
                    "pipeline_config_hash": r.pipeline_config_hash,
                    "status": r.status,
                    "summary_metrics": r.summary_metrics or {},
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                }
            )
        output.append(
            {
                "id": exp.id,
                "name": exp.name,
                "description": exp.description,
                "dataset_version_id": exp.dataset_version_id,
                "configuration_hash": exp.configuration_hash,
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
                "runs_count": len(exp.runs),
                "runs": runs_data,
            }
        )
    return output


@api_router.get("/experiments/{experiment_id}", summary="Get Experiment Details")
async def get_experiment(
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, Any]:
    """Fetch complete experiment details including configuration and runs."""
    repo = ExperimentRepository(session)
    exp = await repo.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")

    runs_data = []
    for r in exp.runs:
        metric_summaries = [
            {
                "metric_name": m.metric_name,
                "mean": m.mean,
                "median": m.median,
                "min": m.min,
                "max": m.max,
                "stddev": m.stddev,
                "count": m.count,
            }
            for m in r.metric_summaries
        ]
        runs_data.append(
            {
                "run_id": r.id,
                "pipeline_config_hash": r.pipeline_config_hash,
                "status": r.status,
                "summary_metrics": r.summary_metrics or {},
                "metric_summaries": metric_summaries,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            }
        )

    return {
        "id": exp.id,
        "name": exp.name,
        "description": exp.description,
        "dataset_version_id": exp.dataset_version_id,
        "configuration": exp.configuration,
        "configuration_hash": exp.configuration_hash,
        "benchmark_hash": exp.benchmark_hash,
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
        "runs": runs_data,
    }


def validate_sweep_compatibility(request: MatrixPreviewRequest) -> tuple[bool, str, list[str]]:
    """Authoritative backend validation for sweep execution and topology compatibility."""
    warnings: list[str] = []
    is_executable = True
    status = "COMPATIBLE"

    # 1. Chunk geometry validation
    for sz in request.chunk_sizes:
        for ov in request.chunk_overlaps:
            if ov >= sz:
                warnings.append(
                    f"Invalid chunk geometry: overlap ({ov}) "
                    f"must be strictly less than chunk_size ({sz})"
                )
                is_executable = False
                status = "INCOMPATIBLE"

    # 2. Top-k validation
    for k in request.top_k_values:
        if k <= 0:
            warnings.append(f"Invalid retrieval parameter: top_k ({k}) must be positive")
            is_executable = False
            status = "INCOMPATIBLE"

    # 3. BM25 vs neural embedding compatibility note
    has_bm25 = "bm25" in [s.lower() for s in request.retrieval_strategies]
    if has_bm25 and len(request.embedding_models) > 0:
        warnings.append(
            "BM25 lexical strategy operates via inverted index "
            "and bypasses dense embedding inference."
        )
        if status == "COMPATIBLE":
            status = "WARNINGS"

    # 4. Generation model check
    if all(m.lower() == "none" for m in request.generation_models):
        warnings.append(
            "Retrieval-only evaluation active "
            "(generation faithfulness and citation metrics disabled)."
        )
        if status == "COMPATIBLE":
            status = "WARNINGS"

    return is_executable, status, warnings


@api_router.post("/experiments/matrix/preview", summary="Preview Matrix Sweep Combinations")
async def preview_matrix_sweep(request: MatrixPreviewRequest) -> MatrixPreviewResponse:
    """Calculate combinatorial sweep parameters and preview combinations."""
    is_executable, status, warnings = validate_sweep_compatibility(request)

    combos = list(
        product(
            request.chunking_strategies,
            request.chunk_sizes,
            request.chunk_overlaps,
            request.embedding_models,
            request.retrieval_strategies,
            request.rerankers,
            request.generation_models,
            request.top_k_values,
        )
    )
    total_count = len(combos)

    category = "LOW"
    if total_count > 48:
        category = "HEAVY"
    elif total_count > 12:
        category = "MODERATE"

    sample_points: list[MatrixConfigurationPoint] = []
    for idx, (c_strat, c_sz, c_ov, emb_m, ret_m, rerank, gen_m, k_val) in enumerate(combos[:10]):
        dim = 1024 if "bge-m3" in emb_m.lower() else 384
        sample_points.append(
            MatrixConfigurationPoint(
                index=idx + 1,
                chunking={"strategy": c_strat, "chunk_size": c_sz, "chunk_overlap": c_ov},
                embedding={"model_name": emb_m, "dimension": dim},
                retrieval={"strategy": ret_m, "top_k": k_val},
                reranker={"enabled": rerank != "none", "strategy": rerank},
                generation={"model_name": gen_m},
            )
        )

    controlled_dims = {
        "dataset_id": request.dataset_id,
        "dataset_version_id": request.dataset_version_id,
        "evaluation_protocol": "ragbench-protocol-v1.0",
        "metric_suite": "metrics-v1.0",
        "baseline_reference_commit": "eb3bb4d",
    }

    return MatrixPreviewResponse(
        total_combinations=total_count,
        complexity_category=category,
        estimated_queries_per_run=100,
        total_pipeline_points=total_count,
        is_executable=is_executable,
        compatibility_status=status,
        validation_warnings=warnings,
        controlled_dimensions=controlled_dims,
        sample_configurations=sample_points,
    )


@api_router.post("/experiments/matrix/yaml", summary="Generate Sweep YAML Payload")
async def generate_matrix_yaml(request: MatrixPreviewRequest) -> MatrixYamlResponse:
    """Generate canonical YAML string representing the sweep experiment."""
    is_executable, _, warnings = validate_sweep_compatibility(request)

    combos = list(
        product(
            request.chunking_strategies,
            request.chunk_sizes,
            request.chunk_overlaps,
            request.embedding_models,
            request.retrieval_strategies,
            request.rerankers,
            request.generation_models,
            request.top_k_values,
        )
    )
    total_count = len(combos)

    yaml_dict: dict[str, Any] = {
        "name": request.name,
        "description": request.description,
        "dataset": {
            "dataset_id": request.dataset_id,
            "dataset_version_id": request.dataset_version_id,
        },
        "provenance": {
            "baseline_commit": "eb3bb4d",
            "protocol_version": "ragbench-protocol-v1.0",
            "metrics_version": "metrics-v1.0",
        },
        "parameters": {
            "chunking": {
                "strategies": request.chunking_strategies,
                "chunk_sizes": request.chunk_sizes,
                "chunk_overlaps": request.chunk_overlaps,
            },
            "embedding": {
                "models": request.embedding_models,
            },
            "retrieval": {
                "strategies": request.retrieval_strategies,
                "top_k_values": request.top_k_values,
            },
            "reranker": {
                "strategies": request.rerankers,
            },
            "generation": {
                "models": request.generation_models,
            },
            "evaluation": {
                "metrics": ["recall", "mrr", "ndcg", "precision", "faithfulness"],
                "k_values": request.top_k_values,
            },
        },
    }

    yaml_str = yaml.dump(yaml_dict, sort_keys=False)
    config_hash = hashlib.sha256(yaml_str.encode("utf-8")).hexdigest()

    return MatrixYamlResponse(
        yaml_string=yaml_str,
        total_combinations=total_count,
        configuration_hash=config_hash,
        is_executable=is_executable,
        validation_warnings=warnings,
    )


@api_router.get("/runs/{run_id}/traces", summary="List Query Traces for Run")
async def list_run_traces(
    run_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[dict[str, Any]]:
    """Fetch all query evidence trails for a specific experiment run."""
    trace_repo = QueryTraceRepository(session)
    trails = await trace_repo.get_run_evidence_trails(run_id)
    return [trail.model_dump() for trail in trails]


@api_router.get("/traces/{query_run_id}", summary="Get Detailed Query Trace")
async def get_query_trace(
    query_run_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, Any]:
    """Fetch forensic evidence trail for a specific query run."""
    trace_repo = QueryTraceRepository(session)
    trail = await trace_repo.get_query_evidence_trail(query_run_id)
    if not trail:
        raise HTTPException(status_code=404, detail=f"Query trace '{query_run_id}' not found")
    return trail.model_dump()


@api_router.get("/benchmarks", summary="List Registered Benchmarks")
async def list_benchmarks(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[dict[str, Any]]:
    """List all registered canonical benchmarks and datasets."""
    bench_repo = BenchmarkRepository(session)
    benchmarks = await bench_repo.list_benchmarks()
    return [
        {
            "id": b.id,
            "name": b.name,
            "version_number": b.version_number,
            "benchmark_hash": b.benchmark_hash,
            "query_count": b.query_count,
            "allow_unresolved_passages": b.allow_unresolved_passages,
        }
        for b in benchmarks
    ]
