"""Repository for granular QueryRun trace persistence and full evidence trail reconstruction."""

from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.entities import (
    GenerationResult,
    MetricResult,
    PackedContext,
    QueryRun,
    RerankedChunk,
    RetrievedChunk,
    TransformedQuery,
)


class QueryEvidenceTrail(BaseModel):
    """Authoritative representation of a single benchmark query's complete evidence trail.

    Enables full reconstruction of query lineage from raw question to expansions,
    retrieval rankings, reranking shifts, packed context, LLM answer, and metrics.
    """

    query_run_id: str
    experiment_run_id: str
    query_id: str
    original_query: str
    expected_answer: str | None = None
    ground_truth_chunks: list[str] = Field(default_factory=list)
    latency_ms: float
    status: str
    transformed_queries: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    reranked_chunks: list[dict[str, Any]] = Field(default_factory=list)
    packed_context: dict[str, Any] | None = None
    generation_result: dict[str, Any] | None = None
    metric_results: dict[str, float] = Field(default_factory=dict)
    detailed_metrics: list[dict[str, Any]] = Field(default_factory=list)


class QueryTraceRepository:
    """Async repository for recording query traces and reconstructing evidence trails."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record_query_trace(
        self,
        experiment_run_id: str,
        query_id: str,
        original_query: str,
        expected_answer: str | None = None,
        ground_truth_chunks: list[str] | None = None,
        latency_ms: float = 0.0,
        status: str = "SUCCESS",
        metadata: dict[str, Any] | None = None,
        transformed_queries: list[dict[str, Any]] | None = None,
        retrieved_chunks: list[dict[str, Any]] | None = None,
        reranked_chunks: list[dict[str, Any]] | None = None,
        packed_context: dict[str, Any] | None = None,
        generation_result: dict[str, Any] | None = None,
        metric_results: list[dict[str, Any]] | None = None,
    ) -> QueryRun:
        """Atomically record all stages of a query's execution trace."""
        query_run = QueryRun(
            experiment_run_id=experiment_run_id,
            query_id=query_id,
            original_query=original_query,
            expected_answer=expected_answer,
            ground_truth_chunks=ground_truth_chunks or [],
            latency_ms=latency_ms,
            status=status,
            metadata_json=metadata or {},
        )
        self.session.add(query_run)
        await self.session.flush()

        # 1. Transformed Queries
        if transformed_queries:
            for idx, tq in enumerate(transformed_queries):
                trans = TransformedQuery(
                    query_run_id=query_run.id,
                    transformation_type=tq["transformation_type"],
                    query_text=tq["query_text"],
                    sequence_index=tq.get("sequence_index", idx),
                    metadata_json=tq.get("metadata", {}),
                )
                self.session.add(trans)

        # 2. Retrieved Chunks
        if retrieved_chunks:
            for rc in retrieved_chunks:
                ret = RetrievedChunk(
                    query_run_id=query_run.id,
                    retriever_type=rc.get("retriever_type", "unknown"),
                    chunk_id=rc["chunk_id"],
                    rank=rc["rank"],
                    score=float(rc["score"]),
                    metadata_json=rc.get("metadata", {}),
                )
                self.session.add(ret)

        # 3. Reranked Chunks
        if reranked_chunks:
            for rk in reranked_chunks:
                rerank = RerankedChunk(
                    query_run_id=query_run.id,
                    chunk_id=rk["chunk_id"],
                    original_rank=rk["original_rank"],
                    reranked_rank=rk["reranked_rank"],
                    original_score=float(rk["original_score"]),
                    reranker_score=float(rk["reranker_score"]),
                    metadata_json=rk.get("metadata", {}),
                )
                self.session.add(rerank)

        # 4. Packed Context
        if packed_context:
            ctx = PackedContext(
                query_run_id=query_run.id,
                context_text=packed_context["context_text"],
                token_count=packed_context["token_count"],
                token_budget=packed_context["token_budget"],
                ordering_strategy=packed_context.get("ordering_strategy", "standard"),
                chunk_ids=packed_context.get("chunk_ids", []),
                metadata_json=packed_context.get("metadata", {}),
            )
            self.session.add(ctx)

        # 5. Generation Result
        if generation_result:
            gen = GenerationResult(
                query_run_id=query_run.id,
                model=generation_result["model"],
                answer=generation_result["answer"],
                prompt_version=generation_result.get("prompt_version", "v1"),
                input_tokens=generation_result.get("input_tokens", 0),
                output_tokens=generation_result.get("output_tokens", 0),
                total_tokens=generation_result.get("total_tokens", 0),
                latency_ms=float(generation_result.get("latency_ms", 0.0)),
                estimated_cost=float(generation_result.get("estimated_cost", 0.0)),
                metadata_json=generation_result.get("metadata", {}),
            )
            self.session.add(gen)

        # 6. Metric Results
        if metric_results:
            for mr in metric_results:
                metric = MetricResult(
                    query_run_id=query_run.id,
                    metric_name=mr["metric_name"],
                    metric_value=float(mr["metric_value"]),
                    metric_version=mr.get("metric_version", "v1"),
                    evaluator=mr.get("evaluator", "standard"),
                    metadata_json=mr.get("metadata", {}),
                )
                self.session.add(metric)

        await self.session.flush()
        return query_run

    async def get_query_evidence_trail(
        self,
        query_run_id: str,
    ) -> QueryEvidenceTrail | None:
        """Eagerly load complete query lineage and reconstruct authoritative evidence trail."""
        stmt = (
            select(QueryRun)
            .where(QueryRun.id == query_run_id)
            .options(
                selectinload(QueryRun.transformed_queries),
                selectinload(QueryRun.retrieved_chunks),
                selectinload(QueryRun.reranked_chunks),
                selectinload(QueryRun.packed_context),
                selectinload(QueryRun.generation_result),
                selectinload(QueryRun.metric_results),
            )
        )
        result = await self.session.execute(stmt)
        qr = result.scalar_one_or_none()
        if not qr:
            return None

        # Build clean, serializable evidence trail
        transformed = [
            {
                "transformation_type": t.transformation_type,
                "query_text": t.query_text,
                "sequence_index": t.sequence_index,
                "metadata": t.metadata_json,
            }
            for t in sorted(qr.transformed_queries, key=lambda x: x.sequence_index)
        ]

        retrieved = [
            {
                "retriever_type": r.retriever_type,
                "chunk_id": r.chunk_id,
                "rank": r.rank,
                "score": r.score,
                "metadata": r.metadata_json,
            }
            for r in sorted(qr.retrieved_chunks, key=lambda x: x.rank)
        ]

        reranked = [
            {
                "chunk_id": r.chunk_id,
                "original_rank": r.original_rank,
                "reranked_rank": r.reranked_rank,
                "original_score": r.original_score,
                "reranker_score": r.reranker_score,
                "metadata": r.metadata_json,
            }
            for r in sorted(qr.reranked_chunks, key=lambda x: x.reranked_rank)
        ]

        packed = (
            {
                "context_text": qr.packed_context.context_text,
                "token_count": qr.packed_context.token_count,
                "token_budget": qr.packed_context.token_budget,
                "ordering_strategy": qr.packed_context.ordering_strategy,
                "chunk_ids": qr.packed_context.chunk_ids,
                "metadata": qr.packed_context.metadata_json,
            }
            if qr.packed_context
            else None
        )

        generation = (
            {
                "model": qr.generation_result.model,
                "answer": qr.generation_result.answer,
                "prompt_version": qr.generation_result.prompt_version,
                "input_tokens": qr.generation_result.input_tokens,
                "output_tokens": qr.generation_result.output_tokens,
                "total_tokens": qr.generation_result.total_tokens,
                "latency_ms": qr.generation_result.latency_ms,
                "estimated_cost": qr.generation_result.estimated_cost,
                "metadata": qr.generation_result.metadata_json,
            }
            if qr.generation_result
            else None
        )

        metrics_map = {m.metric_name: m.metric_value for m in qr.metric_results}
        detailed_metrics = [
            {
                "metric_name": m.metric_name,
                "metric_value": m.metric_value,
                "metric_version": m.metric_version,
                "evaluator": m.evaluator,
                "metadata": m.metadata_json,
            }
            for m in qr.metric_results
        ]

        return QueryEvidenceTrail(
            query_run_id=qr.id,
            experiment_run_id=qr.experiment_run_id,
            query_id=qr.query_id,
            original_query=qr.original_query,
            expected_answer=qr.expected_answer,
            ground_truth_chunks=qr.ground_truth_chunks,
            latency_ms=qr.latency_ms,
            status=qr.status,
            transformed_queries=transformed,
            retrieved_chunks=retrieved,
            reranked_chunks=reranked,
            packed_context=packed,
            generation_result=generation,
            metric_results=metrics_map,
            detailed_metrics=detailed_metrics,
        )

    async def get_run_evidence_trails(
        self,
        experiment_run_id: str,
    ) -> list[QueryEvidenceTrail]:
        """Fetch evidence trails for all queries executed under an experiment run."""
        stmt = (
            select(QueryRun.id)
            .where(QueryRun.experiment_run_id == experiment_run_id)
            .order_by(QueryRun.started_at)
        )
        result = await self.session.execute(stmt)
        query_run_ids = list(result.scalars().all())

        trails: list[QueryEvidenceTrail] = []
        for qid in query_run_ids:
            trail = await self.get_query_evidence_trail(qid)
            if trail:
                trails.append(trail)

        return trails
