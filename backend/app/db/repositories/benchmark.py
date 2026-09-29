"""Repository for first-class BenchmarkVersion persistence and retrieval."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.protocols import DEFAULT_EVALUATION_PROTOCOL, DEFAULT_METRIC_PROTOCOL
from app.models.entities import BenchmarkVersion
from app.schemas.benchmark import BenchmarkQuerySet


class BenchmarkRepository:
    """Async repository for managing versioned benchmark specifications."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_benchmark(
        self,
        name: str,
        version_number: int = 1,
        benchmark_set: BenchmarkQuerySet | None = None,
        benchmark_hash: str | None = None,
        description: str = "",
        query_count: int | None = None,
        allow_unresolved_passages: bool = False,
        benchmark_data: dict[str, Any] | None = None,
        evaluation_protocol_version: str | None = None,
        metric_definition_version: str | None = None,
    ) -> BenchmarkVersion:
        """Create and persist an immutable BenchmarkVersion snapshot."""
        if benchmark_set is not None:
            b_hash = benchmark_set.compute_benchmark_hash()
            q_cnt = len(benchmark_set.queries)
            unres_allowed = benchmark_set.allow_unresolved_passages
            b_data = benchmark_set.model_dump()
            eval_proto = getattr(
                benchmark_set, "evaluation_protocol_version", DEFAULT_EVALUATION_PROTOCOL
            )
            metric_proto = getattr(
                benchmark_set, "metric_definition_version", DEFAULT_METRIC_PROTOCOL
            )
        else:
            b_hash = benchmark_hash or ""
            q_cnt = query_count or 0
            unres_allowed = allow_unresolved_passages
            b_data = benchmark_data or {}
            eval_proto = evaluation_protocol_version or DEFAULT_EVALUATION_PROTOCOL
            metric_proto = metric_definition_version or DEFAULT_METRIC_PROTOCOL

        # Check if identical benchmark specification already exists by hash
        existing = await self.get_benchmark_by_hash(b_hash)
        if existing is not None:
            return existing

        bv = BenchmarkVersion(
            name=name,
            version_number=version_number,
            benchmark_hash=b_hash,
            query_count=q_cnt,
            allow_unresolved_passages=unres_allowed,
            evaluation_protocol_version=eval_proto,
            metric_definition_version=metric_proto,
            benchmark_data=b_data,
        )
        self.session.add(bv)
        await self.session.flush()
        return bv

    async def get_benchmark_by_hash(self, benchmark_hash: str) -> BenchmarkVersion | None:
        """Fetch BenchmarkVersion by its deterministic canonical content hash."""
        stmt = select(BenchmarkVersion).where(BenchmarkVersion.benchmark_hash == benchmark_hash)
        res = await self.session.execute(stmt)
        return res.scalar_one_or_none()

    async def get_benchmark(self, name: str, version_number: int) -> BenchmarkVersion | None:
        """Fetch BenchmarkVersion by name and version number."""
        stmt = select(BenchmarkVersion).where(
            BenchmarkVersion.name == name, BenchmarkVersion.version_number == version_number
        )
        res = await self.session.execute(stmt)
        return res.scalar_one_or_none()
