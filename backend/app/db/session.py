"""Asynchronous database engine, session factory, and SQLite WAL pragma configuration."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.base import Base

DEFAULT_SQLITE_URL = "sqlite+aiosqlite:///./ragbench.db"


def configure_sqlite_pragmas(engine: AsyncEngine) -> None:
    """Register connection hooks to enforce WAL mode and foreign keys on SQLite."""

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


def get_async_engine(database_url: str | None = None) -> AsyncEngine:
    """Create configured SQLAlchemy AsyncEngine with SQLite WAL support."""
    url = database_url or DEFAULT_SQLITE_URL
    engine = create_async_engine(
        url,
        echo=False,
        future=True,
    )

    if "sqlite" in url:
        configure_sqlite_pragmas(engine)

    return engine


def get_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create async sessionmaker bound to provided engine."""
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def init_db(engine: AsyncEngine) -> None:
    """Create all relational tables defined in the metadata."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db(engine: AsyncEngine) -> None:
    """Drop all tables defined in the metadata (useful for test isolation)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@asynccontextmanager
async def get_session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Async context manager providing a scoped session with automatic rollback on error."""
    session_factory = get_session_factory(engine)
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
