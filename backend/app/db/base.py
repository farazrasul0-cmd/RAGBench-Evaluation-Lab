"""SQLAlchemy declarative base and common model utilities."""

import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import DeclarativeBase


def generate_uuid() -> str:
    """Generate a standard UUID4 hexadecimal string."""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base class for all RAGBench relational models."""

    pass
