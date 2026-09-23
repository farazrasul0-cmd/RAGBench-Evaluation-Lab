"""Core REST API Routes."""

from typing import Any

from fastapi import APIRouter

from app.core.config import settings

api_router = APIRouter()


@api_router.get("/health", summary="Health Check")
async def health_check() -> dict[str, Any]:
    """Return health status and system version."""
    return {
        "status": "healthy",
        "project": settings.project_name,
        "version": settings.version,
    }
