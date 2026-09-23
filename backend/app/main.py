"""FastAPI Application Entry Point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="Enterprise & Academic Evaluation Laboratory for Retrieval-Augmented Generation",
    openapi_url=f"{settings.api_v1_str}/openapi.json",
    docs_url=f"{settings.api_v1_str}/docs",
    redoc_url=f"{settings.api_v1_str}/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_str)


@app.get("/", summary="Root Index")
async def root() -> dict[str, str]:
    """Redirect or inform root access."""
    return {
        "message": "Welcome to RAGBench Evaluation Laboratory API",
        "docs": f"{settings.api_v1_str}/docs",
    }
