"""Data models for context window formulation and citation tracking."""

from typing import Any

from pydantic import BaseModel, Field


class ContextChunk(BaseModel):
    """Represents a discrete passage selected and placed within the context window."""

    source_index: int
    doc_id: str
    chunk_id: str
    content: str
    token_count: int
    score: float
    original_rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class PackedContext(BaseModel):
    """Structured result of budget-constrained context formulation."""

    text: str
    total_tokens: int
    token_budget: int
    strategy: str
    chunks: list[ContextChunk]
    included_chunks_count: int
    truncated_chunks_count: int
