"""Embedding provider configuration schema for reproducibility (Phase G Amendment 2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmbeddingProviderConfig(BaseModel):
    """Frozen empirical embedding configuration for reproducibility.

    Every RQ3 run must persist all fields of this schema so experiments can be
    independently reproduced.  In particular, ``query_instruction`` and
    ``document_instruction`` must be explicit (None for symmetric models such as
    bge-m3) and never silently inferred by the provider.
    """

    model_identifier: str = Field(
        description="HuggingFace model ID or canonical model name (e.g. 'BAAI/bge-m3')"
    )
    model_revision: str | None = Field(
        default=None,
        description=(
            "Exact model commit hash / version tag from HuggingFace Hub if available. "
            "Records the precise checkpoint used so results are independently reproducible."
        ),
    )
    embedding_dimension: int = Field(
        description="Number of dimensions in the output embedding vector (e.g. 1024 for bge-m3)"
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Whether L2 normalization is applied to embeddings before indexing/querying",
    )
    embedding_provider: str = Field(
        description="Provider backend used (e.g. 'fastembed', 'sentence-transformers', 'mock')"
    )
    similarity_metric: str = Field(
        default="cosine",
        description="Distance / similarity metric used for nearest-neighbour retrieval",
    )
    query_instruction: str | None = Field(
        default=None,
        description=(
            "Model-specific query prefix (e.g. 'query: ' for E5-style models). "
            "None for symmetric models such as BAAI/bge-m3 which use identical "
            "encoding for queries and passages."
        ),
    )
    document_instruction: str | None = Field(
        default=None,
        description=(
            "Model-specific document prefix (e.g. 'passage: ' for E5-style models). "
            "None for symmetric models such as BAAI/bge-m3."
        ),
    )
