"""SQLAlchemy 2.0 async relational domain models for RAGBench lineage and persistence."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, generate_uuid, utc_now

# ============================================================================
# 1. Dataset Management Entities
# ============================================================================


class Dataset(Base):
    """The root organizational entity for a corpus of knowledge."""

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    versions: Mapped[list["DatasetVersion"]] = relationship(
        "DatasetVersion", back_populates="dataset", cascade="all, delete-orphan"
    )


class DatasetVersion(Base):
    """An immutable snapshot of a dataset corpus."""

    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    document_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="versions")
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="dataset_version", cascade="all, delete-orphan"
    )
    experiments: Mapped[list["Experiment"]] = relationship(
        "Experiment", back_populates="dataset_version"
    )

    __table_args__ = (
        UniqueConstraint("dataset_id", "version_number", name="uq_dataset_version_number"),
        Index("idx_dataset_version", "dataset_id", "version_number"),
    )


class Document(Base):
    """A discrete ingested file belonging to a specific dataset version."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    dataset_version_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("dataset_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(64), default="text/plain", nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    dataset_version: Mapped["DatasetVersion"] = relationship(
        "DatasetVersion", back_populates="documents"
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("dataset_version_id", "content_hash", name="uq_document_content_hash"),
        Index("idx_document_hash", "dataset_version_id", "content_hash"),
    )


class DocumentChunk(Base):
    """A discrete segmented passage derived from a Document."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    chunking_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
        Index("idx_chunk_document", "document_id", "chunk_index"),
    )


# ============================================================================
# 2. Experiment Management Entities
# ============================================================================


class Experiment(Base):
    """Defines the hyperparameter specification and target dataset for an evaluation study."""

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset_version_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("dataset_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(32), default="DRAFT", nullable=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    configuration_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    dataset_version: Mapped["DatasetVersion"] = relationship(
        "DatasetVersion", back_populates="experiments"
    )
    runs: Mapped[list["ExperimentRun"]] = relationship(
        "ExperimentRun", back_populates="experiment", cascade="all, delete-orphan"
    )


class ExperimentRun(Base):
    """Represents an individual physical execution of a pipeline point within an Experiment."""

    __tablename__ = "experiment_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    experiment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pipeline_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    cache_key: Mapped[str] = mapped_column(String(255), nullable=False)
    cache_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), default="RUNNING", nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    git_commit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    environment: Mapped[str] = mapped_column(String(64), default="local", nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, default=42, nullable=False)
    summary_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="runs")
    query_runs: Mapped[list["QueryRun"]] = relationship(
        "QueryRun", back_populates="experiment_run", cascade="all, delete-orphan"
    )
    metric_summaries: Mapped[list["RunMetricSummary"]] = relationship(
        "RunMetricSummary", back_populates="experiment_run", cascade="all, delete-orphan"
    )


# ============================================================================
# 3. Query-Level Trace & Evidence Trail Entities
# ============================================================================


class QueryRun(Base):
    """Represents execution of a single benchmark query within an ExperimentRun."""

    __tablename__ = "query_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    experiment_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("experiment_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    query_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    original_query: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    ground_truth_chunks: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="SUCCESS", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    experiment_run: Mapped["ExperimentRun"] = relationship(
        "ExperimentRun", back_populates="query_runs"
    )
    transformed_queries: Mapped[list["TransformedQuery"]] = relationship(
        "TransformedQuery", back_populates="query_run", cascade="all, delete-orphan"
    )
    retrieved_chunks: Mapped[list["RetrievedChunk"]] = relationship(
        "RetrievedChunk", back_populates="query_run", cascade="all, delete-orphan"
    )
    reranked_chunks: Mapped[list["RerankedChunk"]] = relationship(
        "RerankedChunk", back_populates="query_run", cascade="all, delete-orphan"
    )
    packed_context: Mapped["PackedContext | None"] = relationship(
        "PackedContext", back_populates="query_run", uselist=False, cascade="all, delete-orphan"
    )
    generation_result: Mapped["GenerationResult | None"] = relationship(
        "GenerationResult", back_populates="query_run", uselist=False, cascade="all, delete-orphan"
    )
    metric_results: Mapped[list["MetricResult"]] = relationship(
        "MetricResult", back_populates="query_run", cascade="all, delete-orphan"
    )


class TransformedQuery(Base):
    """Stores query expansions produced by query pre-processing strategies."""

    __tablename__ = "transformed_queries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    query_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("query_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    transformation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    sequence_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    query_run: Mapped["QueryRun"] = relationship("QueryRun", back_populates="transformed_queries")


class RetrievedChunk(Base):
    """Captures raw passage retrieval rankings before reranking."""

    __tablename__ = "retrieved_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    query_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("query_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    retriever_type: Mapped[str] = mapped_column(String(32), nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    query_run: Mapped["QueryRun"] = relationship("QueryRun", back_populates="retrieved_chunks")


class RerankedChunk(Base):
    """Captures re-scoring and rank shifts produced by cross-encoders."""

    __tablename__ = "reranked_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    query_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("query_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    original_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    reranked_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    original_score: Mapped[float] = mapped_column(Float, nullable=False)
    reranker_score: Mapped[float] = mapped_column(Float, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    query_run: Mapped["QueryRun"] = relationship("QueryRun", back_populates="reranked_chunks")


class PackedContext(Base):
    """Captures the final context window provided to the generator."""

    __tablename__ = "packed_contexts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    query_run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("query_runs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    context_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    token_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    ordering_strategy: Mapped[str] = mapped_column(String(32), default="standard", nullable=False)
    chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    query_run: Mapped["QueryRun"] = relationship("QueryRun", back_populates="packed_context")


class GenerationResult(Base):
    """Captures LLM completion and resource metrics."""

    __tablename__ = "generation_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    query_run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("query_runs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), default="v1", nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    query_run: Mapped["QueryRun"] = relationship("QueryRun", back_populates="generation_result")


class MetricResult(Base):
    """Captures individual evaluation metrics per query."""

    __tablename__ = "metric_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    query_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("query_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_version: Mapped[str] = mapped_column(String(32), default="v1", nullable=False)
    evaluator: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    query_run: Mapped["QueryRun"] = relationship("QueryRun", back_populates="metric_results")

    __table_args__ = (
        UniqueConstraint("query_run_id", "metric_name", name="uq_query_run_metric"),
        Index("idx_metric_query_name", "query_run_id", "metric_name"),
    )


class RunMetricSummary(Base):
    """Stores aggregate statistical metrics across all queries in a run."""

    __tablename__ = "run_metric_summaries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    experiment_run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("experiment_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    mean: Mapped[float] = mapped_column(Float, nullable=False)
    median: Mapped[float] = mapped_column(Float, nullable=False)
    min: Mapped[float] = mapped_column(Float, nullable=False)
    max: Mapped[float] = mapped_column(Float, nullable=False)
    stddev: Mapped[float] = mapped_column(Float, nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    experiment_run: Mapped["ExperimentRun"] = relationship(
        "ExperimentRun", back_populates="metric_summaries"
    )

    __table_args__ = (
        UniqueConstraint("experiment_run_id", "metric_name", name="uq_run_metric_summary"),
        Index("idx_run_metric_summary", "experiment_run_id", "metric_name"),
    )
