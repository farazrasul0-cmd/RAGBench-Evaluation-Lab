"""Schemas for first-class benchmark QA datasets, queries, and ground-truth passages."""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


class GroundTruthPassage(BaseModel):
    """Represents an authoritative gold passage proving answer factuality."""

    doc_id: str = Field(description="Document ID or filename containing passage")
    passage_id: str | None = Field(default=None, description="Optional explicit passage identifier")
    text_snippet: str = Field(description="Exact textual passage snippet from source document")
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkQuery(BaseModel):
    """An individual standardized benchmark evaluation question."""

    query_id: str = Field(description="Unique benchmark query identifier")
    query: str = Field(description="Question / query text")
    ground_truth_answer: str = Field(description="Reference gold answer text")
    ground_truth_passages: list[GroundTruthPassage] = Field(
        default_factory=list,
        description="Authoritative passages required to answer the query",
    )
    domain: str = Field(default="general", description="Subject matter domain")
    language: str = Field(default="en", description="ISO 639-1 language code")
    difficulty: str = Field(default="medium", description="Subjective difficulty rating")
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkQuerySet(BaseModel):
    """An authoritative collection of benchmark queries associated with a dataset version."""

    name: str = Field(description="Human-readable benchmark name")
    version: int = Field(default=1, description="Benchmark specification version number")
    queries: list[BenchmarkQuery] = Field(
        default_factory=list, description="Ordered benchmark queries"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def compute_benchmark_hash(self) -> str:
        """Compute deterministic SHA-256 hash of the canonical benchmark query set."""
        payload = [
            {
                "query_id": q.query_id,
                "query": q.query.strip(),
                "ground_truth_answer": q.ground_truth_answer.strip(),
                "passages": [
                    {"doc_id": p.doc_id, "text_snippet": p.text_snippet.strip()}
                    for p in q.ground_truth_passages
                ],
                "domain": q.domain,
                "language": q.language,
            }
            for q in sorted(self.queries, key=lambda x: x.query_id)
        ]
        canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
