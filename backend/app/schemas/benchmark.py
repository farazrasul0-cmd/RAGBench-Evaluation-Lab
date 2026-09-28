"""Schemas for first-class benchmark QA datasets, queries, and ground-truth passages."""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


class GroundTruthPassage(BaseModel):
    """Represents an authoritative gold source passage proving answer factuality.

    This representation is strictly chunking-independent: it references the source document
    and text snippet / character span, never a specific chunking configuration ID.
    """

    doc_id: str = Field(description="Document ID or filename containing passage in the corpus")
    passage_id: str | None = Field(default=None, description="Optional explicit passage identifier")
    text_snippet: str = Field(description="Exact textual passage snippet from source document")
    start_char: int | None = Field(
        default=None, description="Optional character start offset in source document"
    )
    end_char: int | None = Field(
        default=None, description="Optional character end offset in source document"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkQuery(BaseModel):
    """An individual standardized benchmark evaluation question."""

    query_id: str = Field(description="Unique benchmark query identifier")
    query: str = Field(description="Question / query text")
    ground_truth_answer: str = Field(description="Reference gold answer text")
    ground_truth_passages: list[GroundTruthPassage] = Field(
        default_factory=list,
        description="Authoritative source passages required to answer the query",
    )
    domain: str = Field(default="general", description="Subject matter domain")
    language: str = Field(default="en", description="ISO 639-1 language code")
    difficulty: str = Field(default="medium", description="Subjective difficulty rating")
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkQuerySet(BaseModel):
    """An authoritative collection of benchmark queries associated with a dataset version.

    The benchmark identity and hash are computed purely from source text, questions, answers,
    and benchmark specification policies, ensuring complete independence from chunking.
    """

    name: str = Field(description="Human-readable benchmark name")
    version: int = Field(default=1, description="Benchmark specification version number")
    description: str = Field(default="", description="Optional benchmark description")
    queries: list[BenchmarkQuery] = Field(
        default_factory=list, description="Ordered benchmark queries"
    )
    allow_unresolved_passages: bool = Field(
        default=False,
        description="Explicit benchmark-level specification policy on unresolved passages",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def compute_benchmark_hash(self) -> str:
        """Compute deterministic SHA-256 hash of the canonical chunking-independent benchmark."""
        payload = [
            {
                "query_id": q.query_id,
                "query": q.query.strip(),
                "ground_truth_answer": q.ground_truth_answer.strip(),
                "passages": [
                    {
                        "doc_id": p.doc_id.strip(),
                        "passage_id": p.passage_id,
                        "text_snippet": " ".join(p.text_snippet.strip().split()),
                        "start_char": p.start_char,
                        "end_char": p.end_char,
                    }
                    for p in q.ground_truth_passages
                ],
                "domain": q.domain,
                "language": q.language,
                "difficulty": q.difficulty,
            }
            for q in sorted(self.queries, key=lambda x: x.query_id)
        ]
        canonical_dict = {
            "name": self.name.strip(),
            "version": self.version,
            "allow_unresolved_passages": self.allow_unresolved_passages,
            "queries": payload,
        }
        canonical_str = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
