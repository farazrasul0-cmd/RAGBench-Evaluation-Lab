"""Citation precision and recall metrics according to SYSTEM_ARCHITECTURE.md."""

import re
from typing import Any

from pydantic import BaseModel, Field

from app.engine.metrics.generation import BaseEntailmentClassifier


class CitationMetricsResult(BaseModel):
    """Container for citation precision and citation recall scores."""

    citation_precision: float
    citation_recall: float
    total_citations: int
    valid_citations: int
    total_claims: int
    cited_claims: int
    citation_mappings: list[dict[str, Any]] = Field(default_factory=list)


_CITATION_PATTERN = re.compile(r"\[(?:Source\s+)?([0-9]+)\]", re.IGNORECASE)


def extract_citations(text: str) -> list[int]:
    """Extract integer source IDs from text citations like [Source 1] or [1]."""
    matches = _CITATION_PATTERN.findall(text)
    return [int(m) for m in matches]


def evaluate_citations(
    answer: str,
    context_sources: dict[int, str],
    classifier: BaseEntailmentClassifier,
) -> CitationMetricsResult:
    """Calculate Citation Precision and Recall:

    Citation Precision = Number of citations providing direct entailment / Total citations made
    Citation Recall = Number of required context facts cited / Total factual claims
    """
    if not answer.strip():
        return CitationMetricsResult(
            citation_precision=0.0,
            citation_recall=0.0,
            total_citations=0,
            valid_citations=0,
            total_claims=0,
            cited_claims=0,
        )

    # Split answer into sentence-level claims
    raw_sentences = re.split(r"(?<=[.!?\u0964\u0965])\s+", answer.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    total_citations = 0
    valid_citations = 0
    cited_claims_count = 0
    mappings: list[dict[str, Any]] = []

    for sentence in sentences:
        source_ids = extract_citations(sentence)
        has_valid_citation_for_claim = False

        # Clean text without citations for claim statement
        claim_text = _CITATION_PATTERN.sub("", sentence).strip()

        for sid in source_ids:
            total_citations += 1
            source_content = context_sources.get(sid, "")

            is_entailed = False
            reason = "Source ID not found in context"
            if source_content:
                is_entailed, reason = classifier.entails(source_content, claim_text)

            if is_entailed:
                valid_citations += 1
                has_valid_citation_for_claim = True

            mappings.append(
                {
                    "sentence": sentence,
                    "claim": claim_text,
                    "source_id": sid,
                    "is_entailed": is_entailed,
                    "reason": reason,
                }
            )

        if has_valid_citation_for_claim:
            cited_claims_count += 1

    precision = float(valid_citations) / float(total_citations) if total_citations > 0 else 0.0
    recall = float(cited_claims_count) / float(len(sentences)) if sentences else 0.0

    return CitationMetricsResult(
        citation_precision=precision,
        citation_recall=recall,
        total_citations=total_citations,
        valid_citations=valid_citations,
        total_claims=len(sentences),
        cited_claims=cited_claims_count,
        citation_mappings=mappings,
    )
