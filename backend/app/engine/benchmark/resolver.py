"""Deterministic resolution of benchmark ground truth against generated chunks."""

import re

from app.engine.runner.models import EvaluationQuery
from app.schemas.benchmark import BenchmarkQuery, BenchmarkQuerySet, GroundTruthPassage
from app.schemas.chunk import DocumentChunk


class UnresolvedPassageError(Exception):
    """Raised when a gold passage cannot be mapped to any chunk in active configuration."""


class AmbiguousPassageError(Exception):
    """Raised when a benchmark passage is ambiguous across unrelated documents."""


class GroundTruthChunkResolver:
    def __init__(self, allow_unresolved_passages: bool = False) -> None:
        self.allow_unresolved_passages = allow_unresolved_passages

    """Deterministically maps source-level passages to chunks generated for a run.

    This ensures complete decoupling:
      Benchmark Ground Truth: Question -> Document -> Source Passage / Span
      Active Chunking Run:    Fixed / Recursive / Sentence -> Generated Chunks
      Resolver:               Maps Source Passage -> Generated Chunk ID(s) dynamically for this run.
    """

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Normalize whitespace and punctuation for robust substring and span matching."""
        return " ".join(text.strip().split())

    @classmethod
    def resolve_chunks_for_passage(
        cls,
        passage: GroundTruthPassage,
        chunks: list[DocumentChunk],
    ) -> list[str]:
        """Find all chunk IDs that contain or overlap with a source ground truth passage."""
        # 1. Filter candidate chunks belonging to the target source document
        candidates: list[DocumentChunk] = []
        for c in chunks:
            doc_identifier = passage.doc_id.strip()
            chunk_doc_id = c.doc_id.strip() if hasattr(c, "doc_id") else ""
            chunk_filename = str(c.metadata.get("filename", "")).strip()

            if (
                (chunk_doc_id and chunk_doc_id == doc_identifier)
                or (chunk_filename and chunk_filename == doc_identifier)
                or (doc_identifier in c.chunk_id)
            ):
                candidates.append(c)

        if not candidates:
            # Fall back to all chunks if document ID not matched directly
            candidates = chunks

        norm_snippet = cls.normalize_text(passage.text_snippet)
        if not norm_snippet:
            return []

        matching_chunk_ids: list[str] = []

        for c in candidates:
            norm_chunk = cls.normalize_text(c.content)

            # A. Character span overlap check if offsets are available
            if (
                passage.start_char is not None
                and passage.end_char is not None
                and hasattr(c, "start_char")
                and hasattr(c, "end_char")
                and c.start_char is not None
                and c.end_char is not None
            ):
                overlap_len = max(
                    0, min(c.end_char, passage.end_char) - max(c.start_char, passage.start_char)
                )
                if overlap_len > 0:
                    matching_chunk_ids.append(c.chunk_id)
                continue

            # B. Substring containment check:
            # Case 1: Snippet is fully contained within this chunk
            if norm_snippet in norm_chunk:
                matching_chunk_ids.append(c.chunk_id)
                continue

            # Case 2: Snippet spans chunk boundaries (chunk is fully contained within snippet)
            if norm_chunk in norm_snippet and len(norm_chunk) > 15:
                matching_chunk_ids.append(c.chunk_id)
                continue

            # Case 3: Contiguous phrase overlap for boundary splits
            snippet_words_list = re.findall(r"\w+", norm_snippet.lower())
            chunk_words_list = re.findall(r"\w+", norm_chunk.lower())

            matched_phrase = False
            # Check 4-gram sliding window
            if len(snippet_words_list) >= 4:
                chunk_words_str = " " + " ".join(chunk_words_list) + " "
                for w_idx in range(len(snippet_words_list) - 3):
                    phrase = " " + " ".join(snippet_words_list[w_idx : w_idx + 4]) + " "
                    if phrase in chunk_words_str:
                        matching_chunk_ids.append(c.chunk_id)
                        matched_phrase = True
                        break

            if matched_phrase:
                continue

            # Case 4: Significant bag-of-words overlap
            snippet_words = set(snippet_words_list)
            chunk_words = set(chunk_words_list)
            if snippet_words and chunk_words:
                common_words = snippet_words & chunk_words
                overlap_ratio = len(common_words) / len(snippet_words)
                if len(common_words) >= 3 and overlap_ratio >= 0.5:
                    matching_chunk_ids.append(c.chunk_id)

        # Check for ambiguity across documents when doc_id is omitted or generic
        matched_docs = {
            c.doc_id if hasattr(c, "doc_id") and c.doc_id else str(c.metadata.get("filename", ""))
            for c in chunks
            if c.chunk_id in matching_chunk_ids
        }
        matched_docs = {d for d in matched_docs if d}
        if len(matched_docs) > 1 and not passage.doc_id:
            raise AmbiguousPassageError(
                f"Ground truth snippet '{passage.text_snippet[:30]}' is "
                f"AMBIGUOUS across multiple documents: {sorted(matched_docs)}"
            )

        # Deterministically deduplicate and sort
        return sorted(set(matching_chunk_ids))

    @classmethod
    def resolve_query(
        cls,
        query: BenchmarkQuery,
        chunks: list[DocumentChunk],
        allow_unresolved: bool = False,
    ) -> EvaluationQuery:
        """Resolve all ground-truth passages of a query to concrete chunk IDs for a run."""
        all_resolved_chunks: list[str] = []

        for passage in query.ground_truth_passages:
            chunk_ids = cls.resolve_chunks_for_passage(passage, chunks)
            if not chunk_ids and not allow_unresolved:
                snippet_preview = passage.text_snippet[:40]
                raise UnresolvedPassageError(
                    f"Gold snippet in '{query.query_id}' could not be resolved: "
                    f"'{snippet_preview}...'"
                )
            all_resolved_chunks.extend(chunk_ids)

        return EvaluationQuery(
            query_id=query.query_id,
            query_text=query.query,
            ground_truth_chunks=sorted(set(all_resolved_chunks)),
            expected_answer=query.ground_truth_answer,
            metadata={
                **query.metadata,
                "domain": query.domain,
                "language": query.language,
                "difficulty": query.difficulty,
                "source_passages_count": len(query.ground_truth_passages),
            },
        )

    @classmethod
    def resolve_benchmark(
        cls,
        benchmark: BenchmarkQuerySet,
        chunks: list[DocumentChunk] | None = None,
        corpus_chunks: list[DocumentChunk] | None = None,
    ) -> list[EvaluationQuery]:
        """Resolve an entire BenchmarkQuerySet against the concrete chunks of a run."""
        target_chunks = chunks if chunks is not None else (corpus_chunks or [])
        resolved: list[EvaluationQuery] = []
        for q in benchmark.queries:
            eq = cls.resolve_query(
                query=q,
                chunks=target_chunks,
                allow_unresolved=benchmark.allow_unresolved_passages,
            )
            resolved.append(eq)
        return resolved
