"""Semantic similarity chunker with deterministic cosine distance thresholding."""

import math
import re
from collections import Counter
from collections.abc import Callable

from app.core.exceptions import ChunkingError
from app.engine.chunkers.base import BaseChunker
from app.schemas.chunk import DocumentChunk
from app.schemas.document import RawDocument

EmbeddingFunction = Callable[[list[str]], list[list[float]]]


def compute_token_frequencies(text: str) -> Counter[str]:
    """Extract character 3-gram frequencies for robust, language-agnostic text representation."""
    clean = re.sub(r"\s+", " ", text.lower().strip())
    if len(clean) < 3:
        return Counter([clean])
    return Counter([clean[i : i + 3] for i in range(len(clean) - 2)])


def cosine_similarity_ngrams(counter1: Counter[str], counter2: Counter[str]) -> float:
    """Compute cosine similarity between two frequency vectors."""
    intersection = set(counter1.keys()) & set(counter2.keys())
    dot_product = sum(counter1[x] * counter2[x] for x in intersection)

    norm1 = math.sqrt(sum(val * val for val in counter1.values()))
    norm2 = math.sqrt(sum(val * val for val in counter2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)


class SemanticSimilarityChunker(BaseChunker):
    """Inserts chunk boundaries where adjacent sentence windows experience semantic shifts."""

    SENTENCE_REGEX = re.compile(r"(?<=[.!?\u0964])\s+")

    def __init__(
        self,
        similarity_threshold: float = 0.82,
        min_chunk_tokens: int = 40,
        max_chunk_tokens: int = 1024,
        embedding_fn: EmbeddingFunction | None = None,
    ) -> None:
        super().__init__()
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ChunkingError(
                f"similarity_threshold must be between 0.0 and 1.0, got {similarity_threshold}"
            )
        if min_chunk_tokens <= 0:
            raise ChunkingError("min_chunk_tokens must be positive")
        if max_chunk_tokens < min_chunk_tokens:
            raise ChunkingError("max_chunk_tokens must be >= min_chunk_tokens")

        self.similarity_threshold = similarity_threshold
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.embedding_fn = embedding_fn

    def chunk(self, document: RawDocument) -> list[DocumentChunk]:
        """Segment document based on semantic distance boundaries."""
        content = document.content.strip()
        if not content:
            return []

        raw_units = self.SENTENCE_REGEX.split(content)
        units = [u.strip() for u in raw_units if u.strip()]

        if not units:
            return []

        # If embedding function provided, use dense vector embeddings
        if self.embedding_fn is not None:
            vectors = self.embedding_fn(units)
            return self._chunk_with_vectors(document, content, units, vectors)

        # Fallback: deterministic character n-gram cosine similarity model
        return self._chunk_with_ngrams(document, content, units)

    def _chunk_with_ngrams(
        self, document: RawDocument, content: str, units: list[str]
    ) -> list[DocumentChunk]:
        """Chunking using local frequency vectors."""
        unit_counters = [compute_token_frequencies(u) for u in units]
        chunks: list[DocumentChunk] = []

        current_units: list[str] = [units[0]]
        current_tokens = self.count_tokens(units[0])
        chunk_idx = 0
        search_start = 0

        for i in range(1, len(units)):
            sim = cosine_similarity_ngrams(unit_counters[i - 1], unit_counters[i])
            next_tokens = self.count_tokens(units[i])

            semantic_split_triggered = (
                sim < self.similarity_threshold and current_tokens >= self.min_chunk_tokens
            )
            size_limit_reached = (current_tokens + next_tokens) > self.max_chunk_tokens

            if semantic_split_triggered or size_limit_reached:
                chunk_text = " ".join(current_units).strip()
                start_char = content.find(chunk_text[:30], search_start)
                if start_char == -1:
                    start_char = content.find(chunk_text[:30])
                    if start_char == -1:
                        start_char = search_start
                end_char = start_char + len(chunk_text)
                search_start = max(0, start_char + len(chunk_text) // 2)

                chunks.append(
                    self.create_chunk(
                        document=document,
                        chunk_index=chunk_idx,
                        content=chunk_text,
                        start_char=start_char,
                        end_char=end_char,
                        strategy="semantic",
                        metadata={
                            "units_count": len(current_units),
                            "similarity_threshold": self.similarity_threshold,
                        },
                    )
                )
                chunk_idx += 1
                current_units = [units[i]]
                current_tokens = next_tokens
            else:
                current_units.append(units[i])
                current_tokens += next_tokens

        if current_units:
            chunk_text = " ".join(current_units).strip()
            start_char = content.find(chunk_text[:30], search_start)
            if start_char == -1:
                start_char = content.find(chunk_text[:30])
                if start_char == -1:
                    start_char = search_start
            end_char = start_char + len(chunk_text)

            chunks.append(
                self.create_chunk(
                    document=document,
                    chunk_index=chunk_idx,
                    content=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    strategy="semantic",
                    metadata={
                        "units_count": len(current_units),
                        "similarity_threshold": self.similarity_threshold,
                    },
                )
            )

        return chunks

    def _chunk_with_vectors(
        self,
        document: RawDocument,
        content: str,
        units: list[str],
        vectors: list[list[float]],
    ) -> list[DocumentChunk]:
        """Chunking using dense embedding vectors."""
        chunks: list[DocumentChunk] = []
        current_units: list[str] = [units[0]]
        current_tokens = self.count_tokens(units[0])
        chunk_idx = 0
        search_start = 0

        for i in range(1, len(units)):
            sim = self._vector_cosine_sim(vectors[i - 1], vectors[i])
            next_tokens = self.count_tokens(units[i])

            semantic_split_triggered = (
                sim < self.similarity_threshold and current_tokens >= self.min_chunk_tokens
            )
            size_limit_reached = (current_tokens + next_tokens) > self.max_chunk_tokens

            if semantic_split_triggered or size_limit_reached:
                chunk_text = " ".join(current_units).strip()
                start_char = content.find(chunk_text[:30], search_start)
                if start_char == -1:
                    start_char = content.find(chunk_text[:30])
                    if start_char == -1:
                        start_char = search_start
                end_char = start_char + len(chunk_text)
                search_start = max(0, start_char + len(chunk_text) // 2)

                chunks.append(
                    self.create_chunk(
                        document=document,
                        chunk_index=chunk_idx,
                        content=chunk_text,
                        start_char=start_char,
                        end_char=end_char,
                        strategy="semantic",
                        metadata={
                            "units_count": len(current_units),
                            "similarity_threshold": self.similarity_threshold,
                        },
                    )
                )
                chunk_idx += 1
                current_units = [units[i]]
                current_tokens = next_tokens
            else:
                current_units.append(units[i])
                current_tokens += next_tokens

        if current_units:
            chunk_text = " ".join(current_units).strip()
            start_char = content.find(chunk_text[:30], search_start)
            if start_char == -1:
                start_char = content.find(chunk_text[:30])
                if start_char == -1:
                    start_char = search_start
            end_char = start_char + len(chunk_text)

            chunks.append(
                self.create_chunk(
                    document=document,
                    chunk_index=chunk_idx,
                    content=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    strategy="semantic",
                    metadata={
                        "units_count": len(current_units),
                        "similarity_threshold": self.similarity_threshold,
                    },
                )
            )

        return chunks

    @staticmethod
    def _vector_cosine_sim(v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity between two float vectors."""
        dot = sum(a * b for a, b in zip(v1, v2, strict=False))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return dot / (n1 * n2)
