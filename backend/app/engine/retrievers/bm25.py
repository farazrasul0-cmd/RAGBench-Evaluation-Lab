"""Lexical BM25 retrieval engine with multilingual tokenization support."""

import json
import re
import string
import unicodedata
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.core.exceptions import RAGBenchError
from app.engine.retrievers.base import BaseRetriever
from app.schemas.chunk import DocumentChunk


class BM25RetrieverError(RAGBenchError):
    """Raised when BM25 retrieval or indexing fails."""


_PUNCTUATION_SET = "".join(set(string.punctuation + "\u0964\u0965—‘’“”"))
_TOKEN_PATTERN = re.compile(rf"[^\s{re.escape(_PUNCTUATION_SET)}]+")


def tokenize_text(text: str) -> list[str]:
    """Tokenize text supporting English, Bengali, code symbols, and multilingual Unicode.

    Normalizes Unicode via NFKC, handles Bengali punctuation (e.g., । and ॥),
    converts to lower case, and splits on whitespace/punctuation boundaries.
    """
    normalized = unicodedata.normalize("NFKC", text.lower())
    tokens = _TOKEN_PATTERN.findall(normalized)
    return [t for t in tokens if t]


class BM25Retriever(BaseRetriever):
    """BM25Okapi lexical retrieval engine."""

    def __init__(self, chunks: list[DocumentChunk] | None = None) -> None:
        self._chunks: list[DocumentChunk] = []
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []
        if chunks is not None:
            self.index(chunks)

    @property
    def chunks(self) -> list[DocumentChunk]:
        """Return the list of indexed chunks."""
        return self._chunks

    def index(self, chunks: list[DocumentChunk]) -> None:
        """Index a collection of DocumentChunks for lexical search."""
        if not chunks:
            raise BM25RetrieverError("Cannot index an empty list of chunks.")

        self._chunks = list(chunks)
        self._corpus_tokens = [tokenize_text(c.content) for c in self._chunks]
        if not any(self._corpus_tokens):
            raise BM25RetrieverError("Corpus contains no valid tokens to index.")
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[DocumentChunk, float]]:
        """Retrieve top_k chunks using BM25Okapi scores with optional metadata filtering."""
        if not self._chunks or self._bm25 is None:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens)
        scored_pairs: list[tuple[DocumentChunk, float]] = []

        for chunk, score in zip(self._chunks, raw_scores, strict=True):
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue
            scored_pairs.append((chunk, float(score)))

        # Deterministic sort: descending by score, tie-break by ascending chunk_id
        scored_pairs.sort(key=lambda x: (-x[1], x[0].chunk_id))
        return scored_pairs[:top_k]

    def save(self, path: Path | str) -> None:
        """Serialize indexed retriever chunks to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [c.model_dump() for c in self._chunks],
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "BM25Retriever":
        """Load and reconstruct indexed retriever from disk."""
        target = Path(path)
        if not target.exists():
            raise BM25RetrieverError(f"BM25 index file not found: {target}")
        with open(target, encoding="utf-8") as f:
            payload = json.load(f)
        chunks = [DocumentChunk(**item) for item in payload.get("chunks", [])]
        return cls(chunks=chunks)
