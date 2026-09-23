"""Base chunker interface and utilities."""

import math
from abc import ABC, abstractmethod
from typing import Any

import tiktoken

from app.schemas.chunk import DocumentChunk
from app.schemas.document import RawDocument


class BaseChunker(ABC):
    """Abstract base class for all document chunking strategies."""

    def __init__(self) -> None:
        self._tokenizer: tiktoken.Encoding | None = None
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken cl100k_base with whitespace fallback."""
        if not text:
            return 0
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text, disallowed_special=()))
            except Exception:
                pass
        # Fallback approximation: ~1.3 tokens per word or ~4 chars per token
        words = text.split()
        return max(1, math.ceil(len(words) * 1.3))

    def encode_tokens(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if self._tokenizer is not None:
            try:
                return list(self._tokenizer.encode(text, disallowed_special=()))
            except Exception:
                pass
        # Fallback: pseudo-tokens based on character indices
        return [ord(c) for c in text]

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode token IDs back to text."""
        if self._tokenizer is not None:
            try:
                return self._tokenizer.decode(tokens)
            except Exception:
                pass
        return "".join(chr(t) for t in tokens if 0 <= t <= 0x10FFFF)

    def create_chunk(
        self,
        document: RawDocument,
        chunk_index: int,
        content: str,
        start_char: int,
        end_char: int,
        strategy: str,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentChunk:
        """Instantiate a DocumentChunk with deterministic ID calculation."""
        meta = dict(document.metadata)
        if metadata:
            meta.update(metadata)
        token_count = self.count_tokens(content)
        return DocumentChunk.create(
            doc_id=document.doc_id,
            chunk_index=chunk_index,
            content=content,
            token_count=token_count,
            start_char=start_char,
            end_char=end_char,
            strategy=strategy,
            metadata=meta,
        )

    @abstractmethod
    def chunk(self, document: RawDocument) -> list[DocumentChunk]:
        """Split a RawDocument into an ordered list of DocumentChunks."""
