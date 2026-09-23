"""Fixed-size token window chunker with configurable overlap."""

from app.core.exceptions import ChunkingError
from app.engine.chunkers.base import BaseChunker
from app.schemas.chunk import DocumentChunk
from app.schemas.document import RawDocument


class FixedTokenChunker(BaseChunker):
    """Chunks text into uniform token-length windows with configurable token overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ChunkingError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ChunkingError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ChunkingError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: RawDocument) -> list[DocumentChunk]:
        """Segment document into fixed token windows with overlap."""
        content = document.content.strip()
        if not content:
            return []

        tokens = self.encode_tokens(content)
        total_tokens = len(tokens)

        if total_tokens <= self.chunk_size:
            return [
                self.create_chunk(
                    document=document,
                    chunk_index=0,
                    content=content,
                    start_char=0,
                    end_char=len(content),
                    strategy="fixed_token",
                    metadata={"total_document_tokens": total_tokens},
                )
            ]

        chunks: list[DocumentChunk] = []
        step = self.chunk_size - self.chunk_overlap
        chunk_idx = 0
        search_start = 0

        for start_token_idx in range(0, total_tokens, step):
            end_token_idx = min(start_token_idx + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start_token_idx:end_token_idx]
            chunk_text = self.decode_tokens(chunk_tokens).strip()

            if not chunk_text:
                continue

            # Locate character offsets in document content
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
                    strategy="fixed_token",
                    metadata={
                        "chunk_size_tokens": self.chunk_size,
                        "chunk_overlap_tokens": self.chunk_overlap,
                    },
                )
            )
            chunk_idx += 1

            if end_token_idx >= total_tokens:
                break

        return chunks
