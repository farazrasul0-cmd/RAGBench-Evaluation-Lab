"""Recursive character chunker splitting on hierarchical separators."""

from collections.abc import Sequence

from app.core.exceptions import ChunkingError
from app.engine.chunkers.base import BaseChunker
from app.schemas.chunk import DocumentChunk
from app.schemas.document import RawDocument


class RecursiveCharacterChunker(BaseChunker):
    """Hierarchically splits text on semantic separators (paragraphs, lines, sentences)."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Sequence[str] | None = None,
    ) -> None:
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
        self.separators = list(separators or self.DEFAULT_SEPARATORS)

    def chunk(self, document: RawDocument) -> list[DocumentChunk]:
        """Hierarchically segment text into token-budgeted chunks."""
        content = document.content.strip()
        if not content:
            return []

        raw_splits = self._split_text(content, self.separators)
        combined_texts = self._merge_splits(raw_splits)

        chunks: list[DocumentChunk] = []
        search_start = 0

        for idx, text_block in enumerate(combined_texts):
            text_block_clean = text_block.strip()
            if not text_block_clean:
                continue

            start_char = content.find(text_block_clean[:30], search_start)
            if start_char == -1:
                start_char = content.find(text_block_clean[:30])
                if start_char == -1:
                    start_char = search_start
            end_char = start_char + len(text_block_clean)
            search_start = max(0, start_char + len(text_block_clean) // 2)

            chunks.append(
                self.create_chunk(
                    document=document,
                    chunk_index=idx,
                    content=text_block_clean,
                    start_char=start_char,
                    end_char=end_char,
                    strategy="recursive",
                    metadata={
                        "chunk_size_tokens": self.chunk_size,
                        "chunk_overlap_tokens": self.chunk_overlap,
                    },
                )
            )

        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the first available separator."""
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators: list[str] = []

        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        for s in splits:
            if not s:
                continue
            token_count = self.count_tokens(s)
            if token_count <= self.chunk_size:
                final_chunks.append(s)
            elif new_separators:
                final_chunks.extend(self._split_text(s, new_separators))
            else:
                final_chunks.append(s)

        return final_chunks

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Combine smaller split fragments up to chunk_size tokens with overlap."""
        docs: list[str] = []
        current_doc: list[str] = []
        total_tokens = 0

        for piece in splits:
            piece_tokens = self.count_tokens(piece)
            if total_tokens + piece_tokens > self.chunk_size and current_doc:
                combined = " ".join(current_doc).strip()
                if combined:
                    docs.append(combined)

                # Keep overlap pieces
                overlap_doc: list[str] = []
                overlap_tokens = 0
                for back_piece in reversed(current_doc):
                    back_len = self.count_tokens(back_piece)
                    if overlap_tokens + back_len <= self.chunk_overlap:
                        overlap_doc.insert(0, back_piece)
                        overlap_tokens += back_len
                    else:
                        break

                current_doc = overlap_doc
                total_tokens = overlap_tokens

            current_doc.append(piece)
            total_tokens += piece_tokens

        if current_doc:
            combined = " ".join(current_doc).strip()
            if combined:
                docs.append(combined)

        return docs
