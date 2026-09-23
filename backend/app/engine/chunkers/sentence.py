"""Sentence boundary chunker supporting multilingual punctuation."""

import re

from app.core.exceptions import ChunkingError
from app.engine.chunkers.base import BaseChunker
from app.schemas.chunk import DocumentChunk
from app.schemas.document import RawDocument


class SentenceBoundaryChunker(BaseChunker):
    """Splits documents on sentence boundaries without truncating statements mid-sentence."""

    # Matches sentence endings: Western (. ! ?) and Bengali/Indic (। Dari / danda)
    SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?\u0964])\s+")

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
        """Group complete sentences into token-bounded chunks."""
        content = document.content.strip()
        if not content:
            return []

        raw_sentences = self.SENTENCE_SPLIT_REGEX.split(content)
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if not sentences:
            return []

        chunks: list[DocumentChunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        search_start = 0
        chunk_idx = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If adding this sentence exceeds chunk_size and we already have sentences
            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences).strip()
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
                        strategy="sentence",
                        metadata={
                            "sentence_count": len(current_sentences),
                            "chunk_size_tokens": self.chunk_size,
                        },
                    )
                )
                chunk_idx += 1

                # Carry over overlap sentences
                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for back_sentence in reversed(current_sentences):
                    st = self.count_tokens(back_sentence)
                    if overlap_tokens + st <= self.chunk_overlap:
                        overlap_sentences.insert(0, back_sentence)
                        overlap_tokens += st
                    else:
                        break

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Final remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
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
                    strategy="sentence",
                    metadata={
                        "sentence_count": len(current_sentences),
                        "chunk_size_tokens": self.chunk_size,
                    },
                )
            )

        return chunks
