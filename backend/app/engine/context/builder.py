"""Budget-constrained context window builder with citation injection and token guarding."""

import tiktoken

from app.core.exceptions import RAGBenchError
from app.engine.context.models import ContextChunk, PackedContext
from app.engine.context.reorder import reorder_lost_in_the_middle, reorder_standard
from app.schemas.chunk import DocumentChunk


class ContextBuilderError(RAGBenchError):
    """Raised when context building or token budget constraint fails."""


DEFAULT_HEADER_TEMPLATE = "[Source {idx}] (doc: {doc_id}, chunk: {chunk_id})\n{content}"


class ContextBuilder:
    """Sliding-window context builder enforcing exact token ceilings and citation injection."""

    def __init__(
        self,
        token_budget: int = 2048,
        reorder_strategy: str = "standard",
        encoding_name: str = "cl100k_base",
        header_template: str = DEFAULT_HEADER_TEMPLATE,
    ) -> None:
        if token_budget <= 0:
            raise ContextBuilderError("token_budget must be positive.")
        if reorder_strategy not in ("standard", "lost_in_the_middle"):
            raise ContextBuilderError(f"Unsupported reorder_strategy: {reorder_strategy}")

        self.token_budget = token_budget
        self.reorder_strategy = reorder_strategy
        self.encoding_name = encoding_name
        self.header_template = header_template

        try:
            self._encoder = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._encoder = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(self, text: str) -> int:
        """Count exact tokens using tiktoken encoder."""
        return len(self._encoder.encode(text))

    def format_chunk(self, idx: int, chunk: DocumentChunk) -> str:
        """Format chunk with structured citation header."""
        return self.header_template.format(
            idx=idx,
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            content=chunk.content.strip(),
        )

    def build(
        self,
        candidates: list[tuple[DocumentChunk, float]],
    ) -> PackedContext:
        """Pack candidates into context window respecting strict token budget."""
        if not candidates:
            return PackedContext(
                text="",
                total_tokens=0,
                token_budget=self.token_budget,
                strategy=self.reorder_strategy,
                chunks=[],
                included_chunks_count=0,
                truncated_chunks_count=0,
            )

        # 1. Greedy candidate selection by rank order
        selected: list[tuple[DocumentChunk, float, int]] = []
        cumulative_tokens = 0

        for rank_idx, (chunk, score) in enumerate(candidates, start=1):
            # Probe formatted snippet tokens
            candidate_text = self.format_chunk(idx=len(selected) + 1, chunk=chunk) + "\n\n"
            chunk_tokens = self.count_tokens(candidate_text)

            if cumulative_tokens + chunk_tokens <= self.token_budget:
                selected.append((chunk, score, rank_idx))
                cumulative_tokens += chunk_tokens
            else:
                # Token ceiling reached: cannot fit this candidate
                continue

        truncated_count = len(candidates) - len(selected)

        # 2. Re-ordering strategy
        if self.reorder_strategy == "lost_in_the_middle":
            ordered_triples = reorder_lost_in_the_middle(selected)
        else:
            ordered_triples = reorder_standard(selected)

        # 3. Format final text with sequential 1-based source indices
        context_chunks: list[ContextChunk] = []
        snippets: list[str] = []

        for source_idx, (chunk, score, orig_rank) in enumerate(ordered_triples, start=1):
            snippet = self.format_chunk(idx=source_idx, chunk=chunk)
            snippets.append(snippet)
            context_chunks.append(
                ContextChunk(
                    source_index=source_idx,
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    token_count=self.count_tokens(snippet),
                    score=score,
                    original_rank=orig_rank,
                    metadata=chunk.metadata,
                )
            )

        final_text = "\n\n".join(snippets)
        total_tokens = self.count_tokens(final_text)

        # Verification guard: strict token budget invariant
        if total_tokens > self.token_budget:
            raise ContextBuilderError(
                f"Context budget exceeded: {total_tokens} > {self.token_budget}"
            )

        return PackedContext(
            text=final_text,
            total_tokens=total_tokens,
            token_budget=self.token_budget,
            strategy=self.reorder_strategy,
            chunks=context_chunks,
            included_chunks_count=len(context_chunks),
            truncated_chunks_count=truncated_count,
        )
