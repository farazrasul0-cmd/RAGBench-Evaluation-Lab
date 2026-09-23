"""Generation, faithfulness, and answer relevance metrics according to SYSTEM_ARCHITECTURE.md."""

import re
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, Field

from app.engine.embeddings.base import BaseEmbeddingProvider
from app.engine.query_transforms.base import BaseLLMClient


class FaithfulnessResult(BaseModel):
    """Structured result of proposition-level faithfulness evaluation."""

    faithfulness_score: float
    hallucination_rate: float
    total_propositions: int
    entailed_propositions_count: int
    propositions: list[str]
    entailed_flags: list[bool]
    reasoning: list[str] = Field(default_factory=list)


class BaseEntailmentClassifier(ABC):
    """Abstract interface for checking premise entails hypothesis (C ⊢ s)."""

    @abstractmethod
    def entails(self, context: str, proposition: str) -> tuple[bool, str]:
        """Return (is_entailed, reasoning)."""
        ...


class MockEntailmentClassifier(BaseEntailmentClassifier):
    """Deterministic mock entailment classifier for testing without external models."""

    def __init__(self, entailment_map: dict[str, bool] | None = None) -> None:
        self.entailment_map = entailment_map or {}

    def entails(self, context: str, proposition: str) -> tuple[bool, str]:
        # Exact lookup in mapping
        for key, val in self.entailment_map.items():
            if key in proposition:
                return val, f"Matched rule '{key}' -> {val}"

        # Default heuristic: check if major words from proposition appear in context
        words = [w.lower() for w in re.findall(r"\w+", proposition) if len(w) > 3]
        if not words:
            return True, "Empty claim"
        matches = sum(1 for w in words if w in context.lower())
        is_ent = (matches / len(words)) >= 0.5
        return is_ent, f"Word overlap match {matches}/{len(words)}"


class LLMEntailmentClassifier(BaseEntailmentClassifier):
    """Zero-shot LLM-as-a-judge entailment classifier."""

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    def entails(self, context: str, proposition: str) -> tuple[bool, str]:
        prompt = (
            "You are a strict factual consistency judge.\n"
            "Given Reference Context and a Statement, determine whether Statement is directly "
            "supported and entailed by the Context without assuming unstated facts.\n\n"
            f"Context: {context}\n\n"
            f"Statement: {proposition}\n\n"
            "Answer ONLY 'YES' if completely supported, or 'NO' if not supported or contradicts."
        )
        response = self.llm_client.generate(prompt=prompt).strip().upper()
        is_entailed = response.startswith("YES")
        return is_entailed, response


def decompose_propositions(
    answer: str,
    llm_client: BaseLLMClient | None = None,
) -> list[str]:
    """Decompose answer into atomic factual statements S_A = {s_1, ..., s_m}."""
    cleaned = answer.strip()
    if not cleaned:
        return []

    if llm_client is not None:
        prompt = (
            "Decompose the following answer into atomic, independent factual statements. "
            "Each statement must express exactly one factual claim.\n"
            "Provide each proposition on a separate line without bullets or numbers.\n\n"
            f"Answer: {cleaned}\n\n"
            "Propositions:"
        )
        try:
            resp = llm_client.generate(prompt=prompt)
            lines = [line.strip() for line in resp.splitlines() if line.strip()]
            cleaned_lines: list[str] = []
            for line in lines:
                c = re.sub(r"^[0-9]+[.\)]\s*", "", line)
                c = re.sub(r"^[-*•]\s*", "", c).strip()
                if c:
                    cleaned_lines.append(c)
            if cleaned_lines:
                return cleaned_lines
        except Exception:
            pass

    # Fallback: split by punctuation sentences
    sentences = re.split(r"(?<=[.!?\u0964\u0965])\s+", cleaned)
    return [s.strip() for s in sentences if s.strip()]


def evaluate_faithfulness(
    answer: str,
    context: str,
    classifier: BaseEntailmentClassifier,
    llm_client: BaseLLMClient | None = None,
) -> FaithfulnessResult:
    """Calculate Faithfulness(A, C) = |{s in S_A | C ⊢ s}| / |S_A|."""
    propositions = decompose_propositions(answer, llm_client=llm_client)

    if not propositions:
        return FaithfulnessResult(
            faithfulness_score=1.0,
            hallucination_rate=0.0,
            total_propositions=0,
            entailed_propositions_count=0,
            propositions=[],
            entailed_flags=[],
            reasoning=[],
        )

    entailed_flags: list[bool] = []
    reasoning_list: list[str] = []

    for prop in propositions:
        is_ent, reason = classifier.entails(context, prop)
        entailed_flags.append(is_ent)
        reasoning_list.append(reason)

    entailed_count = sum(1 for f in entailed_flags if f)
    faithfulness = float(entailed_count) / float(len(propositions))
    hallucination_rate = 1.0 - faithfulness

    return FaithfulnessResult(
        faithfulness_score=faithfulness,
        hallucination_rate=hallucination_rate,
        total_propositions=len(propositions),
        entailed_propositions_count=entailed_count,
        propositions=propositions,
        entailed_flags=entailed_flags,
        reasoning=reasoning_list,
    )


def calculate_answer_relevance(
    answer: str,
    original_query: str,
    embedding_provider: BaseEmbeddingProvider,
    llm_client: BaseLLMClient,
    num_synthetic_queries: int = 3,
) -> float:
    """Calculate Answer Relevance = 1/N sum_i cos(e(original_query), e(q_gen_i))."""
    if not answer.strip():
        return 0.0

    prompt = (
        f"Generate {num_synthetic_queries} distinct questions that the answer addresses.\n"
        f"Answer: {answer}\n\n"
        "Provide each question on a separate line without numbering."
    )
    resp = llm_client.generate(prompt=prompt)
    synthetic_queries = [line.strip() for line in resp.splitlines() if line.strip()][
        :num_synthetic_queries
    ]

    if not synthetic_queries:
        return 0.0

    query_vec = np.array(embedding_provider.embed_query(original_query), dtype=np.float32)
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0.0:
        return 0.0

    syn_vecs = np.array(embedding_provider.embed_texts(synthetic_queries), dtype=np.float32)
    similarities: list[float] = []

    for vec in syn_vecs:
        v_norm = np.linalg.norm(vec)
        if v_norm > 0.0:
            cos_sim = float(np.dot(query_vec, vec) / (q_norm * v_norm))
            similarities.append(max(0.0, cos_sim))
        else:
            similarities.append(0.0)

    return float(np.mean(similarities)) if similarities else 0.0
