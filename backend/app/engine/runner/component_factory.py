"""Component factory constructing concrete Phase A-E instances for single pipeline execution."""

from dataclasses import dataclass
from typing import Any

from app.engine.context.builder import ContextBuilder
from app.engine.embeddings import get_embedding_provider
from app.engine.embeddings.base import BaseEmbeddingProvider
from app.engine.query_transforms import (
    BaseLLMClient,
    BaseQueryTransformer,
    IdentityTransformer,
    MockLLMClient,
    get_query_transformer,
)
from app.engine.rerankers import BaseReranker, DeterministicMockReranker, get_reranker
from app.engine.retrievers.base import BaseRetriever
from app.engine.retrievers.bm25 import BM25Retriever
from app.engine.retrievers.dense import DenseRetriever
from app.engine.retrievers.hybrid import HybridRetriever
from app.schemas.chunk import DocumentChunk
from app.schemas.experiment import PipelineConfig
from app.services.vector_store import QdrantVectorStore


@dataclass
class PipelineComponents:
    """Encapsulates all instantiated components required to execute a single pipeline run."""

    retriever: BaseRetriever
    reranker: BaseReranker | None
    transformer: BaseQueryTransformer
    context_builder: ContextBuilder
    llm_client: BaseLLMClient


def build_pipeline_components(
    config: PipelineConfig,
    corpus_chunks: list[DocumentChunk],
    llm_client: BaseLLMClient | None = None,
    embedding_provider: BaseEmbeddingProvider | None = None,
) -> PipelineComponents:
    """Instantiate and wire together all pipeline components declared in PipelineConfig.

    Args:
        config: Strongly typed pipeline configuration specifying component hyperparameters.
        corpus_chunks: In-memory or persisted corpus passages to index for retrieval.
        llm_client: Optional custom LLM client (defaults to MockLLMClient if not supplied).
        embedding_provider: Optional custom embedding provider (defaults to factory lookup).

    Returns:
        PipelineComponents bundle containing retriever, reranker,
        transformer, context builder, and LLM.
    """
    # 1. LLM Client
    effective_llm = llm_client or MockLLMClient()

    # 2. Query Transformer
    transform_strategy = config.query_transform.strategy.lower().strip()
    if transform_strategy in ("none", "identity", ""):
        transformer: BaseQueryTransformer = IdentityTransformer()
    elif transform_strategy == "hyde":
        kwargs: dict[str, Any] = {}
        if config.query_transform.prompt_template:
            kwargs["prompt_template"] = config.query_transform.prompt_template
        transformer = get_query_transformer("hyde", llm_client=effective_llm, **kwargs)
    elif transform_strategy in ("multi_query", "multiquery"):
        mq_kwargs: dict[str, Any] = {
            "num_queries": config.query_transform.num_queries,
            "include_original": config.query_transform.include_original,
        }
        if config.query_transform.prompt_template:
            mq_kwargs["prompt_template"] = config.query_transform.prompt_template
        transformer = get_query_transformer("multi_query", llm_client=effective_llm, **mq_kwargs)
    elif transform_strategy in ("step_back", "stepback"):
        sb_kwargs: dict[str, Any] = {
            "include_original": config.query_transform.include_original,
        }
        if config.query_transform.prompt_template:
            sb_kwargs["prompt_template"] = config.query_transform.prompt_template
        transformer = get_query_transformer("step_back", llm_client=effective_llm, **sb_kwargs)
    else:
        transformer = IdentityTransformer()

    # 3. Base Retriever Topology (BM25, Dense, or Hybrid)
    retrieval_mode = config.retrieval.mode.lower().strip()

    bm25: BM25Retriever | None = None
    if retrieval_mode in ("bm25", "hybrid"):
        bm25 = BM25Retriever(chunks=corpus_chunks if corpus_chunks else None)

    dense: DenseRetriever | None = None
    if retrieval_mode in ("dense", "hybrid"):
        emb = embedding_provider or get_embedding_provider(
            provider=config.embedding.provider,
            model_name=config.embedding.model_name,
        )
        vector_store = QdrantVectorStore(location=":memory:")
        clean_ds_id = config.dataset.dataset_id.replace("-", "_")
        clean_ver_id = config.dataset.dataset_version_id.replace("-", "_")
        coll_name = f"eval_{clean_ds_id}_{clean_ver_id}"
        vector_store.create_collection(collection_name=coll_name, vector_size=emb.dimension)
        if corpus_chunks:
            vectors = emb.embed_texts([c.content for c in corpus_chunks])
            vector_store.upsert_chunks(coll_name, corpus_chunks, vectors)
        dense = DenseRetriever(
            embedding_provider=emb,
            vector_store=vector_store,
            collection_name=coll_name,
        )

    base_retriever: BaseRetriever
    if retrieval_mode == "bm25":
        if bm25 is None:
            raise ValueError("BM25 retriever failed to initialize")
        base_retriever = bm25
    elif retrieval_mode == "dense":
        if dense is None:
            raise ValueError("Dense retriever failed to initialize")
        base_retriever = dense
    elif retrieval_mode == "hybrid":
        if dense is None or bm25 is None:
            raise ValueError("Hybrid retriever requires both Dense and BM25 instances")
        base_retriever = HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            fusion_method=config.retrieval.hybrid_fusion,
            rrf_k=config.retrieval.rrf_k,
            alpha=config.retrieval.dense_weight,
        )
    else:
        raise ValueError(f"Unsupported retrieval mode: '{config.retrieval.mode}'")

    # 4. Passage Reranker
    reranker: BaseReranker | None = None
    if config.reranker.enabled:
        r_strat = config.reranker.strategy.lower().strip()
        if r_strat in ("mock", "test"):
            reranker = DeterministicMockReranker()
        elif r_strat in ("cross_encoder", "flashrank"):
            rk_kwargs: dict[str, Any] = {}
            if config.reranker.model_name:
                rk_kwargs["model_name"] = config.reranker.model_name
            reranker = get_reranker(r_strat, **rk_kwargs)

    # 5. Context Builder
    context_builder = ContextBuilder(
        token_budget=config.context.token_budget,
        reorder_strategy=config.context.reorder_strategy,
        encoding_name=config.context.encoding_name,
    )

    return PipelineComponents(
        retriever=base_retriever,
        reranker=reranker,
        transformer=transformer,
        context_builder=context_builder,
        llm_client=effective_llm,
    )
