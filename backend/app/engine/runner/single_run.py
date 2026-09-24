"""Single-Run Pipeline Executor orchestrating Phase A-E components and F3 relational persistence."""

import statistics
import time
from collections import defaultdict
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.dataset import DatasetRepository
from app.db.repositories.experiment import ExperimentRepository
from app.db.repositories.query_trace import QueryTraceRepository
from app.engine.embeddings.base import BaseEmbeddingProvider
from app.engine.metrics.citation import evaluate_citations
from app.engine.metrics.generation import (
    LLMEntailmentClassifier,
    MockEntailmentClassifier,
    evaluate_faithfulness,
)
from app.engine.metrics.ir import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from app.engine.orchestrator.identity import compute_cache_identity
from app.engine.query_transforms.base import BaseLLMClient
from app.engine.retrievers.hybrid import reciprocal_rank_fusion
from app.engine.runner.component_factory import build_pipeline_components
from app.engine.runner.models import EvaluationQuery, SingleRunResult
from app.schemas.chunk import DocumentChunk
from app.schemas.experiment import PipelineConfig


class SingleRunExecutor:
    """Orchestrates execution of a single concrete pipeline configuration."""

    async def execute(
        self,
        experiment_id: str,
        pipeline_config: PipelineConfig,
        queries: list[EvaluationQuery],
        corpus_chunks: list[DocumentChunk],
        session: AsyncSession,
        llm_client: BaseLLMClient | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        validate_chunk_references: bool = True,
        environment: str = "local",
        random_seed: int = 42,
        git_commit: str | None = None,
    ) -> SingleRunResult:
        """Execute a single pipeline configuration over the benchmark evaluation queries.

        Args:
            experiment_id: Unique identifier of the parent Experiment entity.
            pipeline_config: Fully resolved, concrete PipelineConfig to execute.
            queries: Benchmark evaluation queries with ground truth annotations.
            corpus_chunks: In-memory corpus chunks available for retrieval indexing.
            session: Active SQLAlchemy async session for persistence.
            llm_client: Optional custom LLM client for transforms, generation, and factuality.
            embedding_provider: Optional custom embedding provider.
            validate_chunk_references: Whether to enforce chunk reference integrity.
            environment: Execution environment descriptor (default 'local').
            random_seed: Random seed for deterministic reproducibility.
            git_commit: Optional git commit SHA recording code state.

        Returns:
            SingleRunResult containing run status, latency, query counts, and mean metric rollups.
        """
        run_start_time = time.perf_counter()

        exp_repo = ExperimentRepository(session)
        dataset_repo = DatasetRepository(session)
        trace_repo = QueryTraceRepository(session)

        # 1. Fetch Experiment and Dataset Version
        experiment = await exp_repo.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        dataset_version = await dataset_repo.get_version(experiment.dataset_version_id)
        if not dataset_version:
            raise ValueError(f"DatasetVersion '{experiment.dataset_version_id}' not found")

        # 2. Compute canonical hashes & Cache Identity
        pipeline_config_hash = pipeline_config.compute_configuration_hash()
        cache_id = compute_cache_identity(
            dataset_id=dataset_version.dataset_id,
            dataset_version_hash=dataset_version.content_hash,
            pipeline_config_hash=pipeline_config_hash,
        )

        # 3. Create ExperimentRun record in RUNNING status and commit run initialization
        run = await exp_repo.create_run(
            experiment_id=experiment.id,
            pipeline_config_hash=pipeline_config_hash,
            cache_key=cache_id.cache_key,
            cache_hash=cache_id.cache_hash,
            environment=environment,
            random_seed=random_seed,
            git_commit=git_commit,
        )
        await session.commit()

        # 4. Instantiate and wire Phase A-E pipeline components
        components = build_pipeline_components(
            config=pipeline_config,
            corpus_chunks=corpus_chunks,
            llm_client=llm_client,
            embedding_provider=embedding_provider,
        )

        # Maintain explicit chunk-to-document ID mapping to guarantee chunk_id != doc_id
        chunk_to_doc_map = {c.chunk_id: c.doc_id for c in corpus_chunks}

        # 5. Query execution loop with per-query atomic transactions
        all_metric_values: dict[str, list[float]] = defaultdict(list)
        completed_queries = 0
        failed_queries = 0

        for q in queries:
            q_start = time.perf_counter()
            try:
                # 5a. Query Transformation
                trans_res = components.transformer.transform(query=q.query_text)
                transformed_queries_data = [
                    {
                        "transformation_type": trans_res.strategy,
                        "query_text": sq,
                        "sequence_index": idx,
                        "metadata": {"original_query": q.query_text},
                    }
                    for idx, sq in enumerate(trans_res.transformed_queries)
                ]

                # 5b. Multi-channel or single Retrieval
                sub_queries = trans_res.transformed_queries or [q.query_text]
                if len(sub_queries) == 1:
                    retrieved_pairs = components.retriever.retrieve(
                        query=sub_queries[0],
                        top_k=pipeline_config.retrieval.top_k,
                    )
                else:
                    candidate_k = max(pipeline_config.retrieval.top_k * 2, 20)
                    rankings = [
                        components.retriever.retrieve(query=sq, top_k=candidate_k)
                        for sq in sub_queries
                    ]
                    retrieved_pairs = reciprocal_rank_fusion(
                        rankings=rankings,
                        k=pipeline_config.retrieval.rrf_k,
                        top_k=pipeline_config.retrieval.top_k,
                    )

                retrieved_chunks_data = [
                    {
                        "retriever_type": pipeline_config.retrieval.mode,
                        "chunk_id": chunk.chunk_id,
                        "rank": idx,
                        "score": float(score),
                        "metadata": chunk.metadata,
                    }
                    for idx, (chunk, score) in enumerate(retrieved_pairs, start=1)
                ]

                # 5c. Passage Reranking (preserves original retrieval ranking vs reranked ranking)
                if components.reranker is not None and retrieved_pairs:
                    candidates = [chunk for chunk, _ in retrieved_pairs]
                    orig_scores = {
                        chunk.chunk_id: (rank, score)
                        for rank, (chunk, score) in enumerate(retrieved_pairs, start=1)
                    }
                    reranked_pairs = components.reranker.rerank(
                        query=q.query_text,
                        candidates=candidates,
                        top_n=pipeline_config.reranker.top_n,
                    )
                    reranked_chunks_data = [
                        {
                            "chunk_id": chunk.chunk_id,
                            "original_rank": orig_scores.get(chunk.chunk_id, (0, 0.0))[0],
                            "reranked_rank": idx,
                            "original_score": float(orig_scores.get(chunk.chunk_id, (0, 0.0))[1]),
                            "reranker_score": float(score),
                            "metadata": chunk.metadata,
                        }
                        for idx, (chunk, score) in enumerate(reranked_pairs, start=1)
                    ]
                    final_pairs = reranked_pairs
                else:
                    reranked_chunks_data = []
                    final_pairs = retrieved_pairs

                # 5d. Context Window Formulation & Packing
                packed = components.context_builder.build(candidates=final_pairs)
                packed_context_data = {
                    "context_text": packed.text,
                    "token_count": packed.total_tokens,
                    "token_budget": packed.token_budget,
                    "ordering_strategy": packed.strategy,
                    "chunk_ids": [c.chunk_id for c in packed.chunks],
                    "metadata": {
                        "included_count": packed.included_chunks_count,
                        "truncated_count": packed.truncated_chunks_count,
                    },
                }

                # 5e. Downstream Generation
                gen_start = time.perf_counter()
                prompt = f"Context:\n{packed.text}\n\nQuestion: {q.query_text}\nAnswer:"
                answer = components.llm_client.generate(prompt=prompt)
                gen_latency_ms = (time.perf_counter() - gen_start) * 1000.0
                in_tokens = components.context_builder.count_tokens(prompt)
                out_tokens = components.context_builder.count_tokens(answer)
                generation_data = {
                    "model": pipeline_config.generation.model_name,
                    "answer": answer,
                    "prompt_version": "v1",
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "total_tokens": in_tokens + out_tokens,
                    "latency_ms": gen_latency_ms,
                    "estimated_cost": 0.0,
                    "metadata": {},
                }

                # 5f. Quantitative Evaluation Metrics
                # Distinct representations for chunk IDs vs document IDs
                retrieved_chunk_ids = [c.chunk_id for c, _ in final_pairs]
                retrieved_doc_ids = [
                    chunk_to_doc_map.get(c.chunk_id, c.doc_id) for c, _ in final_pairs
                ]

                gt_chunks = q.ground_truth_chunks
                gt_docs = q.ground_truth_docs
                configured_metrics = {m.lower().strip() for m in pipeline_config.evaluation.metrics}
                k_values = pipeline_config.evaluation.k_values or [1, 3, 5, 10, 20]

                metric_results_data: list[dict[str, Any]] = []

                # Chunk-level Information Retrieval metrics
                for k in k_values:
                    if "recall" in configured_metrics:
                        r_val = recall_at_k(retrieved_chunk_ids, gt_chunks, k)
                        metric_results_data.append(
                            {
                                "metric_name": f"recall@{k}",
                                "metric_value": r_val,
                                "metric_version": "v1",
                                "evaluator": "ir_chunk",
                            }
                        )
                        all_metric_values[f"recall@{k}"].append(r_val)

                    if "precision" in configured_metrics:
                        p_val = precision_at_k(retrieved_chunk_ids, gt_chunks, k)
                        metric_results_data.append(
                            {
                                "metric_name": f"precision@{k}",
                                "metric_value": p_val,
                                "metric_version": "v1",
                                "evaluator": "ir_chunk",
                            }
                        )
                        all_metric_values[f"precision@{k}"].append(p_val)

                    if "mrr" in configured_metrics:
                        mrr_val = reciprocal_rank_at_k(retrieved_chunk_ids, gt_chunks, k)
                        metric_results_data.append(
                            {
                                "metric_name": f"mrr@{k}",
                                "metric_value": mrr_val,
                                "metric_version": "v1",
                                "evaluator": "ir_chunk",
                            }
                        )
                        all_metric_values[f"mrr@{k}"].append(mrr_val)

                    if "ndcg" in configured_metrics:
                        ndcg_val = ndcg_at_k(retrieved_chunk_ids, gt_chunks, k)
                        metric_results_data.append(
                            {
                                "metric_name": f"ndcg@{k}",
                                "metric_value": ndcg_val,
                                "metric_version": "v1",
                                "evaluator": "ir_chunk",
                            }
                        )
                        all_metric_values[f"ndcg@{k}"].append(ndcg_val)

                    if "hit" in configured_metrics:
                        hit_val = hit_at_k(retrieved_chunk_ids, gt_chunks, k)
                        metric_results_data.append(
                            {
                                "metric_name": f"hit@{k}",
                                "metric_value": hit_val,
                                "metric_version": "v1",
                                "evaluator": "ir_chunk",
                            }
                        )
                        all_metric_values[f"hit@{k}"].append(hit_val)

                    # Document-level IR metrics (evaluated when ground_truth_docs is supplied)
                    if gt_docs:
                        doc_recall = recall_at_k(retrieved_doc_ids, gt_docs, k)
                        metric_results_data.append(
                            {
                                "metric_name": f"doc_recall@{k}",
                                "metric_value": doc_recall,
                                "metric_version": "v1",
                                "evaluator": "ir_doc",
                            }
                        )
                        all_metric_values[f"doc_recall@{k}"].append(doc_recall)

                # Factuality & Citation metrics
                classifier = (
                    MockEntailmentClassifier()
                    if pipeline_config.generation.model_name in ("mock", "test")
                    else LLMEntailmentClassifier(components.llm_client)
                )

                if "faithfulness" in configured_metrics:
                    faith_res = evaluate_faithfulness(
                        answer=answer,
                        context=packed.text,
                        classifier=classifier,
                        llm_client=components.llm_client,
                    )
                    metric_results_data.append(
                        {
                            "metric_name": "faithfulness",
                            "metric_value": faith_res.faithfulness_score,
                            "metric_version": "v1",
                            "evaluator": "factuality",
                            "metadata": {
                                "total_propositions": faith_res.total_propositions,
                                "entailed_propositions_count": (
                                    faith_res.entailed_propositions_count
                                ),
                            },
                        }
                    )
                    all_metric_values["faithfulness"].append(faith_res.faithfulness_score)

                if "citations" in configured_metrics:
                    context_sources = {c.source_index: c.content for c in packed.chunks}
                    cit_res = evaluate_citations(
                        answer=answer,
                        context_sources=context_sources,
                        classifier=classifier,
                    )
                    metric_results_data.append(
                        {
                            "metric_name": "citation_precision",
                            "metric_value": cit_res.citation_precision,
                            "metric_version": "v1",
                            "evaluator": "citation",
                        }
                    )
                    metric_results_data.append(
                        {
                            "metric_name": "citation_recall",
                            "metric_value": cit_res.citation_recall,
                            "metric_version": "v1",
                            "evaluator": "citation",
                        }
                    )
                    all_metric_values["citation_precision"].append(cit_res.citation_precision)
                    all_metric_values["citation_recall"].append(cit_res.citation_recall)

                q_latency_ms = (time.perf_counter() - q_start) * 1000.0

                query_metadata = {
                    **q.metadata,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "ground_truth_docs": gt_docs,
                }

                # 5g. Record query trace in savepoint, then commit query transaction
                async with session.begin_nested():
                    await trace_repo.record_query_trace(
                        experiment_run_id=run.id,
                        query_id=q.query_id,
                        original_query=q.query_text,
                        expected_answer=q.expected_answer,
                        ground_truth_chunks=q.ground_truth_chunks,
                        latency_ms=q_latency_ms,
                        status="SUCCESS",
                        metadata=query_metadata,
                        transformed_queries=transformed_queries_data,
                        retrieved_chunks=retrieved_chunks_data,
                        reranked_chunks=reranked_chunks_data,
                        packed_context=packed_context_data,
                        generation_result=generation_data,
                        metric_results=metric_results_data,
                        validate_chunk_references=validate_chunk_references,
                    )
                await session.commit()
                completed_queries += 1

            except Exception as exc:
                failed_queries += 1
                q_latency_ms = (time.perf_counter() - q_start) * 1000.0
                try:
                    async with session.begin_nested():
                        await trace_repo.record_query_trace(
                            experiment_run_id=run.id,
                            query_id=q.query_id,
                            original_query=q.query_text,
                            expected_answer=q.expected_answer,
                            ground_truth_chunks=q.ground_truth_chunks,
                            latency_ms=q_latency_ms,
                            status="FAILED",
                            metadata={"error": str(exc), **q.metadata},
                            validate_chunk_references=False,
                        )
                    await session.commit()
                except Exception:
                    await session.rollback()

        # 6. Aggregate metric summaries (computed strictly from successful observations)
        summaries_data: list[dict[str, Any]] = []
        mean_metrics_map: dict[str, float] = {}

        for m_name, vals in all_metric_values.items():
            if not vals:
                continue
            cnt = len(vals)
            mean_v = sum(vals) / cnt
            med_v = statistics.median(vals)
            min_v = min(vals)
            max_v = max(vals)
            # Sample standard deviation (Bessel's correction with N-1 degrees of freedom)
            std_v = statistics.stdev(vals) if cnt > 1 else 0.0

            summaries_data.append(
                {
                    "metric_name": m_name,
                    "mean": mean_v,
                    "median": med_v,
                    "min": min_v,
                    "max": max_v,
                    "stddev": std_v,
                    "count": cnt,
                    "metadata": {
                        "stddev_type": "sample",
                        "dof": max(cnt - 1, 0),
                    },
                }
            )
            mean_metrics_map[m_name] = mean_v

        if summaries_data:
            await exp_repo.add_metric_summaries(run.id, summaries_data)

        # 7. Finalize ExperimentRun status
        if failed_queries == 0:
            final_status = "COMPLETED"
            error_msg = None
        elif completed_queries == 0:
            final_status = "FAILED"
            error_msg = f"All {failed_queries} queries failed"
        else:
            final_status = "PARTIAL"
            error_msg = f"{failed_queries} of {len(queries)} queries failed"

        await exp_repo.complete_run(
            run_id=run.id,
            summary_metrics=mean_metrics_map,
            status=final_status,
            error=error_msg,
        )

        await session.commit()

        duration_ms = (time.perf_counter() - run_start_time) * 1000.0

        return SingleRunResult(
            experiment_run_id=run.id,
            experiment_id=experiment.id,
            pipeline_config_hash=pipeline_config_hash,
            cache_hash=cache_id.cache_hash,
            status=final_status,
            total_queries=len(queries),
            completed_queries=completed_queries,
            failed_queries=failed_queries,
            mean_metrics=mean_metrics_map,
            duration_ms=duration_ms,
        )
