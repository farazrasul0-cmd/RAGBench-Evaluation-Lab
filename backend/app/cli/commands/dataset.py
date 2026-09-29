"""Dataset CLI commands: register and list."""

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import click

from app.cli.formatting import render_table
from app.db.repositories.benchmark import BenchmarkRepository
from app.db.repositories.dataset import DatasetRepository
from app.db.session import get_async_engine, get_session_factory, init_db
from app.engine.benchmark.resolver import (
    AmbiguousPassageError,
    GroundTruthChunkResolver,
    UnresolvedPassageError,
)
from app.engine.chunkers.base import BaseChunker
from app.engine.chunkers.fixed_token import FixedTokenChunker
from app.engine.chunkers.recursive import RecursiveCharacterChunker
from app.engine.chunkers.sentence import SentenceBoundaryChunker
from app.engine.parsers import get_parser_for_file
from app.schemas.benchmark import BenchmarkQuery, BenchmarkQuerySet, GroundTruthPassage
from app.schemas.chunk import DocumentChunk
from app.schemas.document import RawDocument


@click.group(name="dataset")
def dataset_group() -> None:
    """Manage local dataset corpora, versions, documents, and chunks."""


@dataset_group.command(name="register")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--name", "-n", default=None, help="Name of dataset (defaults to folder/file name).")
@click.option("--version", "-v", default=1, type=int, help="Version number (defaults to 1).")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["fixed", "recursive", "sentence"], case_sensitive=False),
    default="fixed",
    help="Phase A chunking strategy.",
)
@click.option("--chunk-size", default=512, type=int, help="Target chunk token size.")
@click.option("--chunk-overlap", default=64, type=int, help="Target chunk overlap in tokens.")
@click.option(
    "--queries",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to JSON file containing benchmark evaluation queries.",
)
@click.option(
    "--eval-qa",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to JSONL file containing benchmark evaluation QA pairs.",
)
@click.option(
    "--allow-unresolved-passages",
    is_flag=True,
    default=False,
    help="Allow ground truth passages that cannot be mapped to any chunk.",
)
@click.option("--db-url", default=None, help="Database connection URL override.")
def register_command(
    path: Path,
    name: str | None,
    version: int,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    queries: Path | None,
    eval_qa: Path | None,
    allow_unresolved_passages: bool,
    db_url: str | None,
) -> None:
    """Register and snapshot local document files using Phase A canonical chunking."""
    dataset_name = name or path.stem

    files: list[Path] = []
    if path.is_file():
        files.append(path)
    else:
        for ext in ("*.txt", "*.md", "*.json"):
            files.extend(path.glob(ext))

    if not files:
        click.echo(f"No document files found in '{path}'.", err=True)
        sys.exit(1)

    # Load optional benchmark queries from legacy JSON file
    queries_data: list[dict[str, Any]] = []
    if queries is not None:
        try:
            with open(queries, encoding="utf-8") as qf:
                loaded_queries = json.load(qf)
                if not isinstance(loaded_queries, list):
                    click.echo(
                        "Error: Queries file must contain a JSON list of query objects.", err=True
                    )
                    sys.exit(1)
                queries_data = loaded_queries
        except Exception as exc:
            click.echo(f"Error reading queries file '{queries}': {exc}", err=True)
            sys.exit(1)

    # Load first-class benchmark QA pairs from JSONL file
    benchmark_query_set: BenchmarkQuerySet | None = None
    if eval_qa is not None:
        try:
            b_queries: list[BenchmarkQuery] = []
            with open(eval_qa, encoding="utf-8") as eqf:
                for line_idx, line in enumerate(eqf, start=1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    item = json.loads(line_str)
                    q_id = str(item.get("query_id") or item.get("id") or f"q_{line_idx}")
                    q_text = str(
                        item.get("query") or item.get("query_text") or item.get("question") or ""
                    )
                    gt_ans = str(
                        item.get("ground_truth_answer")
                        or item.get("answer")
                        or item.get("reference_answer")
                        or ""
                    )
                    passages_raw = item.get("ground_truth_passages") or item.get("passages") or []
                    passages: list[GroundTruthPassage] = []
                    for p in passages_raw:
                        if isinstance(p, dict):
                            passages.append(
                                GroundTruthPassage(
                                    doc_id=str(
                                        p.get("doc_id")
                                        or p.get("filename")
                                        or p.get("document_id")
                                        or ""
                                    ),
                                    passage_id=p.get("passage_id") or p.get("chunk_id"),
                                    text_snippet=str(
                                        p.get("text_snippet")
                                        or p.get("snippet")
                                        or p.get("text")
                                        or ""
                                    ),
                                    metadata={
                                        k: v
                                        for k, v in p.items()
                                        if k
                                        not in (
                                            "doc_id",
                                            "filename",
                                            "document_id",
                                            "passage_id",
                                            "chunk_id",
                                            "text_snippet",
                                            "snippet",
                                            "text",
                                        )
                                    },
                                )
                            )
                        elif isinstance(p, str):
                            passages.append(GroundTruthPassage(doc_id="", text_snippet=p))
                    b_queries.append(
                        BenchmarkQuery(
                            query_id=q_id,
                            query=q_text,
                            ground_truth_answer=gt_ans,
                            ground_truth_passages=passages,
                            domain=item.get("domain", "general"),
                            language=item.get("language", "en"),
                            difficulty=item.get("difficulty", "medium"),
                            metadata={
                                k: v
                                for k, v in item.items()
                                if k
                                not in (
                                    "query_id",
                                    "id",
                                    "query",
                                    "query_text",
                                    "question",
                                    "ground_truth_answer",
                                    "answer",
                                    "reference_answer",
                                    "ground_truth_passages",
                                    "passages",
                                    "domain",
                                    "language",
                                    "difficulty",
                                )
                            },
                        )
                    )
            benchmark_query_set = BenchmarkQuerySet(
                name=f"{dataset_name}_benchmark",
                version=version,
                queries=b_queries,
                allow_unresolved_passages=allow_unresolved_passages,
                metadata={"source_file": str(eval_qa)},
            )
        except Exception as exc:
            click.echo(f"Error reading benchmark QA file '{eval_qa}': {exc}", err=True)
            sys.exit(1)

    # Initialize Phase A Chunker
    strategy_lower = strategy.lower()
    overlap = max(0, min(chunk_overlap, chunk_size - 1)) if chunk_size > 1 else 0
    chunker: BaseChunker
    if strategy_lower == "recursive":
        chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    elif strategy_lower == "sentence":
        chunker = SentenceBoundaryChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    else:
        chunker = FixedTokenChunker(chunk_size=chunk_size, chunk_overlap=overlap)

    async def _register() -> int:
        engine = get_async_engine(db_url)
        await init_db(engine)
        session_factory = get_session_factory(engine)

        async with session_factory() as session:
            dataset_repo = DatasetRepository(session)

            # Check if dataset already exists by name or id
            ds = await dataset_repo.get_dataset_by_name(dataset_name)
            if not ds:
                ds = await dataset_repo.get_dataset(dataset_name)
            if not ds:
                ds = await dataset_repo.create_dataset(name=dataset_name)

            # Parse files strictly with Phase A parsers (no silent fallback)
            parsed_docs: list[RawDocument] = []
            full_content_bytes = bytearray()

            for f in sorted(files, key=lambda x: x.name):
                try:
                    parser = get_parser_for_file(f)
                    raw_doc = parser.parse(f)
                except Exception as exc:
                    click.echo(
                        f"Error parsing file '{f}': {exc}. Ingestion aborted.",
                        err=True,
                    )
                    return 1

                parsed_docs.append(raw_doc)
                full_content_bytes.extend(raw_doc.content.encode("utf-8"))

            ver_content_hash = hashlib.sha256(full_content_bytes).hexdigest()

            # Chunk all documents first and collect all chunks for passage mapping
            total_chunks = 0
            doc_chunk_map: list[tuple[RawDocument, list[DocumentChunk]]] = []
            all_chunks: list[DocumentChunk] = []

            for raw_doc in parsed_docs:
                chunks = chunker.chunk(raw_doc)
                doc_chunk_map.append((raw_doc, chunks))
                all_chunks.extend(chunks)
                total_chunks += len(chunks)

            # Validate source document references and persist BenchmarkVersion
            benchmark_hash: str | None = None
            if benchmark_query_set is not None:
                known_docs = (
                    {d.doc_id for d in parsed_docs}
                    | {d.filename for d in parsed_docs}
                    | {Path(d.filename).name for d in parsed_docs}
                    | {Path(d.filename).stem for d in parsed_docs}
                )
                for b_q in benchmark_query_set.queries:
                    for passage in b_q.ground_truth_passages:
                        if passage.doc_id:
                            matched_doc = (
                                passage.doc_id in known_docs
                                or any(passage.doc_id in kd for kd in known_docs)
                                or any(kd in passage.doc_id for kd in known_docs)
                            )
                            if not matched_doc and not allow_unresolved_passages:
                                click.echo(
                                    f"Error: Passage in query '{b_q.query_id}' is UNRESOLVED. "
                                    f"Unknown doc '{passage.doc_id}'. Ingestion aborted.",
                                    err=True,
                                )
                                return 1

                benchmark_repo = BenchmarkRepository(session)
                benchmark_hash = benchmark_query_set.compute_benchmark_hash()
                await benchmark_repo.create_benchmark(
                    benchmark_hash=benchmark_hash,
                    name=benchmark_query_set.name,
                    description=getattr(benchmark_query_set, "description", ""),
                    query_count=len(benchmark_query_set.queries),
                    allow_unresolved_passages=allow_unresolved_passages,
                    benchmark_data=benchmark_query_set.model_dump(),
                )

            # Create immutable dataset version snapshot with metadata
            version_metadata: dict[str, Any] = {
                "chunking_strategy": strategy_lower,
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
            }
            if queries_data:
                version_metadata["queries"] = queries_data
            elif benchmark_query_set is not None:
                # Dynamic passage-to-chunk resolution for this version's chunks
                resolver = GroundTruthChunkResolver()
                try:
                    resolved_queries = resolver.resolve_benchmark(benchmark_query_set, all_chunks)
                    version_metadata["queries"] = [q.model_dump() for q in resolved_queries]

                    # Annotate passages for this version snapshot
                    for b_q in benchmark_query_set.queries:
                        for passage in b_q.ground_truth_passages:
                            c_ids = resolver.resolve_chunks_for_passage(passage, all_chunks)
                            if c_ids:
                                passage.passage_id = c_ids[0]
                                passage.metadata["resolved_chunk_id"] = c_ids[0]
                            elif allow_unresolved_passages:
                                passage.metadata["resolved_chunk_id"] = None
                except AmbiguousPassageError as exc:
                    click.echo(f"Error: Ground truth passage is AMBIGUOUS: {exc}", err=True)
                    return 1
                except UnresolvedPassageError as exc:
                    click.echo(f"Error: Ground truth passage is UNRESOLVED: {exc}", err=True)
                    return 1
                except Exception as exc:
                    click.echo(f"Error resolving benchmark passages to chunks: {exc}", err=True)
                    return 1

            if benchmark_query_set is not None and benchmark_hash is not None:
                version_metadata["benchmark"] = benchmark_query_set.model_dump()
                version_metadata["benchmark_hash"] = benchmark_hash

            ver = await dataset_repo.create_version(
                dataset_id=ds.id,
                version_number=version,
                content_hash=ver_content_hash,
                metadata=version_metadata,
            )

            # Persist documents and canonical Phase A chunks
            for raw_doc, chunks in doc_chunk_map:
                db_docs = await dataset_repo.add_documents(
                    ver.id,
                    [
                        {
                            "filename": raw_doc.filename,
                            "content": raw_doc.content,
                            "content_hash": raw_doc.checksum,
                            "mime_type": raw_doc.metadata.get("mime_type", "text/plain"),
                            "size_bytes": len(raw_doc.content.encode("utf-8")),
                            "page_count": raw_doc.metadata.get("page_count", 1),
                            "metadata": raw_doc.metadata,
                        }
                    ],
                )
                db_doc = db_docs[0]

                chunks_payload = [
                    {
                        "id": c.chunk_id,
                        "chunk_index": c.chunk_index,
                        "content": c.content,
                        "content_hash": hashlib.sha256(c.content.encode("utf-8")).hexdigest(),
                        "token_count": c.token_count,
                        "char_count": c.char_count,
                        "start_char": c.start_char,
                        "end_char": c.end_char,
                        "strategy": c.strategy,
                        "chunking_config_hash": f"cfg-{c.strategy}-{chunk_size}",
                        "metadata": c.metadata,
                    }
                    for c in chunks
                ]

                if chunks_payload:
                    await dataset_repo.add_chunks(db_doc.id, chunks_payload)

            await session.commit()

            click.echo(f"\nSuccessfully registered dataset '{dataset_name}':")
            click.echo(f"  Dataset ID:         {ds.id}")
            click.echo(f"  Dataset Version ID: {ver.id} (Version {ver.version_number})")
            click.echo(f"  Content Hash:       {ver.content_hash[:16]}...")
            click.echo(
                f"  Chunking Strategy:  {strategy_lower} (size={chunk_size}, overlap={overlap})"
            )
            click.echo(f"  Documents:          {len(parsed_docs)}")
            click.echo(f"  Total Chunks:       {total_chunks}")
            if benchmark_query_set is not None:
                b_hash = version_metadata["benchmark_hash"]
                b_count = len(benchmark_query_set.queries)
                click.echo(f"  Benchmark Queries:  {b_count} (Hash: {b_hash[:16]}...)")
            elif queries_data:
                click.echo(f"  Benchmark Queries:  {len(queries_data)}")

        await engine.dispose()
        return 0

    code = asyncio.run(_register())
    sys.exit(code)


@dataset_group.command(name="list")
@click.option("--db-url", default=None, help="Database connection URL override.")
def list_command(db_url: str | None) -> None:
    """List all registered datasets and their versions."""

    async def _list() -> int:
        engine = get_async_engine(db_url)
        session_factory = get_session_factory(engine)

        async with session_factory() as session:
            dataset_repo = DatasetRepository(session)
            datasets = await dataset_repo.list_datasets()

            headers = ["Dataset ID", "Name", "Versions", "Created At"]
            rows: list[list[Any]] = []
            for ds in datasets:
                rows.append(
                    [
                        ds.id,
                        ds.name,
                        len(ds.versions),
                        str(ds.created_at)[:19],
                    ]
                )

            click.echo(render_table(headers, rows, title="Registered Datasets"))

        await engine.dispose()
        return 0

    code = asyncio.run(_list())
    sys.exit(code)
