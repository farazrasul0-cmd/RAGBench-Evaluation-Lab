"""Dataset CLI commands: register and list."""

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import click

from app.cli.formatting import render_table
from app.db.repositories.dataset import DatasetRepository
from app.db.session import get_async_engine, get_session_factory, init_db
from app.engine.chunkers.base import BaseChunker
from app.engine.chunkers.fixed_token import FixedTokenChunker
from app.engine.chunkers.recursive import RecursiveCharacterChunker
from app.engine.chunkers.sentence import SentenceBoundaryChunker
from app.engine.parsers import PlainTextParser, get_parser_for_file
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
@click.option("--db-url", default=None, help="Database connection URL override.")
def register_command(
    path: Path,
    name: str | None,
    version: int,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    queries: Path | None,
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

    # Load optional benchmark queries from JSON file
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

    # Initialize Phase A Chunker
    strategy_lower = strategy.lower()
    overlap = max(0, min(chunk_overlap, chunk_size - 1)) if chunk_size > 1 else 0
    chunker: BaseChunker
    if strategy_lower == "recursive":
        chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    elif strategy_lower == "sentence":
        chunker = SentenceBoundaryChunker()
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

            # Parse files with Phase A parsers and calculate aggregate version content hash
            parsed_docs: list[RawDocument] = []
            full_content_bytes = bytearray()

            for f in sorted(files, key=lambda x: x.name):
                try:
                    parser = get_parser_for_file(f)
                    raw_doc = parser.parse(f)
                except Exception:
                    raw_doc = PlainTextParser().parse(f)

                parsed_docs.append(raw_doc)
                full_content_bytes.extend(raw_doc.content.encode("utf-8"))

            ver_content_hash = hashlib.sha256(full_content_bytes).hexdigest()

            # Create immutable dataset version snapshot with metadata including benchmark queries
            version_metadata: dict[str, Any] = {
                "chunking_strategy": strategy_lower,
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
            }
            if queries_data:
                version_metadata["queries"] = queries_data

            ver = await dataset_repo.create_version(
                dataset_id=ds.id,
                version_number=version,
                content_hash=ver_content_hash,
                metadata=version_metadata,
            )

            # Persist documents and canonical Phase A chunks
            total_chunks = 0
            for raw_doc in parsed_docs:
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

                # Chunk using Phase A canonical chunker
                chunks = chunker.chunk(raw_doc)
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
                    total_chunks += len(chunks_payload)

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
            if queries_data:
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
