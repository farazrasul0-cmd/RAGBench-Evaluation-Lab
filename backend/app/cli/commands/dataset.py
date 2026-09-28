"""Dataset CLI commands: register and list."""

import asyncio
import hashlib
import sys
from pathlib import Path
from typing import Any

import click

from app.cli.formatting import render_table
from app.db.repositories.dataset import DatasetRepository
from app.db.session import get_async_engine, get_session_factory, init_db


@click.group(name="dataset")
def dataset_group() -> None:
    """Manage local dataset corpora, versions, documents, and chunks."""


@dataset_group.command(name="register")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--name", "-n", default=None, help="Name of dataset (defaults to folder/file name).")
@click.option("--version", "-v", default=1, type=int, help="Version number (defaults to 1).")
@click.option("--chunk-size", default=512, type=int, help="Target chunk token/character size.")
@click.option("--db-url", default=None, help="Database connection URL override.")
def register_command(
    path: Path,
    name: str | None,
    version: int,
    chunk_size: int,
    db_url: str | None,
) -> None:
    """Register and snapshot local document files into SQLite dataset version."""
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

    async def _register() -> int:
        engine = get_async_engine(db_url)
        await init_db(engine)
        session_factory = get_session_factory(engine)

        # Ingest file contents
        doc_payloads = []
        full_content_bytes = bytearray()

        for f in files:
            content = f.read_text(encoding="utf-8", errors="replace")
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            full_content_bytes.extend(content.encode("utf-8"))
            doc_payloads.append(
                {
                    "filename": f.name,
                    "content": content,
                    "content_hash": content_hash,
                }
            )

        ver_content_hash = hashlib.sha256(full_content_bytes).hexdigest()

        async with session_factory() as session:
            dataset_repo = DatasetRepository(session)

            # Check if dataset already exists
            ds = await dataset_repo.get_dataset_by_name(dataset_name)
            if not ds:
                ds = await dataset_repo.get_dataset(dataset_name)
            if not ds:
                ds = await dataset_repo.create_dataset(name=dataset_name)

            ver = await dataset_repo.create_version(
                dataset_id=ds.id,
                version_number=version,
                content_hash=ver_content_hash,
            )

            # Add documents
            db_docs = await dataset_repo.add_documents(ver.id, doc_payloads)

            total_chunks = 0
            for doc_entity, p_info in zip(db_docs, doc_payloads, strict=False):
                # Simple fixed-size chunking
                text = p_info["content"]
                chunks_data = []
                starts = list(range(0, max(len(text), 1), chunk_size))
                for c_idx, start in enumerate(starts):
                    chunk_text = text[start : start + chunk_size]
                    c_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                    c_id = f"{doc_entity.id}-c{c_idx}"
                    chunks_data.append(
                        {
                            "id": c_id,
                            "chunk_index": c_idx,
                            "content": chunk_text,
                            "content_hash": c_hash,
                            "token_count": len(chunk_text.split()),
                            "strategy": "fixed",
                            "chunking_config_hash": "cfg-fixed",
                        }
                    )

                await dataset_repo.add_chunks(doc_entity.id, chunks_data)
                total_chunks += len(chunks_data)

            await session.commit()

            click.echo(f"\nSuccessfully registered dataset '{dataset_name}':")
            click.echo(f"  Dataset ID:         {ds.id}")
            click.echo(f"  Dataset Version ID: {ver.id} (Version {ver.version_number})")
            click.echo(f"  Content Hash:       {ver.content_hash[:16]}...")
            click.echo(f"  Documents:          {len(db_docs)}")
            click.echo(f"  Total Chunks:       {total_chunks}")

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
