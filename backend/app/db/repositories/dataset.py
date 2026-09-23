"""Repository for Dataset, DatasetVersion, Document, and Chunk persistence."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.entities import Dataset, DatasetVersion, Document, DocumentChunk


class DatasetRepository:
    """Async repository for managing datasets, snapshots, documents, and passages."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_dataset(
        self,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Dataset:
        """Create and persist a new dataset entity."""
        dataset = Dataset(
            name=name,
            description=description,
            metadata_json=metadata or {},
        )
        self.session.add(dataset)
        await self.session.flush()
        return dataset

    async def get_dataset(self, dataset_id: str) -> Dataset | None:
        """Fetch dataset by unique ID."""
        stmt = (
            select(Dataset).where(Dataset.id == dataset_id).options(selectinload(Dataset.versions))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_dataset_by_name(self, name: str) -> Dataset | None:
        """Fetch dataset by unique name."""
        stmt = select(Dataset).where(Dataset.name == name).options(selectinload(Dataset.versions))
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_version(
        self,
        dataset_id: str,
        version_number: int,
        content_hash: str,
        document_count: int = 0,
        total_bytes: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetVersion:
        """Create an immutable dataset version snapshot."""
        version = DatasetVersion(
            dataset_id=dataset_id,
            version_number=version_number,
            content_hash=content_hash,
            document_count=document_count,
            total_bytes=total_bytes,
            metadata_json=metadata or {},
        )
        self.session.add(version)
        await self.session.flush()

        # Update dataset current_version_id
        dataset = await self.get_dataset(dataset_id)
        if dataset:
            dataset.current_version_id = version.id

        return version

    async def get_version(self, version_id: str) -> DatasetVersion | None:
        """Fetch dataset version snapshot by ID."""
        stmt = select(DatasetVersion).where(DatasetVersion.id == version_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def add_documents(
        self,
        version_id: str,
        documents_data: list[dict[str, Any]],
    ) -> list[Document]:
        """Bulk insert documents for a dataset version."""
        documents: list[Document] = []
        for doc_data in documents_data:
            doc = Document(
                dataset_version_id=version_id,
                external_id=doc_data.get("external_id"),
                filename=doc_data["filename"],
                content=doc_data["content"],
                content_hash=doc_data["content_hash"],
                mime_type=doc_data.get("mime_type", "text/plain"),
                size_bytes=doc_data.get("size_bytes", len(doc_data["content"].encode("utf-8"))),
                page_count=doc_data.get("page_count", 1),
                metadata_json=doc_data.get("metadata", {}),
            )
            self.session.add(doc)
            documents.append(doc)

        await self.session.flush()

        # Update version document count and total bytes
        version = await self.get_version(version_id)
        if version:
            version.document_count += len(documents)
            version.total_bytes += sum(d.size_bytes for d in documents)

        return documents

    async def add_chunks(
        self,
        document_id: str,
        chunks_data: list[dict[str, Any]],
    ) -> list[DocumentChunk]:
        """Bulk insert document chunks for a document."""
        chunks: list[DocumentChunk] = []
        for c in chunks_data:
            chunk = DocumentChunk(
                id=c["id"],
                document_id=document_id,
                chunk_index=c["chunk_index"],
                content=c["content"],
                content_hash=c["content_hash"],
                token_count=c["token_count"],
                char_count=c.get("char_count", len(c["content"])),
                start_char=c.get("start_char", 0),
                end_char=c.get("end_char", len(c["content"])),
                strategy=c["strategy"],
                chunking_config_hash=c["chunking_config_hash"],
                metadata_json=c.get("metadata", {}),
            )
            self.session.add(chunk)
            chunks.append(chunk)

        await self.session.flush()
        return chunks

    async def get_document_chunks(self, document_id: str) -> list[DocumentChunk]:
        """Retrieve all ordered chunks belonging to a document."""
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
