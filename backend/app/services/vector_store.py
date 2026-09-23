"""Vector store adapters for Qdrant and in-memory execution."""

import hashlib
import json
import uuid
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.exceptions import RAGBenchError
from app.schemas.chunk import DocumentChunk


class VectorStoreError(RAGBenchError):
    """Raised when vector database operations fail."""


def compute_collection_name(
    dataset_id: str,
    chunking_config: dict[str, Any],
    embedding_model_name: str,
) -> str:
    """Generate collection name conforming to SYSTEM_ARCHITECTURE.md contract:

    ragbench_{dataset_id}_{chunking_hash}_{embedding_model_hash}
    """
    clean_ds = "".join(c if c.isalnum() else "_" for c in dataset_id.lower())
    canonical_chunking = json.dumps(chunking_config, sort_keys=True)
    c_hash = hashlib.sha256(canonical_chunking.encode("utf-8")).hexdigest()[:8]
    e_hash = hashlib.sha256(embedding_model_name.encode("utf-8")).hexdigest()[:8]
    return f"ragbench_{clean_ds}_{c_hash}_{e_hash}"


class VectorStoreAdapter(ABC):
    """Abstract interface for dense vector persistence and similarity retrieval."""

    @abstractmethod
    def create_collection(
        self, collection_name: str, vector_size: int, distance: str = "Cosine"
    ) -> None:
        """Create a collection configured with distance metric and vector dimensionality."""

    @abstractmethod
    def upsert_chunks(
        self, collection_name: str, chunks: list[DocumentChunk], vectors: list[list[float]]
    ) -> int:
        """Batch upsert chunks alongside their dense vectors."""

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Retrieve top-K most similar chunks with similarity scores."""

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check whether collection exists."""

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete an existing collection."""

    @abstractmethod
    def get_collection_count(self, collection_name: str) -> int:
        """Return total number of points in collection."""


class QdrantVectorStore(VectorStoreAdapter):
    """Production Qdrant adapter supporting in-memory mode, local storage, and remote server."""

    TOP_LEVEL_FIELDS = {"doc_id", "chunk_id", "chunk_index", "strategy"}

    def __init__(
        self,
        location: str = ":memory:",
        url: str | None = None,
        api_key: str | None = None,
        path: str | None = None,
    ) -> None:
        self.is_memory = location == ":memory:" and not url and not path
        try:
            if url:
                self.client = QdrantClient(url=url, api_key=api_key)
            elif path:
                self.client = QdrantClient(path=path)
            else:
                self.client = QdrantClient(location=location)
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize QdrantClient: {e}") from e

    def create_collection(
        self, collection_name: str, vector_size: int, distance: str = "Cosine"
    ) -> None:
        """Create HNSW indexed Qdrant collection with payload indexing."""
        dist_enum = (
            models.Distance.COSINE if distance.lower() == "cosine" else models.Distance.EUCLID
        )
        try:
            if not self.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=dist_enum,
                        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
                    ),
                )
                if not self.is_memory:
                    # Create payload indexes on remote/persistent Qdrant
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        for field_name in ["doc_id", "strategy", "metadata.domain"]:
                            self.client.create_payload_index(
                                collection_name=collection_name,
                                field_name=field_name,
                                field_schema=models.PayloadSchemaType.KEYWORD,
                            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to create Qdrant collection '{collection_name}': {e}"
            ) from e

    def collection_exists(self, collection_name: str) -> bool:
        """Check existence of collection in Qdrant."""
        try:
            return bool(self.client.collection_exists(collection_name=collection_name))
        except Exception as e:
            raise VectorStoreError(f"Error checking collection existence: {e}") from e

    def delete_collection(self, collection_name: str) -> None:
        """Delete collection from Qdrant."""
        try:
            if self.collection_exists(collection_name):
                self.client.delete_collection(collection_name=collection_name)
        except Exception as e:
            raise VectorStoreError(f"Failed to delete collection '{collection_name}': {e}") from e

    def get_collection_count(self, collection_name: str) -> int:
        """Return point count in Qdrant collection."""
        try:
            res = self.client.count(collection_name=collection_name, exact=True)
            return int(res.count)
        except Exception as e:
            raise VectorStoreError(f"Failed to count points in '{collection_name}': {e}") from e

    def upsert_chunks(
        self, collection_name: str, chunks: list[DocumentChunk], vectors: list[list[float]]
    ) -> int:
        """Batch upsert points with chunk payload into Qdrant."""
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"Chunks count ({len(chunks)}) does not match vectors count ({len(vectors)})"
            )
        if not chunks:
            return 0

        points: list[models.PointStruct] = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload = chunk.model_dump()
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                )
            )

        try:
            batch_size = 500
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(collection_name=collection_name, points=batch, wait=True)
            return len(points)
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert points into '{collection_name}': {e}") from e

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search top-K most similar points in Qdrant with payload filtering."""
        query_filter: models.Filter | None = None
        if filter_metadata:
            must_conditions: list[models.Condition] = []
            for k, v in filter_metadata.items():
                # Support matching both top-level and metadata nested fields
                if k in self.TOP_LEVEL_FIELDS:
                    must_conditions.append(
                        models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    )
                else:
                    must_conditions.append(
                        models.Filter(
                            should=[
                                models.FieldCondition(key=k, match=models.MatchValue(value=v)),
                                models.FieldCondition(
                                    key=f"metadata.{k}", match=models.MatchValue(value=v)
                                ),
                            ]
                        )
                    )
            if must_conditions:
                query_filter = models.Filter(must=must_conditions)

        try:
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            hits: list[tuple[DocumentChunk, float]] = []
            for point in search_result.points:
                if point.payload:
                    chunk = DocumentChunk.model_validate(point.payload)
                    score = float(point.score) if point.score is not None else 0.0
                    hits.append((chunk, score))
            return hits
        except Exception as e:
            raise VectorStoreError(f"Search failed in collection '{collection_name}': {e}") from e


class InMemoryVectorStore(VectorStoreAdapter):
    """Pure NumPy-based in-memory vector store for offline testing and fast CI validation."""

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, Any]] = {}

    def create_collection(
        self, collection_name: str, vector_size: int, distance: str = "Cosine"
    ) -> None:
        """Create an in-memory collection registry."""
        if collection_name not in self._collections:
            self._collections[collection_name] = {
                "vector_size": vector_size,
                "distance": distance,
                "chunks": [],
                "vectors": [],
            }

    def collection_exists(self, collection_name: str) -> bool:
        """Check if in-memory collection exists."""
        return collection_name in self._collections

    def delete_collection(self, collection_name: str) -> None:
        """Delete an in-memory collection."""
        self._collections.pop(collection_name, None)

    def get_collection_count(self, collection_name: str) -> int:
        """Count stored points in collection."""
        if collection_name not in self._collections:
            return 0
        return len(self._collections[collection_name]["chunks"])

    def upsert_chunks(
        self, collection_name: str, chunks: list[DocumentChunk], vectors: list[list[float]]
    ) -> int:
        """Store chunks and vectors in memory."""
        if collection_name not in self._collections:
            raise VectorStoreError(f"Collection '{collection_name}' does not exist")
        if len(chunks) != len(vectors):
            raise VectorStoreError("Chunks and vectors must have matching length")

        col = self._collections[collection_name]
        col["chunks"].extend(chunks)
        col["vectors"].extend(vectors)
        return len(chunks)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Compute cosine similarity ranking over in-memory points."""
        if collection_name not in self._collections:
            raise VectorStoreError(f"Collection '{collection_name}' does not exist")

        col = self._collections[collection_name]
        chunks: list[DocumentChunk] = col["chunks"]
        vectors: list[list[float]] = col["vectors"]

        if not chunks:
            return []

        # Filter points by metadata if requested
        valid_indices: list[int] = []
        for idx, chunk in enumerate(chunks):
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    val = getattr(chunk, k, None)
                    if val is None:
                        val = chunk.metadata.get(k)
                    if val != v:
                        match = False
                        break
                if match:
                    valid_indices.append(idx)
            else:
                valid_indices.append(idx)

        if not valid_indices:
            return []

        filtered_vectors = np.array([vectors[i] for i in valid_indices], dtype=np.float32)
        q_vec = np.array(query_vector, dtype=np.float32)

        # Cosine similarity
        q_norm = np.linalg.norm(q_vec)
        v_norms = np.linalg.norm(filtered_vectors, axis=1)

        denominator = v_norms * q_norm
        denominator[denominator == 0.0] = 1e-12

        scores = np.dot(filtered_vectors, q_vec) / denominator
        ranked_order = np.argsort(-scores)[:top_k]

        results: list[tuple[DocumentChunk, float]] = []
        for rank_idx in ranked_order:
            orig_idx = valid_indices[rank_idx]
            results.append((chunks[orig_idx], float(scores[rank_idx])))

        return results
