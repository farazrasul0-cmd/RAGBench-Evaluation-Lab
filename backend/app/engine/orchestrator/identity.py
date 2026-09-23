"""Dataset-version-aware cache identity computation for RAGBench experiments."""

import hashlib

from pydantic import BaseModel, Field


class CacheIdentity(BaseModel):
    """Encapsulates dataset-version-aware cache identity."""

    dataset_id: str = Field(description="Unique dataset identifier")
    dataset_version_hash: str = Field(description="Cryptographic hash of the dataset version")
    pipeline_config_hash: str = Field(description="SHA-256 hash of the pipeline configuration")
    cache_key: str = Field(description="Formatted composite key")
    cache_hash: str = Field(description="SHA-256 digest of the composite cache key")


def compute_cache_identity(
    dataset_id: str,
    dataset_version_hash: str,
    pipeline_config_hash: str,
) -> CacheIdentity:
    """Compute dataset-version-aware cache identity for a pipeline execution.

    A pipeline configuration hash alone is insufficient for caching because the underlying
    dataset content may change across versions. The cache identity binds the dataset identifier,
    dataset version hash, and pipeline configuration hash together.
    """
    clean_dataset_id = dataset_id.strip()
    clean_version_hash = dataset_version_hash.strip()
    clean_config_hash = pipeline_config_hash.strip()

    if not clean_dataset_id:
        raise ValueError("dataset_id cannot be empty")
    if not clean_version_hash:
        raise ValueError("dataset_version_hash cannot be empty")
    if not clean_config_hash:
        raise ValueError("pipeline_config_hash cannot be empty")

    cache_key = f"{clean_dataset_id}@{clean_version_hash}:{clean_config_hash}"
    cache_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()

    return CacheIdentity(
        dataset_id=clean_dataset_id,
        dataset_version_hash=clean_version_hash,
        pipeline_config_hash=clean_config_hash,
        cache_key=cache_key,
        cache_hash=cache_hash,
    )
