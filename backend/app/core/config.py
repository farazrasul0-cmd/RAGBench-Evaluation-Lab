"""Application Configuration using Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global system configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    project_name: str = "RAGBench - Retrieval-Augmented Generation Evaluation Laboratory"
    version: str = "1.0.0-PROD"
    api_v1_str: str = "/api/v1"
    debug: bool = False

    # Storage paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = base_dir / "data"
    results_dir: Path = base_dir / "results"
    qdrant_storage_path: Path = base_dir / "qdrant_data"


settings = Settings()
