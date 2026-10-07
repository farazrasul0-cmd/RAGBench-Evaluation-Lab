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

    # CORS Allowed Origins
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

    # Storage paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = base_dir / "data"
    results_dir: Path = base_dir / "results"
    qdrant_storage_path: Path = base_dir / "qdrant_data"


settings = Settings()
