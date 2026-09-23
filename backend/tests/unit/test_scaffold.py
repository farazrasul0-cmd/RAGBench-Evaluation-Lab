"""Unit tests for Phase 0 scaffold verification."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Provide a TestClient instance."""
    return TestClient(app)


def test_settings_loaded() -> None:
    """Verify settings load with correct defaults."""
    assert settings.version == "1.0.0-PROD"
    assert "RAGBench" in settings.project_name


def test_health_check(client: TestClient) -> None:
    """Verify health endpoint responds correctly."""
    response = client.get(f"{settings.api_v1_str}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0-PROD"


def test_root_endpoint(client: TestClient) -> None:
    """Verify root endpoint returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "RAGBench" in response.json()["message"]
