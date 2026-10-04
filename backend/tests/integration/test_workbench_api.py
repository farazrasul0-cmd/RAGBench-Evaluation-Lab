"""Integration tests for Academic Research Workbench REST API endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """Verify system health endpoint returns healthy status and version."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_matrix_preview_combinatorics():
    """Verify matrix preview endpoint computes exact combinatorial points and category."""
    payload = {
        "name": "Factorial Sweep Test",
        "description": "Preview test",
        "chunking_strategies": ["fixed", "recursive"],
        "chunk_sizes": [200, 400],
        "chunk_overlaps": [20],
        "embedding_models": ["BAAI/bge-m3"],
        "retrieval_strategies": ["dense", "bm25", "hybrid"],
        "rerankers": ["none"],
        "generation_models": ["mock"],
        "top_k_values": [5],
    }
    response = client.post("/api/v1/experiments/matrix/preview", json=payload)
    assert response.status_code == 200
    data = response.json()
    # 2 * 2 * 1 * 1 * 3 * 1 * 1 * 1 = 12
    assert data["total_combinations"] == 12
    assert data["complexity_category"] == "LOW"
    assert len(data["sample_configurations"]) == 10  # Capped at first 10 samples
    first_pt = data["sample_configurations"][0]
    assert first_pt["chunking"]["strategy"] in ["fixed", "recursive"]
    assert first_pt["retrieval"]["top_k"] == 5


def test_matrix_yaml_export():
    """Verify matrix YAML generation endpoint produces valid YAML and hash."""
    payload = {
        "name": "YAML Export Test",
        "description": "Export test",
        "chunking_strategies": ["fixed"],
        "chunk_sizes": [200],
        "chunk_overlaps": [20],
        "embedding_models": ["BAAI/bge-m3"],
        "retrieval_strategies": ["dense", "hybrid"],
        "rerankers": ["none"],
        "generation_models": ["mock"],
        "top_k_values": [5],
    }
    response = client.post("/api/v1/experiments/matrix/yaml", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_combinations"] == 2
    assert "name: YAML Export Test" in data["yaml_string"]
    assert "parameters:" in data["yaml_string"]
    assert len(data["configuration_hash"]) == 64  # SHA-256


def test_list_experiments_endpoint():
    """Verify experiments list endpoint returns array."""
    response = client.get("/api/v1/experiments?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_list_benchmarks_endpoint():
    """Verify benchmarks list endpoint returns array."""
    response = client.get("/api/v1/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
