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
    assert data["is_executable"] is True
    assert data["compatibility_status"] in ["COMPATIBLE", "WARNINGS"]
    assert "baseline_reference_commit" in data["controlled_dimensions"]
    assert data["controlled_dimensions"]["baseline_reference_commit"] == "eb3bb4d"
    assert len(data["sample_configurations"]) == 10  # Capped at first 10 samples
    first_pt = data["sample_configurations"][0]
    assert first_pt["chunking"]["strategy"] in ["fixed", "recursive"]
    assert first_pt["retrieval"]["top_k"] == 5


def test_matrix_preview_incompatible_chunk_geometry():
    """Verify matrix preview rejects geometry where overlap >= chunk_size."""
    payload = {
        "name": "Invalid Geometry Sweep",
        "chunking_strategies": ["fixed"],
        "chunk_sizes": [100],
        "chunk_overlaps": [100],  # overlap equals chunk size -> invalid
        "embedding_models": ["BAAI/bge-m3"],
        "retrieval_strategies": ["dense"],
        "rerankers": ["none"],
        "generation_models": ["none"],
        "top_k_values": [5],
    }
    response = client.post("/api/v1/experiments/matrix/preview", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_executable"] is False
    assert data["compatibility_status"] == "INCOMPATIBLE"
    assert any("overlap (100) must be strictly less than chunk_size (100)" in w for w in data["validation_warnings"])


def test_matrix_yaml_export():
    """Verify matrix YAML generation endpoint produces valid YAML, hash, and provenance."""
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
    assert "provenance:" in data["yaml_string"]
    assert "eb3bb4d" in data["yaml_string"]
    assert data["is_executable"] is True
    assert len(data["configuration_hash"]) == 64  # SHA-256


def test_list_experiments_endpoint():
    """Verify experiments list endpoint returns array."""
    response = client.get("/api/v1/experiments?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_experiment_404_not_found():
    """Verify non-existent experiment ID returns 404."""
    response = client.get("/api/v1/experiments/nonexistent_exp_uuid_99999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_get_query_trace_404_not_found():
    """Verify non-existent query trace ID returns 404."""
    response = client.get("/api/v1/traces/nonexistent_trace_uuid_99999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_list_benchmarks_endpoint():
    """Verify benchmarks list endpoint returns array."""
    response = client.get("/api/v1/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
