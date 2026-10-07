"""Unit tests verifying CORS policy hardening and credential security."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_cors_allowed_local_origins() -> None:
    """Verify configured developer and frontend origins are permitted with credentials."""
    client = TestClient(app)

    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    for origin in allowed_origins:
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == origin
        assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_rejects_unauthorized_origin() -> None:
    """Verify unauthorized external origins do not receive CORS authorization headers."""
    client = TestClient(app)

    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://malicious-website.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Fastapi CORSMiddleware does not include access-control-allow-origin for unlisted origins
    assert "access-control-allow-origin" not in response.headers
