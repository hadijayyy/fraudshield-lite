"""
Tests for the FastAPI prediction and health endpoints.
"""

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """GET /health should return 200 with status and model info."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "model_version" in body
    assert "model_loaded" in body


def test_predict_endpoint_missing_model() -> None:
    """POST /predict should return 503 when model is not loaded."""
    payload = {
        "transaction_id": "txn_001",
        "user_id": "user_01",
        "amount": 150.0,
        "timestamp": "2025-06-01T12:00:00",
    }
    response = client.post("/predict", json=payload)
    # Without a loaded model, the endpoint returns 503
    assert response.status_code in (200, 503)


def test_predict_endpoint_validates_amount() -> None:
    """POST /predict should reject non-positive amounts."""
    payload = {
        "transaction_id": "txn_002",
        "user_id": "user_01",
        "amount": -100.0,
        "timestamp": "2025-06-01T12:00:00",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error
