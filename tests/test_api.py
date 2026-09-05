"""Tests for the FastAPI application endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.utils.config import MODEL_PATH

MODEL_AVAILABLE = MODEL_PATH.exists()
NO_MODEL_REASON = "Trained model artifact not found; run scripts/train_model.py first."

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


@pytest.mark.skipif(not MODEL_AVAILABLE, reason=NO_MODEL_REASON)
def test_predict_endpoint_with_valid_input_returns_price():
    response = client.post(
        "/predict",
        json={
            "name": "Hyundai Grand i10",
            "company": "Hyundai",
            "year": 2016,
            "kms_driven": 30000,
            "fuel_type": "Petrol",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "predicted_price" in body
    assert body["currency"] == "INR"
    assert body["predicted_price"] > 0


def test_predict_endpoint_rejects_missing_fields():
    response = client.post("/predict", json={"name": "Hyundai Grand i10"})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_endpoint_rejects_invalid_fuel_type():
    response = client.post(
        "/predict",
        json={
            "name": "Hyundai Grand i10",
            "company": "Hyundai",
            "year": 2016,
            "kms_driven": 30000,
            "fuel_type": "Electric",
        },
    )
    assert response.status_code == 422


def test_predict_endpoint_rejects_negative_kms():
    response = client.post(
        "/predict",
        json={
            "name": "Hyundai Grand i10",
            "company": "Hyundai",
            "year": 2016,
            "kms_driven": -500,
            "fuel_type": "Petrol",
        },
    )
    assert response.status_code == 422


def test_predict_endpoint_rejects_year_out_of_range():
    response = client.post(
        "/predict",
        json={
            "name": "Hyundai Grand i10",
            "company": "Hyundai",
            "year": 1800,
            "kms_driven": 30000,
            "fuel_type": "Petrol",
        },
    )
    assert response.status_code == 422


def test_predict_endpoint_never_leaks_internal_stack_trace(monkeypatch):
    """Even if prediction raises unexpectedly, the response must stay generic."""
    import app.main as main_module

    def broken_predict(**kwargs):
        raise RuntimeError("simulated internal failure with sensitive details")

    monkeypatch.setattr(main_module, "predict_price", broken_predict)

    response = client.post(
        "/predict",
        json={
            "name": "Hyundai Grand i10",
            "company": "Hyundai",
            "year": 2016,
            "kms_driven": 30000,
            "fuel_type": "Petrol",
        },
    )
    assert response.status_code == 500
    assert "simulated internal failure" not in response.text


def test_index_page_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_meta_endpoint_returns_known_categories():
    response = client.get("/meta")
    assert response.status_code == 200
    body = response.json()
    assert "companies" in body
    assert "fuel_types" in body
