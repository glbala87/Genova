"""Unit tests for the FastAPI server factory and middleware."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from genova.api.server import create_app


@pytest.fixture
def app_no_model():
    """Create an app without loading a real model."""
    with patch.dict(
        os.environ,
        {
            "GENOVA_AUTH_ENABLED": "0",
            "GENOVA_RATE_LIMIT_ENABLED": "0",
            "GENOVA_METRICS_ENABLED": "0",
            "GENOVA_TRACING_ENABLED": "0",
            "GENOVA_REQUEST_LOGGING_ENABLED": "0",
        },
    ):
        app = create_app(model_path=None, device="cpu")
    return app


@pytest.fixture
def client(app_no_model):
    """Test client with security disabled."""
    return TestClient(app_no_model, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_fields(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_model_not_loaded(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["model_loaded"] is False
        assert data["status"] == "degraded"


class TestModelNotLoaded:
    """Test that inference endpoints return 503 when model is not loaded."""

    def test_predict_variant_503(self, client):
        resp = client.post(
            "/predict_variant",
            json={
                "variants": [{"ref": "A", "alt": "G"}],
            },
        )
        assert resp.status_code == 503

    def test_predict_expression_503(self, client):
        resp = client.post(
            "/predict_expression",
            json={"sequences": ["ACGTACGT"]},
        )
        assert resp.status_code == 503

    def test_predict_methylation_503(self, client):
        resp = client.post(
            "/predict_methylation",
            json={"sequences": ["ACGTACGT"]},
        )
        assert resp.status_code == 503

    def test_embed_503(self, client):
        resp = client.post(
            "/embed",
            json={"sequences": ["ACGTACGT"]},
        )
        assert resp.status_code == 503


class TestCORSConfiguration:
    """Test CORS middleware configuration."""

    def test_default_cors_allows_origin(self, client):
        resp = client.get(
            "/health",
            headers={"Origin": "http://example.com"},
        )
        # Without GENOVA_CORS_ORIGINS set, defaults to *
        assert resp.status_code == 200

    def test_restricted_cors(self):
        with patch.dict(
            os.environ,
            {
                "GENOVA_AUTH_ENABLED": "0",
                "GENOVA_RATE_LIMIT_ENABLED": "0",
                "GENOVA_METRICS_ENABLED": "0",
                "GENOVA_TRACING_ENABLED": "0",
                "GENOVA_REQUEST_LOGGING_ENABLED": "0",
                "GENOVA_CORS_ORIGINS": "https://allowed.com",
            },
        ):
            app = create_app(model_path=None, device="cpu")
            restricted_client = TestClient(app, raise_server_exceptions=False)
            resp = restricted_client.get("/health")
            assert resp.status_code == 200


class TestInputValidation:
    """Test that API rejects malformed input."""

    def test_empty_variants_list(self, client):
        resp = client.post("/predict_variant", json={"variants": []})
        assert resp.status_code == 422

    def test_invalid_sequence_characters(self, client):
        resp = client.post(
            "/embed",
            json={"sequences": ["ACGTXYZ"]},
        )
        assert resp.status_code == 422

    def test_invalid_pooling_strategy(self, client):
        resp = client.post(
            "/embed",
            json={"sequences": ["ACGT"], "pooling": "invalid"},
        )
        assert resp.status_code == 422

    def test_window_size_too_small(self, client):
        resp = client.post(
            "/predict_variant",
            json={
                "variants": [{"ref": "A", "alt": "G"}],
                "window_size": 10,
            },
        )
        assert resp.status_code == 422

    def test_window_size_too_large(self, client):
        resp = client.post(
            "/predict_variant",
            json={
                "variants": [{"ref": "A", "alt": "G"}],
                "window_size": 99999,
            },
        )
        assert resp.status_code == 422


class TestErrorResponsesDoNotLeakDetails:
    """Ensure 500 errors don't expose internal exception messages."""

    def test_global_exception_handler(self, client):
        # model/info requires loaded model, will trigger _get_engine -> 503
        resp = client.get("/model/info")
        assert resp.status_code == 503
        data = resp.json()
        assert "detail" in data
        # Should not contain Python traceback or class names
        assert "Traceback" not in data["detail"]
