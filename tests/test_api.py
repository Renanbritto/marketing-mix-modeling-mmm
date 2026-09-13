"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app


@pytest.fixture(scope="module")
def client():
    """Create a test client instance with pre-initialized model."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


class TestAPIEndpoints:
    """Test suite for API routes."""

    def test_health_endpoint(self, client):
        """GET /health should return 200 with operational status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_dashboard_endpoint(self, client):
        """GET / should return 200 with HTML content."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Marketing Mix Modeling" in response.text

    def test_metrics_endpoint(self, client):
        """GET /api/v1/model/metrics should return valid goodness-of-fit stats."""
        response = client.get("/api/v1/model/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "r_squared" in data
        assert data["r_squared"] > 0.70

    def test_decomposition_endpoint(self, client):
        """GET /api/v1/model/decomposition should return complete breakdown."""
        response = client.get("/api/v1/model/decomposition")
        assert response.status_code == 200
        data = response.json()
        assert "channel_contributions" in data
        assert "weekly_data" in data

    def test_response_curves_endpoint(self, client):
        """GET /api/v1/model/response-curves should return curves for 5 channels."""
        response = client.get("/api/v1/model/response-curves")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

    def test_simulate_endpoint(self, client):
        """POST /api/v1/simulate should return scenario projection."""
        payload = {
            "budget_allocation": {
                "meta_ads": 35000.0,
                "google_ads": 30000.0,
                "tiktok_ads": 15000.0,
                "influencers": 10000.0,
                "crm_email": 5000.0,
            }
        }
        response = client.post("/api/v1/simulate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_weekly_budget"] == 95000.0
        assert data["projected_weekly_revenue"] > 0

    def test_optimize_endpoint(self, client):
        """POST /api/v1/optimize should return mathematically optimal budget."""
        payload = {
            "total_budget": 100000.0,
            "min_per_channel": {"crm_email": 2000.0},
            "max_per_channel": {"meta_ads": 50000.0},
        }
        response = client.post("/api/v1/optimize", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_budget"] == 100000.0
        assert "optimal_allocation" in data
        assert sum(data["optimal_allocation"].values()) == pytest.approx(100000.0, 1.0)

    def test_channels_summary_endpoint(self, client):
        """GET /api/v1/channels/summary should return health and recommendations."""
        response = client.get("/api/v1/channels/summary")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5
        assert all("health" in ch for ch in data)
