import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "EcoStream: Explainable Runtime Orchestrator" in html
    assert "chart.umd.min.js" in html


def test_telemetry_route(client):
    response = client.get("/telemetry")
    assert response.status_code == 200
    data = response.get_json()
    assert "carbon_intensity" in data
    assert "active_model" in data
    assert "power_watts" in data
    assert "energy_saved_pct" in data


def test_fallback_frame_route(client):
    response = client.get("/fallback_frame")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert len(response.data) > 0
