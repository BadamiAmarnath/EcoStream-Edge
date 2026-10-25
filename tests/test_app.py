import pytest
import os
from app import app
from config import SCENARIOS, CARBON_HIGH, CARBON_LOW, interpolate_carbon_profile


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_landing_page(client):
    """Landing page at '/' returns 200 and displays scenario cards."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "EcoStream: Edge Vision Orchestrator" in html
    assert "Traffic Flow Analytics" in html
    assert "Retail Store Footfall" in html
    assert "/scenario/traffic_demo" in html
    assert "/scenario/retail_store" in html


def test_scenario_pages(client):
    """Both scenario pages at '/scenario/<id>' return 200 with parameterized templates."""
    # 1. Traffic Demo Scenario Page
    res_traffic = client.get("/scenario/traffic_demo")
    assert res_traffic.status_code == 200
    html_traffic = res_traffic.data.decode("utf-8")
    assert "Traffic Flow Analytics" in html_traffic
    assert "← All Scenarios" in html_traffic
    assert "chartCarbon" in html_traffic
    assert "chartLoad" in html_traffic
    assert "Estimated energy saved (current loop)" in html_traffic
    assert "chart.umd.min.js" in html_traffic

    # 2. Retail Store Scenario Page
    res_retail = client.get("/scenario/retail_store")
    assert res_retail.status_code == 200
    html_retail = res_retail.data.decode("utf-8")
    assert "Retail Store Footfall" in html_retail
    assert "← All Scenarios" in html_retail
    assert "chartCarbon" in html_retail
    assert "chartLoad" in html_retail
    assert "Estimated energy saved (current loop)" in html_retail


def test_unknown_scenario_returns_404(client):
    """Unknown scenario id returns 404."""
    response = client.get("/scenario/unknown_nonexistent_id")
    assert response.status_code == 404
    assert "404" in response.data.decode("utf-8")

    # Also per-scenario endpoints return 404 for unknown ids
    assert client.get("/telemetry/unknown_id").status_code == 404
    assert client.get("/video_feed/unknown_id").status_code == 404


def test_scenario_config_sanity():
    """Verify scenario configuration integrity and required fields."""
    required_keys = [
        "id",
        "name",
        "description",
        "video_path",
        "target_classes",
        "class_names",
        "object_label",
        "load_threshold",
        "carbon_profile",
        "timeline_labels",
        "device_profile",
    ]

    assert "traffic_demo" in SCENARIOS
    assert "retail_store" in SCENARIOS

    for s_id, cfg in SCENARIOS.items():
        assert cfg["id"] == s_id
        for k in required_keys:
            assert k in cfg, f"Missing required key '{k}' in scenario '{s_id}'"

        # Check classes are valid list of integers
        assert isinstance(cfg["target_classes"], list)
        assert len(cfg["target_classes"]) > 0
        for cls_id in cfg["target_classes"]:
            assert isinstance(cls_id, int) and 0 <= cls_id <= 80

        # Check load threshold
        assert isinstance(cfg["load_threshold"], int) and cfg["load_threshold"] > 0

        # Check carbon profile has points staying above CARBON_HIGH
        assert isinstance(cfg["carbon_profile"], list)
        assert len(cfg["carbon_profile"]) >= 2
        for pt in cfg["carbon_profile"]:
            assert len(pt) == 2
            progress, carbon = pt
            assert 0.0 <= progress <= 1.0
            assert carbon > CARBON_HIGH, f"Carbon {carbon} must stay above {CARBON_HIGH} in {s_id}"

        # Check device profile
        dev = cfg["device_profile"]
        assert "name" in dev
        assert dev["heavy_power_w"] > dev["eco_power_w"] > 0
        assert 0.0 < dev["power_saved_pct"] < 100.0


def test_carbon_profile_interpolation():
    """Test linear interpolation of simulated carbon profile across video progress."""
    profile = [(0.0, 280.0), (0.5, 320.0), (1.0, 300.0)]
    assert interpolate_carbon_profile(profile, 0.0) == 280.0
    assert interpolate_carbon_profile(profile, 0.25) == 300.0
    assert interpolate_carbon_profile(profile, 0.5) == 320.0
    assert interpolate_carbon_profile(profile, 0.75) == 310.0
    assert interpolate_carbon_profile(profile, 1.0) == 300.0


def test_telemetry_routes(client):
    """Test telemetry JSON structure for scenarios."""
    res = client.get("/telemetry/traffic_demo")
    assert res.status_code == 200
    data = res.get_json()
    assert data["scenario_id"] == "traffic_demo"
    assert "carbon_intensity" in data
    assert "detected_count" in data
    assert "active_model" in data
    assert "state" in data
    assert "xai_explanation" in data
    assert "energy_saved_pct" in data


def test_fallback_frame_route(client):
    """Fallback frame returns valid JPEG."""
    response = client.get("/fallback_frame")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert len(response.data) > 0
