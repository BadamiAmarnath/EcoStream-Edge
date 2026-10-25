import logging
import pytest
import pandas as pd
from unittest.mock import MagicMock
from config import CARBON_HIGH, CARBON_LOW, DENSITY_THRESHOLD, FEATURES
from orchestrator import choose_mode, predict_carbon


# ---------------------------------------------------------
# choose_mode() Test Suite
# ---------------------------------------------------------

def test_choose_mode_safety_override():
    """Density >= DENSITY_THRESHOLD must always force heavy_safety regardless of carbon."""
    assert choose_mode(carbon=100.0, vehicle_count=DENSITY_THRESHOLD, current_mode="heavy_default") == "heavy_safety"
    assert choose_mode(carbon=350.0, vehicle_count=DENSITY_THRESHOLD + 3, current_mode="eco") == "heavy_safety"
    assert choose_mode(carbon=CARBON_HIGH + 50, vehicle_count=DENSITY_THRESHOLD, current_mode="eco") == "heavy_safety"


def test_choose_mode_clean_grid_below_threshold():
    """When grid is clean (< CARBON_LOW) and traffic is sparse, run heavy_default."""
    assert choose_mode(carbon=200.0, vehicle_count=2, current_mode="heavy_default") == "heavy_default"
    assert choose_mode(carbon=CARBON_LOW - 10, vehicle_count=0, current_mode="heavy_default") == "heavy_default"


def test_choose_mode_dirty_grid_above_threshold():
    """When carbon > CARBON_HIGH and traffic is sparse, transition into eco mode."""
    assert choose_mode(carbon=CARBON_HIGH + 10, vehicle_count=2, current_mode="heavy_default") == "eco"
    assert choose_mode(carbon=350.0, vehicle_count=0, current_mode="heavy_safety") == "eco"


def test_choose_mode_hysteresis_band():
    """
    Hysteresis test:
    Between CARBON_LOW (240) and CARBON_HIGH (260), mode must NOT oscillate.
    It must preserve the current state.
    """
    mid_carbon = (CARBON_HIGH + CARBON_LOW) / 2.0  # 250.0

    # If currently in eco mode, carbon at 250 must STAY in eco mode
    assert choose_mode(carbon=mid_carbon, vehicle_count=2, current_mode="eco") == "eco"

    # If currently in heavy_default mode, carbon at 250 must STAY in heavy_default mode
    assert choose_mode(carbon=mid_carbon, vehicle_count=2, current_mode="heavy_default") == "heavy_default"

    # Eco mode only exits once carbon drops strictly below CARBON_LOW
    assert choose_mode(carbon=CARBON_LOW - 1.0, vehicle_count=2, current_mode="eco") == "heavy_default"

    # Heavy mode only transitions to eco once carbon rises strictly above CARBON_HIGH
    assert choose_mode(carbon=CARBON_HIGH + 1.0, vehicle_count=2, current_mode="heavy_default") == "eco"


def test_choose_mode_safety_override_exit():
    """When safety override condition subsides and carbon is high, transition to eco."""
    assert choose_mode(carbon=CARBON_HIGH + 20, vehicle_count=2, current_mode="heavy_safety") == "eco"


# ---------------------------------------------------------
# predict_carbon() Test Suite
# ---------------------------------------------------------

def test_predict_carbon_success():
    """Predictor builds DataFrame matching FEATURES and returns prediction."""
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = [285.5]

    val = predict_carbon(mock_predictor, hour=19, fallback=250.0, day_of_week=3, month=1)
    assert val == 285.5

    mock_predictor.predict.assert_called_once()
    passed_df = mock_predictor.predict.call_args[0][0]
    assert isinstance(passed_df, pd.DataFrame)
    assert list(passed_df.columns) == FEATURES
    assert passed_df.iloc[0].to_dict() == {"hour": 19, "day_of_week": 3, "month": 1}


def test_predict_carbon_fallback_logs_warning_on_exception(caplog):
    """When predictor raises an exception, it must log a warning and return the fallback value."""
    mock_predictor = MagicMock()
    mock_predictor.predict.side_effect = ValueError("Feature mismatch dummy error")

    with caplog.at_level(logging.WARNING):
        val = predict_carbon(mock_predictor, hour=14, fallback=220.0)

    assert val == 220.0
    assert any("Predictor prediction failed" in record.message for record in caplog.records)


def test_predict_carbon_none_predictor_logs_warning(caplog):
    """When predictor is None, it must log a warning and return the fallback value."""
    with caplog.at_level(logging.WARNING):
        val = predict_carbon(None, hour=10, fallback=215.0)

    assert val == 215.0
    assert any("No grid_predictor model loaded" in record.message for record in caplog.records)


def test_log_decision(tmp_path):
    """log_decision must correctly write header and CSV transition records."""
    from orchestrator import log_decision
    log_file = tmp_path / "test_decisions.csv"

    log_decision(
        timestamp="2026-10-25T11:00:00",
        simulated_hour=19,
        carbon=275.5,
        vehicle_count=2,
        mode="eco",
        reason="Eco-Mode Abatement",
        filepath=str(log_file),
    )

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 2
    assert "timestamp,simulated_hour,carbon,vehicle_count,mode,reason" in content[0]
    assert "2026-10-25T11:00:00,19,275.5,2,eco,Eco-Mode Abatement" in content[1]

