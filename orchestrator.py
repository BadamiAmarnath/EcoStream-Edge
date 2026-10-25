import os
import csv
import logging
import pandas as pd
from datetime import datetime
from config import (
    FEATURES,
    CARBON_HIGH,
    CARBON_LOW,
    DENSITY_THRESHOLD,
    HEAVY_POWER_W,
    ECO_POWER_W,
    POWER_SAVED_PCT,
    DECISIONS_LOG_PATH,
    LOGS_DIR,
)

logger = logging.getLogger("ecostream.orchestrator")


def predict_carbon(predictor, hour, fallback, day_of_week=3, month=1):
    """
    Predict grid carbon intensity using the trained Random Forest model.
    Falls back to a fallback value with a logged warning if prediction fails.
    """
    if predictor is None:
        logger.warning("No grid_predictor model loaded; using fallback carbon value %.1f", fallback)
        return float(fallback)

    try:
        input_df = pd.DataFrame([[hour, day_of_week, month]], columns=FEATURES)
        pred = float(predictor.predict(input_df)[0])
        return pred
    except Exception as e:
        logger.warning(
            "Predictor prediction failed for hour=%s, day_of_week=%s, month=%s: %s. Using fallback %.1f",
            hour,
            day_of_week,
            month,
            e,
            fallback,
        )
        return float(fallback)


def choose_mode(carbon, vehicle_count, current_mode="heavy_default"):
    """
    Choose operating mode based on carbon intensity and vehicle density with hysteresis.

    Modes returned:
      - 'heavy_safety': Traffic density >= DENSITY_THRESHOLD (safety override forces high fidelity).
      - 'eco': High carbon (> CARBON_HIGH) and sparse traffic (< DENSITY_THRESHOLD).
      - 'heavy_default': Clean grid (< CARBON_LOW) or default baseline.

    Hysteresis logic:
      - When in 'eco', carbon must drop below CARBON_LOW to return to 'heavy_default'.
      - When not in 'eco', carbon must rise above CARBON_HIGH to enter 'eco'.
    """
    # Safety Critical Override: high density always demands maximum detection precision
    if vehicle_count >= DENSITY_THRESHOLD:
        return "heavy_safety"

    # Eco-mode hysteresis logic
    if current_mode == "eco":
        if carbon < CARBON_LOW:
            return "heavy_default"
        return "eco"
    else:
        if carbon > CARBON_HIGH:
            return "eco"
        return "heavy_default"


def get_mode_metadata(mode, carbon, vehicle_count):
    """
    Return UI styling, tags, and XAI explanations for each mode.
    """
    if mode == "heavy_safety":
        return {
            "model_tag": "YOLOv8m (Heavy Core)",
            "state_label": "Safety-Critical Override",
            "model_key": "heavy",
            "power_w": HEAVY_POWER_W,
            "color_bgr": (60, 60, 245),  # Coral Red
            "explanation": (
                f"[SAFETY OVERRIDE] Traffic density is {vehicle_count} (>= {DENSITY_THRESHOLD}). "
                "Safety override forces YOLOv8m to ensure zero vehicle tracking dropouts."
            ),
        }
    elif mode == "eco":
        return {
            "model_tag": "YOLOv8n (Eco Core)",
            "state_label": "Eco-Mode Abatement",
            "model_key": "eco",
            "power_w": ECO_POWER_W,
            "color_bgr": (50, 220, 100),  # Neon Green
            "explanation": (
                f"[ECO-MODE ACTIVE] Grid carbon is {carbon:.1f} gCO2/kWh (> {CARBON_HIGH:.0f}). "
                f"Traffic is sparse ({vehicle_count} < {DENSITY_THRESHOLD}). "
                f"Throttled to YOLOv8n to save {POWER_SAVED_PCT:.0f}% power."
            ),
        }
    else:  # heavy_default
        return {
            "model_tag": "YOLOv8m (Heavy Core)",
            "state_label": "Default Mode",
            "model_key": "heavy",
            "power_w": HEAVY_POWER_W,
            "color_bgr": (255, 165, 0),  # Amber
            "explanation": (
                f"[DEFAULT MODE] Grid carbon is {carbon:.1f} gCO2/kWh (<= {CARBON_HIGH:.0f}). "
                "Clean energy available; executing high-fidelity core."
            ),
        }


def log_decision(
    timestamp=None,
    simulated_hour=0,
    carbon=0.0,
    vehicle_count=0,
    mode="heavy_default",
    reason="",
    filepath=DECISIONS_LOG_PATH,
):
    """
    Append an orchestration mode transition decision to CSV.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "simulated_hour", "carbon", "vehicle_count", "mode", "reason"])
            writer.writerow([timestamp, simulated_hour, round(carbon, 2), vehicle_count, mode, reason])
    except Exception as e:
        logger.error("Failed to write decision log to %s: %s", filepath, e)
