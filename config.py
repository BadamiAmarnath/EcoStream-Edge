import os

# Base paths calculated relative to config.py location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# File paths
GRID_DATA_PATH = os.path.join(DATA_DIR, "carbon_grid_data.csv")
DEFAULT_VIDEO_PATH = os.path.join(DATA_DIR, "sample_video.mp4")
PREDICTOR_PATH = os.path.join(MODELS_DIR, "grid_predictor.pkl")
HEAVY_MODEL_PATH = os.path.join(MODELS_DIR, "heavy_yolo.pt")
LIGHT_MODEL_PATH = os.path.join(MODELS_DIR, "light_yolo.pt")
DECISIONS_LOG_PATH = os.path.join(LOGS_DIR, "decisions.csv")

# Demo Mode Configuration
DEMO_MODE = True
DEMO_CARBON_3AM = 300.0  # Legacy constant maintained for backward compatibility
DEMO_CARBON_9AM = 300.0  # Legacy constant maintained for backward compatibility

# Feature columns used for both ML training and serving
FEATURES = ["hour", "day_of_week", "month"]

# Orchestration thresholds & hysteresis
CARBON_HIGH = 260.0  # Enter eco mode when carbon exceeds this (gCO2/kWh)
CARBON_LOW = 240.0   # Leave eco mode only when carbon drops below this (gCO2/kWh)
DENSITY_THRESHOLD = 5  # Default scene load threshold: >= 5 objects forces heavy model

# Model specifications & power consumption (W) - Default baseline assumptions
HEAVY_POWER_W = 35.0  # Assumed YOLOv8m nominal power
ECO_POWER_W = 12.0    # Assumed YOLOv8n throttled power
POWER_SAVED_PCT = ((HEAVY_POWER_W - ECO_POWER_W) / HEAVY_POWER_W) * 100.0

# Detection parameters
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck
IMG_SIZE = 416  # YOLO input image size for fast CPU inference

# Eco-mode safety probe interval (Probe with heavy model every N frames to avoid undercounting)
ECO_SAFETY_PROBE_INTERVAL = 15

# Server configuration
DEFAULT_HOST = os.environ.get("ECOSTREAM_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("ECOSTREAM_PORT", 5000))

# Multi-Scenario Configuration
SCENARIOS = {
    "traffic_demo": {
        "id": "traffic_demo",
        "name": "Traffic Flow Analytics",
        "description": "Dual-core vehicle surveillance with density-based safety override",
        "video_path": os.path.join(DATA_DIR, "sample_video.mp4"),
        "target_classes": [2, 3, 5, 7],  # COCO: car, motorcycle, bus, truck
        "class_names": ["car", "motorcycle", "bus", "truck"],
        "object_label": "vehicles",
        "load_threshold": 5,
        "carbon_profile": [
            (0.0, 285.0),
            (0.25, 310.0),
            (0.5, 335.0),
            (0.75, 315.0),
            (1.0, 290.0),
        ],
        "timeline_labels": {
            "first": "03:00 AM (Low Traffic)",
            "second": "09:00 AM (Rush Hour)",
        },
        "device_profile": {
            "name": "NVIDIA Jetson AGX Orin (Assumed)",
            "heavy_power_w": 35.0,
            "eco_power_w": 12.0,
            "power_saved_pct": round(((35.0 - 12.0) / 35.0) * 100.0, 1),
        },
    },
    "retail_store": {
        "id": "retail_store",
        "name": "Retail Store Footfall",
        "description": "Edge occupancy monitoring with real-time crowd density scaling",
        "video_path": (
            os.path.join(DATA_DIR, "scenario2_retail.mp4")
            if os.path.exists(os.path.join(DATA_DIR, "scenario2_retail.mp4"))
            else os.path.join(DATA_DIR, "Untitled design.mp4")
        ),
        "target_classes": [0],  # COCO: person
        "class_names": ["person"],
        "object_label": "people",
        "load_threshold": 6,
        "carbon_profile": [
            (0.0, 295.0),
            (0.3, 320.0),
            (0.6, 340.0),
            (0.85, 310.0),
            (1.0, 300.0),
        ],
        "timeline_labels": {
            "first": "07:00 AM (Quiet)",
            "second": "06:00 PM (Rush)",
        },
        "device_profile": {
            "name": "NVIDIA Jetson Orin Nano (Assumed)",
            "heavy_power_w": 25.0,
            "eco_power_w": 10.0,
            "power_saved_pct": round(((25.0 - 10.0) / 25.0) * 100.0, 1),
        },
    },
}


def interpolate_carbon_profile(profile, progress):
    """
    Interpolate simulated carbon intensity (gCO2/kWh) from a scenario's profile
    given normalized video progress [0.0, 1.0].
    """
    if not profile:
        return 300.0
    if len(profile) == 1:
        return float(profile[0][1])

    p = max(0.0, min(1.0, float(progress)))
    sorted_pts = sorted(profile, key=lambda x: x[0])

    if p <= sorted_pts[0][0]:
        return float(sorted_pts[0][1])
    if p >= sorted_pts[-1][0]:
        return float(sorted_pts[-1][1])

    for i in range(len(sorted_pts) - 1):
        p0, c0 = sorted_pts[i]
        p1, c1 = sorted_pts[i + 1]
        if p0 <= p <= p1:
            if p1 == p0:
                return float(c0)
            t = (p - p0) / (p1 - p0)
            return float(c0 + t * (c1 - c0))

    return float(sorted_pts[-1][1])
