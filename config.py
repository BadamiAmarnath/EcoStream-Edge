import os

# Base paths calculated relative to config.py location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# File paths
GRID_DATA_PATH = os.path.join(DATA_DIR, 'carbon_grid_data.csv')
DEFAULT_VIDEO_PATH = os.path.join(DATA_DIR, 'sample_video.mp4')
PREDICTOR_PATH = os.path.join(MODELS_DIR, 'grid_predictor.pkl')
HEAVY_MODEL_PATH = os.path.join(MODELS_DIR, 'heavy_yolo.pt')
LIGHT_MODEL_PATH = os.path.join(MODELS_DIR, 'light_yolo.pt')
DECISIONS_LOG_PATH = os.path.join(LOGS_DIR, 'decisions.csv')

# Feature columns used for both ML training and serving
FEATURES = ['hour', 'day_of_week', 'month']

# Orchestration thresholds & hysteresis
CARBON_HIGH = 260.0  # Enter eco mode when carbon exceeds this
CARBON_LOW = 240.0   # Leave eco mode only when carbon drops below this
DENSITY_THRESHOLD = 5  # Density safety override: >= 5 vehicles forces heavy model

# Model specifications & power consumption (W)
HEAVY_POWER_W = 35.0  # YOLOv8m nominal power
ECO_POWER_W = 12.0    # YOLOv8n throttled power
POWER_SAVED_PCT = ((HEAVY_POWER_W - ECO_POWER_W) / HEAVY_POWER_W) * 100.0

# Detection parameters
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck

# Eco-mode safety probe interval (Probe with heavy model every N frames to avoid undercounting)
ECO_SAFETY_PROBE_INTERVAL = 15

# Server configuration
DEFAULT_HOST = os.environ.get('ECOSTREAM_HOST', '127.0.0.1')
DEFAULT_PORT = int(os.environ.get('ECOSTREAM_PORT', 5000))
