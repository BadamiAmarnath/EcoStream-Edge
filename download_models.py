import os
import shutil
from ultralytics import YOLO
from config import MODELS_DIR, HEAVY_MODEL_PATH, LIGHT_MODEL_PATH


def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Heavy Model: YOLOv8m (25.9M params, ~35W nominal power)
    if os.path.exists(HEAVY_MODEL_PATH):
        print(f"[INFO] Heavy model already present at {HEAVY_MODEL_PATH}. Skipping download.")
    else:
        print("[INFO] Downloading Heavy Model (YOLOv8m)...")
        # Trigger ultralytics download
        YOLO("yolov8m.pt")
        if os.path.exists("yolov8m.pt"):
            shutil.move("yolov8m.pt", HEAVY_MODEL_PATH)
        print(f"[SUCCESS] Heavy model saved to {HEAVY_MODEL_PATH}")

    # 2. Light Model: YOLOv8n (3.2M params, ~12W throttled power)
    if os.path.exists(LIGHT_MODEL_PATH):
        print(f"[INFO] Light model already present at {LIGHT_MODEL_PATH}. Skipping download.")
    else:
        print("[INFO] Downloading Light Model (YOLOv8n)...")
        YOLO("yolov8n.pt")
        if os.path.exists("yolov8n.pt"):
            shutil.move("yolov8n.pt", LIGHT_MODEL_PATH)
        print(f"[SUCCESS] Light model saved to {LIGHT_MODEL_PATH}")

    print("[SUCCESS] Model synchronization complete.")


if __name__ == "__main__":
    download_models()
