import os
import cv2
import pickle
import warnings
import logging
import pandas as pd
from datetime import datetime

# Silence noisy third-party logging
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
os.environ["YOLO_VERBOSE"] = "False"

from ultralytics import YOLO
from config import (
    PREDICTOR_PATH,
    HEAVY_MODEL_PATH,
    LIGHT_MODEL_PATH,
    DEFAULT_VIDEO_PATH,
    GRID_DATA_PATH,
    VEHICLE_CLASSES,
    DENSITY_THRESHOLD,
    CARBON_HIGH,
    CARBON_LOW,
)
from orchestrator import predict_carbon, choose_mode, get_mode_metadata, log_decision


def main():
    print("🧠 Initializing EcoStream Intelligent Controller...")

    # Guard: check video existence
    if not os.path.exists(DEFAULT_VIDEO_PATH):
        print(f"\n[ERROR] Video file not found at: {DEFAULT_VIDEO_PATH}")
        print("Please place a video file at 'data/sample_video.mp4' to run the controller.")
        return

    # Guard: check model artifacts
    if not os.path.exists(PREDICTOR_PATH):
        print(f"\n[ERROR] Grid predictor not found at: {PREDICTOR_PATH}")
        print("Please execute: python train_predictor.py")
        return

    if not os.path.exists(HEAVY_MODEL_PATH) or not os.path.exists(LIGHT_MODEL_PATH):
        print("\n[ERROR] Vision models missing in models/.")
        print("Please execute: python download_models.py")
        return

    # Load Random Forest predictor
    with open(PREDICTOR_PATH, "rb") as f:
        predictor = pickle.load(f)

    # Load vision models
    print("[INFO] Loading YOLO models into memory...")
    heavy_model = YOLO(HEAVY_MODEL_PATH)
    light_model = YOLO(LIGHT_MODEL_PATH)

    # Precompute historical hourly fallback from CSV
    try:
        df_csv = pd.read_csv(GRID_DATA_PATH)
        hourly_fallback = df_csv.groupby("hour")["carbon_intensity"].mean().to_dict()
    except Exception:
        hourly_fallback = {h: 250.0 for h in range(24)}

    cap = cv2.VideoCapture(DEFAULT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video capture for: {DEFAULT_VIDEO_PATH}")
        return

    frame_idx = 0
    current_mode = "heavy_default"
    current_sim_hour = -1
    current_carbon = 250.0
    cached_vehicle_count = 0

    print("🚀 Starting video analysis pipeline with dynamic carbon orchestration...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        # Synchronized simulated clock matching app.py
        simulated_hour = (frame_idx // 20) % 24

        # Predict carbon once per simulated hour
        if simulated_hour != current_sim_hour:
            current_sim_hour = simulated_hour
            current_carbon = predict_carbon(
                predictor,
                hour=simulated_hour,
                fallback=hourly_fallback.get(simulated_hour, 250.0),
            )

        # Pre-resize frame to standard 640x360
        frame_resized = cv2.resize(frame, (640, 360))

        # Perform decision & inference every 3rd frame
        if frame_idx % 3 == 0 or frame_idx == 1:
            new_mode = choose_mode(current_carbon, cached_vehicle_count, current_mode)

            # Log mode switch transition
            if new_mode != current_mode:
                meta = get_mode_metadata(new_mode, current_carbon, cached_vehicle_count)
                log_decision(
                    simulated_hour=simulated_hour,
                    carbon=current_carbon,
                    vehicle_count=cached_vehicle_count,
                    mode=new_mode,
                    reason=meta["state_label"],
                )
                current_mode = new_mode

            active_model = light_model if current_mode == "eco" else heavy_model
            results = active_model(frame_resized, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
            boxes = results[0].boxes
            cached_vehicle_count = len(boxes)

        # Status output every 30 frames
        if frame_idx % 30 == 0:
            meta = get_mode_metadata(current_mode, current_carbon, cached_vehicle_count)
            print(
                f"Frame {frame_idx:04d} | Sim Hour: {simulated_hour:02d}:00 | "
                f"Grid Carbon: {current_carbon:.1f} gCO2/kWh | Vehicles: {cached_vehicle_count:02d} | "
                f"Mode: {meta['state_label']} [{meta['model_tag']}]"
            )

    cap.release()
    print("✅ Pipeline processing finished successfully!")


if __name__ == "__main__":
    main()