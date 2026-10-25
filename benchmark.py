"""
EcoStream Benchmark Evaluation Script
Evaluates latency (ms) and vehicle detection counts between Heavy (YOLOv8m)
and Light (YOLOv8n) models on data/sample_video.mp4.
"""

import os
import cv2
import time
import pandas as pd
from ultralytics import YOLO
from config import (
    HEAVY_MODEL_PATH,
    LIGHT_MODEL_PATH,
    DEFAULT_VIDEO_PATH,
    VEHICLE_CLASSES,
    IMG_SIZE,
    LOGS_DIR,
)


def run_benchmark(
    video_path=DEFAULT_VIDEO_PATH,
    max_frames=100,
    output_csv=os.path.join(LOGS_DIR, "benchmark_results.csv"),
):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    if not os.path.exists(video_path):
        print(f"[WARNING] Video file not found at '{video_path}'.")
        print("Please place a video file at 'data/sample_video.mp4' to benchmark on real footage.")
        # Write clean stub results to satisfy automated runs without crashing
        stub_df = pd.DataFrame(
            [
                {
                    "status": "VIDEO_NOT_FOUND",
                    "video_path": video_path,
                    "avg_heavy_latency_ms": 0.0,
                    "avg_light_latency_ms": 0.0,
                    "latency_reduction_pct": 0.0,
                    "avg_heavy_vehicles": 0.0,
                    "avg_light_vehicles": 0.0,
                    "count_difference": 0.0,
                }
            ]
        )
        stub_df.to_csv(output_csv, index=False)
        print(f"[INFO] Empty benchmark stub written to {output_csv}")
        return stub_df

    if not os.path.exists(HEAVY_MODEL_PATH) or not os.path.exists(LIGHT_MODEL_PATH):
        print("[ERROR] YOLO model weights missing in models/. Run 'python download_models.py' first.")
        return None

    print(f"[INFO] Loading models: {HEAVY_MODEL_PATH} and {LIGHT_MODEL_PATH}...")
    heavy_model = YOLO(HEAVY_MODEL_PATH)
    light_model = YOLO(LIGHT_MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file at {video_path}")
        return None

    heavy_latencies = []
    light_latencies = []
    heavy_counts = []
    light_counts = []

    print(f"[INFO] Benchmarking both models across up to {max_frames} frames...")
    frame_idx = 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        frame_resized = cv2.resize(frame, (640, 360))

        # 1. Benchmark Heavy Model
        t0 = time.perf_counter()
        res_heavy = heavy_model(frame_resized, imgsz=IMG_SIZE, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
        t1 = time.perf_counter()
        heavy_latencies.append((t1 - t0) * 1000.0)
        heavy_counts.append(len(res_heavy[0].boxes))

        # 2. Benchmark Light Model
        t2 = time.perf_counter()
        res_light = light_model(frame_resized, imgsz=IMG_SIZE, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
        t3 = time.perf_counter()
        light_latencies.append((t3 - t2) * 1000.0)
        light_counts.append(len(res_light[0].boxes))

    cap.release()

    if not heavy_latencies:
        print("[WARNING] No frames were processed from video.")
        return None

    avg_heavy_lat = sum(heavy_latencies) / len(heavy_latencies)
    avg_light_lat = sum(light_latencies) / len(light_latencies)
    lat_reduction = ((avg_heavy_lat - avg_light_lat) / avg_heavy_lat) * 100.0 if avg_heavy_lat > 0 else 0.0

    avg_heavy_cnt = sum(heavy_counts) / len(heavy_counts)
    avg_light_cnt = sum(light_counts) / len(light_counts)
    cnt_diff = avg_heavy_cnt - avg_light_cnt

    summary = {
        "frames_evaluated": frame_idx,
        "avg_heavy_latency_ms": round(avg_heavy_lat, 2),
        "avg_light_latency_ms": round(avg_light_lat, 2),
        "latency_reduction_pct": round(lat_reduction, 2),
        "avg_heavy_vehicles": round(avg_heavy_cnt, 2),
        "avg_light_vehicles": round(avg_light_cnt, 2),
        "count_difference": round(cnt_diff, 2),
    }

    print("\n" + "=" * 60)
    print("           ECOSTREAM DUAL-MODEL BENCHMARK REPORT")
    print("=" * 60)
    print(f"Frames evaluated          : {frame_idx}")
    print(f"Heavy Model Latency (avg) : {avg_heavy_lat:.2f} ms")
    print(f"Light Model Latency (avg) : {avg_light_lat:.2f} ms")
    print(f"Latency Reduction         : {lat_reduction:.1f}% faster")
    print(f"Heavy Vehicle Count (avg) : {avg_heavy_cnt:.2f} vehicles/frame")
    print(f"Light Vehicle Count (avg) : {avg_light_cnt:.2f} vehicles/frame")
    print(f"Count Difference (undercount): {cnt_diff:.2f} vehicles")
    print("=" * 60 + "\n")

    df_out = pd.DataFrame([summary])
    df_out.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Benchmark results saved to {output_csv}")
    return df_out


if __name__ == "__main__":
    run_benchmark()
