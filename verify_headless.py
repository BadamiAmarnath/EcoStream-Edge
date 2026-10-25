"""
Headless Verification Script for EcoStream Decoupled Pipeline
Runs over exactly one full pass of data/sample_video.mp4 (480 frames, 24 FPS, 20.0s).
Logs per-second telemetry and measures:
  - Average vehicle count in first half (3 AM) vs second half (9 AM)
  - Time at which mode switches from light (eco) to heavy (safety override)
  - Measured latency of both models (at imgsz=416 and imgsz=640 before/after)
  - Real playback FPS achieved by the decoupled playback engine
"""

import os
import cv2
import time
import json
import threading
import numpy as np
from ultralytics import YOLO

from config import (
    DEFAULT_VIDEO_PATH,
    HEAVY_MODEL_PATH,
    LIGHT_MODEL_PATH,
    VEHICLE_CLASSES,
    IMG_SIZE,
    CARBON_HIGH,
    CARBON_LOW,
    DENSITY_THRESHOLD,
    DEMO_MODE,
    DEMO_CARBON_3AM,
    DEMO_CARBON_9AM,
    ECO_SAFETY_PROBE_INTERVAL,
    LOGS_DIR,
)
from orchestrator import choose_mode, get_mode_metadata


class HeadlessInferenceWorker(threading.Thread):
    def __init__(self, heavy_model, light_model):
        super().__init__(daemon=True)
        self.heavy_model = heavy_model
        self.light_model = light_model
        self.running = True
        self.is_busy = False
        self.new_frame_event = threading.Event()
        self.lock = threading.Lock()

        # Input slots
        self.pending_frame = None
        self.pending_meta = {}

        # Tracking state
        self.current_mode = "eco" if DEMO_MODE else "heavy_default"
        self.latest_count = 0
        self.latest_boxes = []
        self.latest_model_tag = "YOLOv8n (Eco Core)" if DEMO_MODE else "YOLOv8m (Heavy Core)"
        self.latest_latency_ms = 0.0
        self.probe_counter = 0

        # Latency statistics per model during run
        self.light_latencies = []
        self.heavy_latencies = []
        self.switch_events = []

    def submit_frame(self, frame, carbon, sim_hour, frame_idx, wall_time):
        with self.lock:
            if self.is_busy:
                return False
            self.pending_frame = frame.copy()
            self.pending_meta = {
                "carbon": carbon,
                "hour": sim_hour,
                "frame_idx": frame_idx,
                "wall_time": wall_time,
            }
            self.is_busy = True
            self.new_frame_event.set()
            return True

    def run(self):
        while self.running:
            self.new_frame_event.wait()
            self.new_frame_event.clear()
            if not self.running:
                break

            with self.lock:
                frame = self.pending_frame
                meta = self.pending_meta

            if frame is None:
                continue

            current_carbon = meta.get("carbon", DEMO_CARBON_3AM)
            sim_hour = meta.get("hour", 3)
            frame_idx = meta.get("frame_idx", 0)
            wall_time = meta.get("wall_time", 0.0)

            t0 = time.perf_counter()

            # Periodic heavy model safety validation probe when in eco mode
            if (
                self.current_mode == "eco"
                and (self.probe_counter % ECO_SAFETY_PROBE_INTERVAL == 0)
                and self.heavy_model
            ):
                probe_res = self.heavy_model(
                    frame, imgsz=IMG_SIZE, classes=VEHICLE_CLASSES, verbose=False, conf=0.25
                )
                probe_cnt = len(probe_res[0].boxes)
                if probe_cnt >= DENSITY_THRESHOLD:
                    self.latest_count = probe_cnt

            self.probe_counter += 1

            # Determine mode via central orchestrator
            new_mode = choose_mode(current_carbon, self.latest_count, self.current_mode)
            if new_mode != self.current_mode:
                event_info = {
                    "from_mode": self.current_mode,
                    "to_mode": new_mode,
                    "frame_idx": frame_idx,
                    "time_sec": round(wall_time, 3),
                    "vehicle_count": self.latest_count,
                    "carbon": current_carbon,
                    "sim_hour": sim_hour,
                }
                self.switch_events.append(event_info)
                self.current_mode = new_mode

            mode_meta = get_mode_metadata(self.current_mode, current_carbon, self.latest_count)
            active_model = (
                self.light_model
                if self.current_mode == "eco" and self.light_model
                else self.heavy_model
            )

            boxes = []
            if active_model is not None:
                results = active_model(
                    frame, imgsz=IMG_SIZE, classes=VEHICLE_CLASSES, verbose=False, conf=0.25
                )
                boxes = [b.xyxy[0].cpu().numpy().astype(int) for b in results[0].boxes]

            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0

            if self.current_mode == "eco":
                self.light_latencies.append(elapsed_ms)
            else:
                self.heavy_latencies.append(elapsed_ms)

            with self.lock:
                self.latest_boxes = boxes
                self.latest_count = len(boxes)
                self.latest_model_tag = mode_meta["model_tag"]
                self.latest_latency_ms = elapsed_ms
                self.is_busy = False


def measure_before_after_benchmarks(sample_frame, heavy_model, light_model):
    """
    Directly measure per-model inference latencies at imgsz=640 (before)
    vs imgsz=416 (after) on the CPU.
    """
    print("\n--- Measuring Per-Model Inference Latency (Before vs After imgsz optimization) ---")
    frame_resized = cv2.resize(sample_frame, (640, 360))

    # Warmup
    for _ in range(2):
        _ = heavy_model(frame_resized, imgsz=640, classes=VEHICLE_CLASSES, verbose=False)
        _ = light_model(frame_resized, imgsz=416, classes=VEHICLE_CLASSES, verbose=False)

    # 1. Heavy Model at imgsz=640 (Before)
    t_heavy_640 = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = heavy_model(frame_resized, imgsz=640, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
        t_heavy_640.append((time.perf_counter() - t0) * 1000.0)
    avg_heavy_640 = float(np.mean(t_heavy_640))

    # 2. Heavy Model at imgsz=416 (After)
    t_heavy_416 = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = heavy_model(frame_resized, imgsz=416, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
        t_heavy_416.append((time.perf_counter() - t0) * 1000.0)
    avg_heavy_416 = float(np.mean(t_heavy_416))

    # 3. Light Model at imgsz=640 (Before)
    t_light_640 = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = light_model(frame_resized, imgsz=640, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
        t_light_640.append((time.perf_counter() - t0) * 1000.0)
    avg_light_640 = float(np.mean(t_light_640))

    # 4. Light Model at imgsz=416 (After)
    t_light_416 = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = light_model(frame_resized, imgsz=416, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
        t_light_416.append((time.perf_counter() - t0) * 1000.0)
    avg_light_416 = float(np.mean(t_light_416))

    print(f"Heavy Model (YOLOv8m) @ 640px (BEFORE) : {avg_heavy_640:.2f} ms")
    print(f"Heavy Model (YOLOv8m) @ 416px (AFTER)  : {avg_heavy_416:.2f} ms ({((avg_heavy_640 - avg_heavy_416)/avg_heavy_640)*100:.1f}% faster)")
    print(f"Light Model (YOLOv8n) @ 640px (BEFORE) : {avg_light_640:.2f} ms")
    print(f"Light Model (YOLOv8n) @ 416px (AFTER)  : {avg_light_416:.2f} ms ({((avg_light_640 - avg_light_416)/avg_light_640)*100:.1f}% faster)")

    return {
        "heavy_640_ms": round(avg_heavy_640, 2),
        "heavy_416_ms": round(avg_heavy_416, 2),
        "light_640_ms": round(avg_light_640, 2),
        "light_416_ms": round(avg_light_416, 2),
    }


def run_headless_verification():
    print("=" * 70)
    print("      ECOSTREAM HEADLESS FULL-PASS VERIFICATION (Task e)")
    print("=" * 70)

    if not os.path.exists(DEFAULT_VIDEO_PATH):
        raise FileNotFoundError(f"Video file not found at {DEFAULT_VIDEO_PATH}")

    cap = cv2.VideoCapture(DEFAULT_VIDEO_PATH)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 480)
    video_duration = total_frames / video_fps
    print(f"Video Source      : {DEFAULT_VIDEO_PATH}")
    print(f"Native Video FPS  : {video_fps:.2f}")
    print(f"Total Frame Count : {total_frames} frames ({video_duration:.2f} seconds)")
    print(f"Configuration     : DEMO_MODE={DEMO_MODE}, IMG_SIZE={IMG_SIZE}")

    print("[INFO] Loading YOLO models...")
    heavy_model = YOLO(HEAVY_MODEL_PATH)
    light_model = YOLO(LIGHT_MODEL_PATH)

    ret, sample_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Benchmark before/after imgsz
    benchmarks = measure_before_after_benchmarks(sample_frame, heavy_model, light_model)

    # Start Async Inference Worker
    worker = HeadlessInferenceWorker(heavy_model, light_model)
    worker.start()

    print("\n--- Starting Real-Time Decoupled Playback (1 Full Pass) ---")
    print(f"{'Sec':>3} | {'Hour':>5} | {'Carbon':>6} | {'Vehicles':>8} | {'Active Mode':>16} | {'Active Model':>18} | {'Latency (ms)':>12}")
    print("-" * 85)

    frame_duration = 1.0 / video_fps
    frame_counter = 0
    displayed_frames = 0
    skipped_frames = 0

    first_half_counts = []
    second_half_counts = []
    per_second_logs = []

    start_wall_clock = time.perf_counter()
    next_log_time = 1.0

    while frame_counter < total_frames:
        now = time.perf_counter()
        expected_elapsed = frame_counter * frame_duration
        actual_elapsed = now - start_wall_clock

        # Frame skipping using cap.grab() if behind schedule
        if actual_elapsed - expected_elapsed > frame_duration:
            frames_to_skip = int((actual_elapsed - expected_elapsed) / frame_duration)
            for _ in range(min(frames_to_skip, 5)):
                if not cap.grab():
                    break
                frame_counter += 1
                skipped_frames += 1

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_counter += 1
        displayed_frames += 1
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        progress = pos / total_frames if total_frames > 0 else 0.0

        if DEMO_MODE:
            sim_hour = 3 if progress < 0.5 else 9
            carbon = DEMO_CARBON_3AM if progress < 0.5 else DEMO_CARBON_9AM
        else:
            sim_hour = (frame_counter // 20) % 24
            carbon = 250.0

        frame_resized = cv2.resize(frame, (640, 360))
        wall_time = time.perf_counter() - start_wall_clock

        # Submit to async inference worker
        worker.submit_frame(frame_resized, carbon, sim_hour, frame_counter, wall_time)

        # Retrieve latest boxes and count
        with worker.lock:
            current_mode = worker.current_mode
            current_count = worker.latest_count
            current_model_tag = worker.latest_model_tag
            current_latency = worker.latest_latency_ms

        if progress < 0.5:
            first_half_counts.append(current_count)
        else:
            second_half_counts.append(current_count)

        # Per-second logger
        if wall_time >= next_log_time or frame_counter == total_frames:
            sec_num = int(next_log_time)
            log_entry = {
                "second": sec_num,
                "progress": round(progress, 3),
                "sim_hour": sim_hour,
                "carbon": carbon,
                "vehicle_count": current_count,
                "mode": current_mode,
                "model_tag": current_model_tag,
                "latency_ms": round(current_latency, 2),
            }
            per_second_logs.append(log_entry)
            print(
                f"{sec_num:>3}s | {sim_hour:02d}:00 | {carbon:>6.1f} | {current_count:>8d} | "
                f"{current_mode:>16} | {current_model_tag:>18} | {current_latency:>12.1f}"
            )
            next_log_time += 1.0

        # Frame pacing sleep
        target_next = start_wall_clock + (frame_counter * frame_duration)
        rem = target_next - time.perf_counter()
        if rem > 0.001:
            time.sleep(rem)

    total_elapsed = time.perf_counter() - start_wall_clock
    cap.release()
    worker.running = False
    worker.new_frame_event.set()

    measured_playback_fps = (displayed_frames + skipped_frames) / total_elapsed
    avg_first_half_vehicles = float(np.mean(first_half_counts)) if first_half_counts else 0.0
    avg_second_half_vehicles = float(np.mean(second_half_counts)) if second_half_counts else 0.0
    avg_run_light_latency = float(np.mean(worker.light_latencies)) if worker.light_latencies else 0.0
    avg_run_heavy_latency = float(np.mean(worker.heavy_latencies)) if worker.heavy_latencies else 0.0

    print("-" * 85)
    print("\n" + "=" * 70)
    print("               FINAL VERIFICATION MEASUREMENTS REPORT")
    print("=" * 70)
    print(f"1. Playback FPS               : {measured_playback_fps:.2f} FPS (Target: {video_fps:.1f} FPS)")
    print(f"   Total Playback Elapsed Time: {total_elapsed:.2f} s (for 20.0s video)")
    print(f"   Frames Displayed           : {displayed_frames} (Skipped for real-time catchup: {skipped_frames})")
    print(f"2. Vehicle Count First Half   : {avg_first_half_vehicles:.2f} avg vehicles (3 AM, low traffic)")
    print(f"   Vehicle Count Second Half  : {avg_second_half_vehicles:.2f} avg vehicles (9 AM, rush hour)")
    print(f"3. Mode Switch Transitions    : {len(worker.switch_events)} event(s)")
    for idx, sw in enumerate(worker.switch_events, 1):
        print(f"   Event {idx}: Switched from '{sw['from_mode']}' -> '{sw['to_mode']}'")
        print(f"            At Time       : {sw['time_sec']:.2f}s (Frame {sw['frame_idx']})")
        print(f"            Simulated Hour: {sw['sim_hour']:02d}:00")
        print(f"            Grid Carbon   : {sw['carbon']:.1f} gCO2/kWh")
        print(f"            Vehicle Count : {sw['vehicle_count']} vehicles (Threshold = {DENSITY_THRESHOLD})")

    print(f"4. Measured Inference Latency during run (imgsz={IMG_SIZE}):")
    print(f"   - Light Model (YOLOv8n)    : {avg_run_light_latency:.2f} ms/call ({len(worker.light_latencies)} calls)")
    print(f"   - Heavy Model (YOLOv8m)    : {avg_run_heavy_latency:.2f} ms/call ({len(worker.heavy_latencies)} calls)")
    print(f"5. Before vs After Latency Comparison (Single-Call Benchmarks):")
    print(f"   - Heavy Model Before (640px): {benchmarks['heavy_640_ms']} ms")
    print(f"   - Heavy Model After  (416px): {benchmarks['heavy_416_ms']} ms ({((benchmarks['heavy_640_ms'] - benchmarks['heavy_416_ms'])/benchmarks['heavy_640_ms'])*100:.1f}% reduction)")
    print(f"   - Light Model Before (640px): {benchmarks['light_640_ms']} ms")
    print(f"   - Light Model After  (416px): {benchmarks['light_416_ms']} ms ({((benchmarks['light_640_ms'] - benchmarks['light_416_ms'])/benchmarks['light_640_ms'])*100:.1f}% reduction)")
    print("=" * 70)

    # Save results to JSON
    report_data = {
        "timestamp": time.time(),
        "video_fps": video_fps,
        "measured_playback_fps": round(measured_playback_fps, 2),
        "total_elapsed_sec": round(total_elapsed, 2),
        "displayed_frames": displayed_frames,
        "skipped_frames": skipped_frames,
        "avg_first_half_vehicles": round(avg_first_half_vehicles, 2),
        "avg_second_half_vehicles": round(avg_second_half_vehicles, 2),
        "switch_events": worker.switch_events,
        "measured_light_latency_ms": round(avg_run_light_latency, 2),
        "measured_heavy_latency_ms": round(avg_run_heavy_latency, 2),
        "benchmarks_before_after": benchmarks,
        "per_second_logs": per_second_logs,
    }

    out_file = os.path.join(LOGS_DIR, "headless_verification.json")
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[SUCCESS] Verification report saved to {out_file}\n")


if __name__ == "__main__":
    run_headless_verification()
