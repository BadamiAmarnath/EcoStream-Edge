"""
Headless Verification Script for Multi-Scenario EcoStream Architecture
Evaluates available scenarios headlessly over one full video loop without a browser.
Measures:
  - Average object count in first half vs second half
  - Exact time/frame of light -> heavy mode switch
  - Measured per-model latency (ms) during live execution
  - Real decoupled playback FPS
  - Absence of missing scenario videos (reports without fabricating counts)
"""

import os
import cv2
import time
import json
import threading
import numpy as np
from ultralytics import YOLO

from config import (
    HEAVY_MODEL_PATH,
    LIGHT_MODEL_PATH,
    IMG_SIZE,
    CARBON_HIGH,
    CARBON_LOW,
    DENSITY_THRESHOLD,
    DEMO_MODE,
    ECO_SAFETY_PROBE_INTERVAL,
    LOGS_DIR,
    SCENARIOS,
    interpolate_carbon_profile,
)
from orchestrator import choose_mode, get_mode_metadata


class HeadlessScenarioInferenceWorker(threading.Thread):
    def __init__(self, scenario_config, heavy_model, light_model):
        super().__init__(daemon=True)
        self.scenario = scenario_config
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

    def submit_frame(self, frame, progress, carbon, sim_hour, frame_idx, wall_time):
        with self.lock:
            if self.is_busy:
                return False
            self.pending_frame = frame.copy()
            self.pending_meta = {
                "progress": progress,
                "carbon": carbon,
                "hour": sim_hour,
                "frame_idx": frame_idx,
                "wall_time": wall_time,
            }
            self.is_busy = True
            self.new_frame_event.set()
            return True

    def run(self):
        load_threshold = self.scenario.get("load_threshold", DENSITY_THRESHOLD)
        target_classes = self.scenario.get("target_classes", [2, 3, 5, 7])
        object_label = self.scenario.get("object_label", "objects")
        device_profile = self.scenario.get("device_profile", {})
        heavy_w = device_profile.get("heavy_power_w", 35.0)
        eco_w = device_profile.get("eco_power_w", 12.0)

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

            current_carbon = meta.get("carbon", 300.0)
            sim_hour = meta.get("hour", 3)
            frame_idx = meta.get("frame_idx", 0)
            wall_time = meta.get("wall_time", 0.0)

            t0 = time.perf_counter()

            active_model = (
                self.light_model
                if self.current_mode == "eco" and self.light_model
                else self.heavy_model
            )

            # 1. Detection call on current frame
            boxes = []
            detected_count = 0
            if active_model is not None:
                results = active_model(
                    frame,
                    imgsz=IMG_SIZE,
                    classes=target_classes,
                    verbose=False,
                    conf=0.25,
                )
                boxes = [b.xyxy[0].cpu().numpy().astype(int) for b in results[0].boxes]
                detected_count = len(boxes)

            # 2. Periodic heavy model safety probe when in eco mode
            if (
                self.current_mode == "eco"
                and (self.probe_counter % ECO_SAFETY_PROBE_INTERVAL == 0)
                and self.heavy_model
            ):
                probe_res = self.heavy_model(
                    frame,
                    imgsz=IMG_SIZE,
                    classes=target_classes,
                    verbose=False,
                    conf=0.25,
                )
                probe_cnt = len(probe_res[0].boxes)
                if probe_cnt >= load_threshold and probe_cnt > detected_count:
                    detected_count = probe_cnt

            self.probe_counter += 1

            # 3. Synchronously choose mode from this detection's count
            new_mode = choose_mode(
                current_carbon,
                detected_count,
                self.current_mode,
                load_threshold=load_threshold,
            )

            if new_mode != self.current_mode:
                event_info = {
                    "from_mode": self.current_mode,
                    "to_mode": new_mode,
                    "frame_idx": frame_idx,
                    "time_sec": round(wall_time, 3),
                    "detected_count": detected_count,
                    "carbon": round(current_carbon, 2),
                    "sim_hour": sim_hour,
                }
                self.switch_events.append(event_info)
                self.current_mode = new_mode

            mode_meta = get_mode_metadata(
                self.current_mode,
                current_carbon,
                detected_count,
                load_threshold=load_threshold,
                object_label=object_label,
                heavy_power_w=heavy_w,
                eco_power_w=eco_w,
            )

            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0

            if self.current_mode == "eco":
                self.light_latencies.append(elapsed_ms)
            else:
                self.heavy_latencies.append(elapsed_ms)

            with self.lock:
                self.latest_boxes = boxes
                self.latest_count = detected_count
                self.latest_model_tag = mode_meta["model_tag"]
                self.latest_latency_ms = elapsed_ms
                self.is_busy = False


def verify_scenario_pass(scenario_cfg, heavy_model, light_model):
    s_id = scenario_cfg["id"]
    s_name = scenario_cfg["name"]
    video_path = scenario_cfg.get("video_path")
    carbon_profile = scenario_cfg.get("carbon_profile", [])
    load_threshold = scenario_cfg.get("load_threshold", DENSITY_THRESHOLD)
    object_label = scenario_cfg.get("object_label", "objects")
    timeline_labels = scenario_cfg.get("timeline_labels", {})

    print("\n" + "=" * 75)
    print(f"VERIFYING SCENARIO: {s_name} ({s_id})")
    print("=" * 75)

    if not video_path or not os.path.exists(video_path):
        print(f"[STATUS] Scenario video NOT found at: {video_path}")
        print("         Skipping headless video pass. No simulated counts will be fabricated.")
        return {
            "scenario_id": s_id,
            "status": "VIDEO_MISSING",
            "video_path": video_path,
        }

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 480)
    video_duration = total_frames / video_fps

    print(f"Video File        : {video_path}")
    print(f"Native Video FPS  : {video_fps:.2f}")
    print(f"Total Frame Count : {total_frames} frames ({video_duration:.2f} seconds)")
    print(f"Load Threshold    : {load_threshold} {object_label}")
    print(f"Carbon Profile    : {carbon_profile}")

    worker = HeadlessScenarioInferenceWorker(scenario_cfg, heavy_model, light_model)
    worker.start()

    print("\n--- Starting Real-Time Decoupled Playback (1 Full Pass) ---")
    print(f"{'Sec':>3} | {'Period/Hour':>20} | {'Carbon':>7} | {object_label.capitalize():>10} | {'Active Mode':>16} | {'Active Model':>18} | {'Latency':>10}")
    print("-" * 96)

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

        current_carbon = interpolate_carbon_profile(carbon_profile, progress)
        sim_hour = 3 if progress < 0.5 else 9
        period_text = timeline_labels.get("first", "Period 1") if progress < 0.5 else timeline_labels.get("second", "Period 2")

        frame_resized = cv2.resize(frame, (640, 360))
        wall_time = time.perf_counter() - start_wall_clock

        # Submit to async inference worker
        worker.submit_frame(frame_resized, progress, current_carbon, sim_hour, frame_counter, wall_time)

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
                "period": period_text,
                "carbon": round(current_carbon, 1),
                "detected_count": current_count,
                "mode": current_mode,
                "model_tag": current_model_tag,
                "latency_ms": round(current_latency, 2),
            }
            per_second_logs.append(log_entry)
            print(
                f"{sec_num:>3}s | {period_text:>20} | {current_carbon:>6.1f}g | {current_count:>10d} | "
                f"{current_mode:>16} | {current_model_tag:>18} | {current_latency:>8.1f}ms"
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
    avg_first_half = float(np.mean(first_half_counts)) if first_half_counts else 0.0
    avg_second_half = float(np.mean(second_half_counts)) if second_half_counts else 0.0
    avg_light_latency = float(np.mean(worker.light_latencies)) if worker.light_latencies else 0.0
    avg_heavy_latency = float(np.mean(worker.heavy_latencies)) if worker.heavy_latencies else 0.0

    print("-" * 96)
    print(f"\n[RESULTS FOR {s_name.upper()}]")
    print(f"1. Playback FPS               : {measured_playback_fps:.2f} FPS (Target: {video_fps:.1f} FPS)")
    print(f"   Total Playback Elapsed Time: {total_elapsed:.2f} s (for {video_duration:.2f}s video)")
    print(f"   Frames Displayed           : {displayed_frames} (Skipped: {skipped_frames})")
    print(f"2. Average Scene Load:")
    print(f"   - First Half               : {avg_first_half:.2f} {object_label} ({timeline_labels.get('first')})")
    print(f"   - Second Half              : {avg_second_half:.2f} {object_label} ({timeline_labels.get('second')})")
    print(f"3. Mode Switch Transitions    : {len(worker.switch_events)} event(s)")
    for idx, sw in enumerate(worker.switch_events, 1):
        print(f"   Event {idx}: Switched from '{sw['from_mode']}' -> '{sw['to_mode']}'")
        print(f"            At Time       : {sw['time_sec']:.2f}s (Frame {sw['frame_idx']})")
        print(f"            Carbon (Sim)  : {sw['carbon']:.1f} gCO2/kWh")
        print(f"            Scene Load    : {sw['detected_count']} {object_label} (Threshold = {load_threshold})")

    # Check whether light model exceeded threshold in second half
    if avg_second_half >= load_threshold:
        print(f"   Safety Override Status: Triggered cleanly (average {avg_second_half:.2f} >= threshold {load_threshold})")
    else:
        print(f"   [WARNING] Second half scene load {avg_second_half:.2f} remained below {load_threshold}.")

    print(f"4. Measured Inference Latency during run (imgsz={IMG_SIZE}):")
    print(f"   - Light Model (YOLOv8n)    : {avg_light_latency:.2f} ms/call ({len(worker.light_latencies)} calls)")
    print(f"   - Heavy Model (YOLOv8m)    : {avg_heavy_latency:.2f} ms/call ({len(worker.heavy_latencies)} calls)")

    return {
        "scenario_id": s_id,
        "scenario_name": s_name,
        "status": "COMPLETED",
        "video_fps": video_fps,
        "measured_playback_fps": round(measured_playback_fps, 2),
        "total_elapsed_sec": round(total_elapsed, 2),
        "displayed_frames": displayed_frames,
        "skipped_frames": skipped_frames,
        "avg_first_half_load": round(avg_first_half, 2),
        "avg_second_half_load": round(avg_second_half, 2),
        "switch_events": worker.switch_events,
        "measured_light_latency_ms": round(avg_light_latency, 2),
        "measured_heavy_latency_ms": round(avg_heavy_latency, 2),
        "per_second_logs": per_second_logs,
    }


def run_all_verifications():
    print("=" * 75)
    print("   ECOSTREAM MULTI-SCENARIO HEADLESS VERIFICATION")
    print("=" * 75)

    print("[INFO] Loading shared YOLO models into memory...")
    heavy_model = YOLO(HEAVY_MODEL_PATH)
    light_model = YOLO(LIGHT_MODEL_PATH)

    # Warm up models on dummy frames to avoid PyTorch cold-start latency
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    _ = heavy_model(dummy, imgsz=IMG_SIZE, verbose=False)
    _ = light_model(dummy, imgsz=IMG_SIZE, verbose=False)

    results = {}
    for s_id, s_cfg in SCENARIOS.items():
        res = verify_scenario_pass(s_cfg, heavy_model, light_model)
        results[s_id] = res

    out_file = os.path.join(LOGS_DIR, "headless_verification.json")
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print(f"[SUCCESS] Multi-scenario verification results saved to: {out_file}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_all_verifications()
