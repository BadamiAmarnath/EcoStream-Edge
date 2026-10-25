import os
import cv2
import glob
import pickle
import time
import threading
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template_string, Response, jsonify, abort

from ultralytics import YOLO
from config import (
    BASE_DIR,
    MODELS_DIR,
    DATA_DIR,
    PREDICTOR_PATH,
    HEAVY_MODEL_PATH,
    LIGHT_MODEL_PATH,
    GRID_DATA_PATH,
    CARBON_HIGH,
    CARBON_LOW,
    DENSITY_THRESHOLD,
    HEAVY_POWER_W,
    ECO_POWER_W,
    POWER_SAVED_PCT,
    ECO_SAFETY_PROBE_INTERVAL,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEMO_MODE,
    IMG_SIZE,
    SCENARIOS,
    interpolate_carbon_profile,
)
from orchestrator import predict_carbon, choose_mode, get_mode_metadata, log_decision

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ecostream.app")

# Precompute historical hourly mean fallback values from CSV
try:
    if os.path.exists(GRID_DATA_PATH):
        df_csv = pd.read_csv(GRID_DATA_PATH)
        HOURLY_FALLBACK = df_csv.groupby("hour")["carbon_intensity"].mean().to_dict()
    else:
        HOURLY_FALLBACK = {h: 250.0 for h in range(24)}
except Exception as e:
    logger.warning("Could not precompute hourly carbon fallback: %s", e)
    HOURLY_FALLBACK = {h: 250.0 for h in range(24)}

# Load Predictor and YOLO Models into memory ONCE (Shared across all scenarios)
grid_predictor = None
if os.path.exists(PREDICTOR_PATH):
    try:
        with open(PREDICTOR_PATH, "rb") as f:
            grid_predictor = pickle.load(f)
        logger.info("Grid carbon predictor loaded successfully from %s", PREDICTOR_PATH)
    except Exception as e:
        logger.error("Failed to load grid predictor: %s", e)
else:
    logger.warning("Predictor not found at %s. Please run train_predictor.py", PREDICTOR_PATH)

logger.info("Pre-warming YOLO models into system memory with IMG_SIZE=%d...", IMG_SIZE)
model_heavy = None
model_light = None
yolo_lock = threading.Lock()

if os.path.exists(HEAVY_MODEL_PATH):
    model_heavy = YOLO(HEAVY_MODEL_PATH)
    logger.info("Heavy Model loaded at memory address: %s", hex(id(model_heavy)))
else:
    logger.warning("Heavy model not found at %s. Please run download_models.py", HEAVY_MODEL_PATH)

if os.path.exists(LIGHT_MODEL_PATH):
    model_light = YOLO(LIGHT_MODEL_PATH)
    logger.info("Eco Model loaded at memory address  : %s", hex(id(model_light)))
else:
    logger.warning("Light model not found at %s. Please run download_models.py", LIGHT_MODEL_PATH)

# Warm up PyTorch computation graph on dummy frames
try:
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    if model_heavy:
        _ = model_heavy(dummy_frame, imgsz=IMG_SIZE, verbose=False)
    if model_light:
        _ = model_light(dummy_frame, imgsz=IMG_SIZE, verbose=False)
    logger.info("YOLO models pre-warmed successfully.")
except Exception as e:
    logger.warning("YOLO model warmup encountered warning: %s", e)


# ==============================================================================
# Multi-Scenario Pipeline Engine
# ==============================================================================


class ScenarioInferenceWorker(threading.Thread):
    """
    Dedicated worker thread that processes YOLO inference for a single scenario.
    Runs asynchronously and independently from video playback.
    """

    def __init__(self, scenario_config, shared_heavy, shared_light, model_mutex, telemetry_ref, telemetry_mutex):
        super().__init__(daemon=True)
        self.scenario = scenario_config
        self.heavy_model = shared_heavy
        self.light_model = shared_light
        self.model_lock = model_mutex
        self.telemetry = telemetry_ref
        self.telemetry_lock = telemetry_mutex

        self.running = True
        self.is_busy = False
        self.new_frame_event = threading.Event()
        self.lock = threading.Lock()

        # Input slots
        self.pending_frame = None
        self.pending_meta = {}

        # Worker state
        self.current_mode = "eco" if DEMO_MODE else "heavy_default"
        self.latest_boxes = []
        self.latest_count = 0
        self.latest_color = (50, 220, 100) if DEMO_MODE else (255, 165, 0)
        self.latest_model_tag = "YOLOv8n (Eco Core)" if DEMO_MODE else "YOLOv8m (Heavy Core)"
        self.latest_state = "Eco-Mode Abatement" if DEMO_MODE else "Default Mode"
        self.active_ptr = hex(id(shared_light))[-6:] if shared_light else "000000"
        self.probe_counter = 0

    def submit_frame(self, frame, progress, sim_hour, current_carbon):
        with self.lock:
            if self.is_busy:
                return False
            self.pending_frame = frame.copy()
            self.pending_meta = {
                "progress": progress,
                "hour": sim_hour,
                "carbon": current_carbon,
            }
            self.is_busy = True
            self.new_frame_event.set()
            return True

    def run(self):
        scenario_id = self.scenario["id"]
        load_threshold = self.scenario.get("load_threshold", DENSITY_THRESHOLD)
        target_classes = self.scenario.get("target_classes", [2, 3, 5, 7])
        object_label = self.scenario.get("object_label", "objects")
        device_profile = self.scenario.get("device_profile", {})
        heavy_w = device_profile.get("heavy_power_w", HEAVY_POWER_W)
        eco_w = device_profile.get("eco_power_w", ECO_POWER_W)

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

            progress = meta.get("progress", 0.0)
            current_carbon = meta.get("carbon", 300.0)
            sim_hour = meta.get("hour", 0)

            t0 = time.perf_counter()

            # Active model pointer
            active_model = (
                self.light_model
                if self.current_mode == "eco" and self.light_model
                else self.heavy_model
            )

            # 1. Run detection on current frame with thread-safe lock on shared YOLO weights
            boxes = []
            detected_count = 0
            if active_model is not None:
                with self.model_lock:
                    results = active_model(
                        frame,
                        imgsz=IMG_SIZE,
                        classes=target_classes,
                        verbose=False,
                        conf=0.25,
                    )
                boxes = [b.xyxy[0].cpu().numpy().astype(int) for b in results[0].boxes]
                detected_count = len(boxes)

            # 2. Periodic heavy model safety probe if in eco mode to avoid undercounting
            if (
                self.current_mode == "eco"
                and (self.probe_counter % ECO_SAFETY_PROBE_INTERVAL == 0)
                and self.heavy_model
            ):
                with self.model_lock:
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

            # 3. Synchronously evaluate mode decision based on THIS frame's actual count
            new_mode = choose_mode(
                current_carbon,
                detected_count,
                self.current_mode,
                load_threshold=load_threshold,
            )

            if new_mode != self.current_mode:
                mode_meta = get_mode_metadata(
                    new_mode,
                    current_carbon,
                    detected_count,
                    load_threshold=load_threshold,
                    object_label=object_label,
                    heavy_power_w=heavy_w,
                    eco_power_w=eco_w,
                )
                log_decision(
                    simulated_hour=sim_hour,
                    carbon=current_carbon,
                    object_count=detected_count,
                    mode=new_mode,
                    reason=mode_meta["state_label"],
                )
                logger.info(
                    "[%s TRANSITION] Mode: %s | Carbon: %.1f | %s: %d | %s",
                    scenario_id.upper(),
                    new_mode,
                    current_carbon,
                    object_label.capitalize(),
                    detected_count,
                    mode_meta["state_label"],
                )
                self.current_mode = new_mode

            # 4. Generate metadata & explanation using the EXACT SAME count and active mode
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
            elapsed_sec = elapsed_ms / 1000.0

            # 5. Calculate estimated energy integration using measured latency
            est_power = mode_meta["power_w"]
            baseline_joules = heavy_w * elapsed_sec
            ecostream_joules = est_power * elapsed_sec

            # Update shared scenario telemetry without lag
            with self.telemetry_lock:
                self.telemetry["cum_baseline_energy"] += baseline_joules
                self.telemetry["cum_ecostream_energy"] += ecostream_joules

                delta_joules = baseline_joules - ecostream_joules
                if delta_joules > 0:
                    delta_carbon_grams = (delta_joules / 3600000.0) * current_carbon
                    self.telemetry["cum_carbon_saved_g"] += delta_carbon_grams

                saved_joules = (
                    self.telemetry["cum_baseline_energy"] - self.telemetry["cum_ecostream_energy"]
                )
                energy_saved_pct = (
                    (saved_joules / self.telemetry["cum_baseline_energy"]) * 100.0
                    if self.telemetry["cum_baseline_energy"] > 0
                    else 0.0
                )

                # Determine timeline display text
                timeline_labels = self.scenario.get("timeline_labels", {})
                first_label = timeline_labels.get("first", f"{sim_hour:02d}:00")
                second_label = timeline_labels.get("second", f"{sim_hour:02d}:00")
                timeline_text = first_label if progress < 0.5 else second_label

                self.telemetry["carbon_intensity"] = current_carbon
                self.telemetry["simulated_hour"] = sim_hour
                self.telemetry["simulated_hour_text"] = timeline_text
                self.telemetry["detected_count"] = detected_count
                self.telemetry["active_model"] = mode_meta["model_tag"]
                self.telemetry["active_model_params"] = "25.9M" if mode_meta["model_key"] == "heavy" else "3.2M"
                self.telemetry["model_memory_address"] = (
                    hex(id(active_model)) if active_model else "0x000000000000"
                )
                self.telemetry["state"] = mode_meta["state_label"]
                self.telemetry["latency_ms"] = elapsed_ms
                self.telemetry["power_watts"] = est_power
                self.telemetry["energy_saved_pct"] = energy_saved_pct
                # Explanation built from the exact same count and mode
                self.telemetry["xai_explanation"] = mode_meta["explanation"]

            # Update worker boxes for playback overlay
            with self.lock:
                self.latest_boxes = boxes
                self.latest_count = detected_count
                self.latest_model_tag = mode_meta["model_tag"]
                self.latest_state = mode_meta["state_label"]
                self.latest_color = mode_meta["color_bgr"]
                self.active_ptr = hex(id(active_model))[-6:] if active_model else "000000"
                self.is_busy = False


class ScenarioPlaybackWorker(threading.Thread):
    """
    Dedicated playback thread for a single scenario.
    Synchronizes strictly with wall-clock time at native video FPS.
    Skips behind-schedule frames using cap.grab().
    Encodes latest JPEG buffer.
    """

    def __init__(self, scenario_config, inference_worker, telemetry_ref, telemetry_mutex):
        super().__init__(daemon=True)
        self.scenario = scenario_config
        self.inference_worker = inference_worker
        self.telemetry = telemetry_ref
        self.telemetry_lock = telemetry_mutex

        self.latest_jpeg_bytes = None
        self.lock = threading.Lock()
        self.running = True

    def build_missing_video_frame(self, video_path):
        """Draw an informative standby frame when a scenario video is missing."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:] = (15, 23, 42)

        # Draw cautionary border
        cv2.rectangle(frame, (8, 8), (632, 352), (245, 158, 11), 2)
        cv2.putText(frame, "VIDEO FILE NOT FOUND", (170, 130), cv2.FONT_HERSHEY_DUPLEX, 0.8, (245, 158, 11), 2, cv2.LINE_AA)

        clean_path = os.path.relpath(video_path, BASE_DIR) if os.path.isabs(video_path) else video_path
        cv2.putText(frame, "Please place your video file at:", (140, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 232, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, clean_path, (140, 215), cv2.FONT_HERSHEY_DUPLEX, 0.58, (56, 189, 248), 1, cv2.LINE_AA)
        cv2.putText(frame, "Pipeline awaiting input footage...", (180, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)
        return frame

    def run(self):
        video_path = self.scenario.get("video_path")
        carbon_profile = self.scenario.get("carbon_profile", [])
        object_label = self.scenario.get("object_label", "objects")

        video_exists = bool(video_path and os.path.exists(video_path))
        with self.telemetry_lock:
            self.telemetry["video_missing"] = not video_exists

        cap = None
        video_fps = 24.0
        total_frames = 480.0

        if video_exists:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 480.0

        frame_duration = 1.0 / video_fps
        frame_counter = 0
        start_wall_clock = time.perf_counter()

        while self.running:
            if not video_exists or not cap or not cap.isOpened():
                # Render missing-video placeholder and sleep
                frame = self.build_missing_video_frame(video_path)
                ret_enc, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret_enc:
                    with self.lock:
                        self.latest_jpeg_bytes = buffer.tobytes()
                time.sleep(0.5)
                continue

            # 1. Wall-clock synchronization & frame skipping using cap.grab()
            now = time.perf_counter()
            expected_elapsed = frame_counter * frame_duration
            actual_elapsed = now - start_wall_clock

            if actual_elapsed - expected_elapsed > frame_duration:
                frames_to_skip = int((actual_elapsed - expected_elapsed) / frame_duration)
                for _ in range(min(frames_to_skip, 5)):
                    if not cap.grab():
                        break
                    frame_counter += 1

            # 2. Retrieve frame
            ret, frame = cap.read()
            if not ret or frame is None:
                # Video loop point -> reset video to frame 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                start_wall_clock = time.perf_counter()
                frame_counter = 0

                # Reset cumulative telemetry counters per loop
                with self.telemetry_lock:
                    self.telemetry["cum_baseline_energy"] = 0.0
                    self.telemetry["cum_ecostream_energy"] = 0.0
                    self.telemetry["cum_carbon_saved_g"] = 0.0
                    self.telemetry["energy_saved_pct"] = 0.0

                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(frame_duration)
                    continue

            frame_counter += 1
            frame_resized = cv2.resize(frame, (640, 360))

            # 3. Calculate video progress & simulated carbon profile
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            progress = pos / total_frames if total_frames > 0 else 0.0

            if DEMO_MODE:
                current_carbon = interpolate_carbon_profile(carbon_profile, progress)
                sim_hour = 3 if progress < 0.5 else 9
            else:
                sim_hour = (frame_counter // 20) % 24
                current_carbon = predict_carbon(
                    grid_predictor,
                    hour=sim_hour,
                    fallback=HOURLY_FALLBACK.get(sim_hour, 250.0),
                )

            # 4. Submit frame to async inference worker (non-blocking)
            self.inference_worker.submit_frame(frame_resized, progress, sim_hour, current_carbon)

            # 5. Fetch latest detection annotations
            with self.inference_worker.lock:
                boxes = self.inference_worker.latest_boxes
                color = self.inference_worker.latest_color
                tag = self.inference_worker.latest_model_tag
                state = self.inference_worker.latest_state
                ptr = self.inference_worker.active_ptr
                count = self.inference_worker.latest_count

            # 6. Draw annotations on displayed frame
            cv2.rectangle(frame_resized, (2, 2), (637, 357), color, 2)
            for xyxy in boxes:
                cv2.rectangle(frame_resized, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)

            # In-video HUD (using generic labels)
            cv2.rectangle(frame_resized, (10, 10), (370, 62), (15, 20, 30), -1)
            cv2.rectangle(frame_resized, (10, 10), (370, 62), color, 1)
            cv2.putText(
                frame_resized,
                f"CORE: {tag.split()[0]} [0x{ptr}] | LOAD: {count} {object_label}",
                (20, 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.46,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_resized,
                f"STATE: {state}",
                (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (200, 220, 240),
                1,
                cv2.LINE_AA,
            )

            # 7. JPEG Frame Encoding
            ret_enc, buffer = cv2.imencode(".jpg", frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret_enc:
                with self.lock:
                    self.latest_jpeg_bytes = buffer.tobytes()

            # 8. Pacing sleep
            target_next = start_wall_clock + (frame_counter * frame_duration)
            remaining_sleep = target_next - time.perf_counter()
            if remaining_sleep > 0.001:
                time.sleep(remaining_sleep)

        if cap:
            cap.release()


class ScenarioPipeline:
    """
    Manages execution lifecycle for a single scenario.
    Starts on first page view; automatically stops after 15s of inactivity.
    """

    def __init__(self, scenario_config, shared_heavy, shared_light, model_mutex):
        self.scenario = scenario_config
        self.heavy_model = shared_heavy
        self.light_model = shared_light
        self.model_lock = model_mutex

        self.last_viewer_time = 0.0
        self.is_active = False
        self.lifecycle_lock = threading.Lock()

        device = scenario_config.get("device_profile", {})
        heavy_w = device.get("heavy_power_w", HEAVY_POWER_W)
        eco_w = device.get("eco_power_w", ECO_POWER_W)

        self.telemetry_lock = threading.Lock()
        self.telemetry = {
            "scenario_id": scenario_config["id"],
            "scenario_name": scenario_config["name"],
            "object_label": scenario_config.get("object_label", "objects"),
            "load_threshold": scenario_config.get("load_threshold", DENSITY_THRESHOLD),
            "video_missing": not os.path.exists(scenario_config.get("video_path", "")),
            "carbon_intensity": 300.0,
            "simulated_hour": 3,
            "simulated_hour_text": scenario_config.get("timeline_labels", {}).get("first", "Period 1"),
            "detected_count": 0,
            "active_model": "YOLOv8n (Eco Core)" if DEMO_MODE else "YOLOv8m (Heavy Core)",
            "active_model_params": "3.2M" if DEMO_MODE else "25.9M",
            "model_memory_address": hex(id(shared_light)) if shared_light else "0x000000000000",
            "state": "Eco-Mode Abatement" if DEMO_MODE else "Default Mode",
            "latency_ms": 0.0,
            "power_watts": eco_w if DEMO_MODE else heavy_w,
            "cum_baseline_energy": 0.0,
            "cum_ecostream_energy": 0.0,
            "energy_saved_pct": 0.0,
            "cum_carbon_saved_g": 0.0,
            "xai_explanation": "Initializing neural orchestration runtime...",
            "demo_mode": DEMO_MODE,
        }

        self.inference_worker = None
        self.playback_worker = None

    def touch(self):
        """Register viewer presence; activate pipeline workers if idle."""
        self.last_viewer_time = time.time()
        with self.lifecycle_lock:
            if not self.is_active:
                self.start_workers()

    def start_workers(self):
        logger.info("[LIFECYCLE] Activating pipeline workers for scenario '%s'...", self.scenario["id"])
        self.inference_worker = ScenarioInferenceWorker(
            self.scenario,
            self.heavy_model,
            self.light_model,
            self.model_lock,
            self.telemetry,
            self.telemetry_lock,
        )
        self.playback_worker = ScenarioPlaybackWorker(
            self.scenario,
            self.inference_worker,
            self.telemetry,
            self.telemetry_lock,
        )
        self.inference_worker.start()
        self.playback_worker.start()
        self.is_active = True

    def stop_workers(self):
        with self.lifecycle_lock:
            if not self.is_active:
                return
            logger.info("[LIFECYCLE] Deactivating pipeline workers for scenario '%s' (Inactivity Timeout)...", self.scenario["id"])
            if self.inference_worker:
                self.inference_worker.running = False
                self.inference_worker.new_frame_event.set()
            if self.playback_worker:
                self.playback_worker.running = False
            self.is_active = False

    def get_latest_jpeg(self):
        if self.playback_worker:
            with self.playback_worker.lock:
                return self.playback_worker.latest_jpeg_bytes
        return None


class ScenarioManager:
    """Registry of all scenario pipelines with background inactivity reaper."""

    def __init__(self, scenarios, heavy_m, light_m, model_mutex):
        self.pipelines = {
            s_id: ScenarioPipeline(cfg, heavy_m, light_m, model_mutex)
            for s_id, cfg in scenarios.items()
        }
        self.janitor_thread = threading.Thread(target=self._janitor_loop, daemon=True)
        self.janitor_thread.start()

    def get_pipeline(self, scenario_id):
        return self.pipelines.get(scenario_id)

    def _janitor_loop(self):
        """Stop worker threads after 15 seconds without viewers to conserve CPU."""
        while True:
            time.sleep(2.0)
            now = time.time()
            for s_id, pipeline in self.pipelines.items():
                if pipeline.is_active and (now - pipeline.last_viewer_time > 15.0):
                    pipeline.stop_workers()


scenario_manager = ScenarioManager(SCENARIOS, model_heavy, model_light, yolo_lock)


# ==============================================================================
# UI HTML Templates
# ==============================================================================

LANDING_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EcoStream AI | Scenario Catalog</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #080c14;
            --card-bg: rgba(17, 24, 39, 0.75);
            --accent-cyan: #38bdf8;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: radial-gradient(circle at top right, #111e38, var(--bg-dark) 85%);
            color: #f1f5f9;
            padding: 30px 20px;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 28px; }
        .brand-logo { font-size: 26px; font-weight: 800; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .chips-group { display: flex; gap: 10px; align-items: center; }
        .chip { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; padding: 5px 14px; border-radius: 99px; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
        
        .hero-banner {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 28px;
        }
        .hero-title { font-size: 20px; font-weight: 800; color: #f8fafc; margin-bottom: 8px; }
        .hero-desc { font-size: 14px; color: #94a3b8; line-height: 1.5; }

        .scenarios-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 22px;
        }
        .scenario-card {
            background: var(--card-bg);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 22px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: all 0.25s ease;
        }
        .scenario-card:hover {
            transform: translateY(-3px);
            border-color: var(--accent-cyan);
            box-shadow: 0 12px 30px rgba(56, 189, 248, 0.15);
        }
        .card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
        .card-title { font-size: 18px; font-weight: 800; color: #f1f5f9; }
        .badge-ready { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
        .badge-missing { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

        .card-desc { font-size: 13px; color: #94a3b8; margin-bottom: 16px; line-height: 1.45; }
        .meta-list { display: flex; flex-direction: column; gap: 8px; margin-bottom: 20px; font-size: 12px; }
        .meta-row { display: flex; justify-content: space-between; padding: 6px 10px; background: rgba(10, 15, 29, 0.6); border-radius: 8px; }
        .meta-key { color: #64748b; font-weight: 600; }
        .meta-val { color: #cbd5e1; font-family: 'JetBrains Mono', monospace; font-weight: 700; }

        .btn-open {
            display: block;
            text-align: center;
            background: linear-gradient(135deg, #0284c7, #2563eb);
            color: #ffffff;
            text-decoration: none;
            padding: 11px 18px;
            border-radius: 10px;
            font-weight: 700;
            font-size: 13px;
            transition: all 0.2s ease;
        }
        .btn-open:hover {
            background: linear-gradient(135deg, #38bdf8, #3b82f6);
            box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="brand-logo">⚡ EcoStream: Edge Vision Orchestrator</div>
            <div class="chips-group">
                <div class="chip">MULTI-SCENARIO PLATFORM ACTIVE</div>
            </div>
        </div>

        <div class="hero-banner">
            <div class="hero-title">Carbon-Aware Intelligent Edge Workload Management</div>
            <div class="hero-desc">
                Select an edge analytics scenario below to launch the live telemetry dashboard. The orchestrator autonomously dynamically switches between high-fidelity and eco-abatement vision models based on real-time grid carbon intensity and scene load safety thresholds.
            </div>
        </div>

        <div class="scenarios-grid">
            {% for sc in scenarios %}
            <div class="scenario-card">
                <div>
                    <div class="card-header">
                        <div class="card-title">{{ sc.name }}</div>
                        {% if sc.video_exists %}
                        <span class="badge-ready">VIDEO READY</span>
                        {% else %}
                        <span class="badge-missing">VIDEO MISSING</span>
                        {% endif %}
                    </div>
                    <div class="card-desc">{{ sc.description }}</div>

                    <div class="meta-list">
                        <div class="meta-row">
                            <span class="meta-key">Target Objects</span>
                            <span class="meta-val">{{ sc.object_label|title }} ({{ sc.class_names|join(', ') }})</span>
                        </div>
                        <div class="meta-row">
                            <span class="meta-key">Scene Load Threshold</span>
                            <span class="meta-val">≥ {{ sc.load_threshold }} {{ sc.object_label }}</span>
                        </div>
                        <div class="meta-row">
                            <span class="meta-key">Assumed Edge Device</span>
                            <span class="meta-val">{{ sc.device_profile.name }}</span>
                        </div>
                        <div class="meta-row">
                            <span class="meta-key">Power Assumptions</span>
                            <span class="meta-val">{{ sc.device_profile.heavy_power_w }}W (Heavy) • {{ sc.device_profile.eco_power_w }}W (Eco)</span>
                        </div>
                    </div>
                </div>

                <a href="/scenario/{{ sc.id }}" class="btn-open">Open Scenario Dashboard →</a>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""


DASHBOARD_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EcoStream AI | {{ scenario.name }}</title>
    <!-- Pinned Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #080c14;
            --card-bg: rgba(17, 24, 39, 0.75);
            --accent-cyan: #38bdf8;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Plus Jakarta Sans', sans-serif; 
            background: radial-gradient(circle at top right, #111e38, var(--bg-dark) 85%); 
            color: #f1f5f9; 
            padding: 20px; 
            min-height: 100vh;
        }
        .header { display: flex; justify-content: space-between; align-items: center; max-width: 1540px; margin: 0 auto 16px auto; }
        .nav-back { color: var(--accent-cyan); text-decoration: none; font-size: 13px; font-weight: 700; display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; transition: all 0.2s; }
        .nav-back:hover { background: rgba(56, 189, 248, 0.2); }
        .brand-logo { font-size: 22px; font-weight: 800; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .chips-group { display: flex; gap: 10px; align-items: center; }
        .live-chip { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; padding: 4px 12px; border-radius: 99px; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
        .demo-chip { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); color: #fbbf24; padding: 4px 12px; border-radius: 99px; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
        
        .container { max-width: 1540px; margin: auto; display: grid; grid-template-columns: 1.8fr 1.2fr; gap: 20px; }
        .glass-card { background: var(--card-bg); backdrop-filter: blur(14px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 18px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        
        .video-wrapper { position: relative; border-radius: 10px; overflow: hidden; border: 2px solid rgba(255,255,255,0.1); background: #000; min-height: 360px; display: flex; align-items: center; justify-content: center; }
        .video-wrapper img { width: 100%; height: auto; display: block; }
        
        .alert-missing { background: rgba(245, 158, 11, 0.15); border-left: 4px solid var(--accent-amber); padding: 12px 14px; border-radius: 8px; font-size: 12px; color: #fde68a; line-height: 1.4; margin-bottom: 12px; }

        /* Dual Architecture Switcher Cards */
        .model-switcher { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
        .model-card { background: rgba(10, 15, 29, 0.7); border: 2px solid rgba(255, 255, 255, 0.06); border-radius: 10px; padding: 12px; transition: all 0.3s ease; opacity: 0.45; transform: scale(0.98); }
        .model-card.active-eco { opacity: 1.0; transform: scale(1.0); border-color: var(--accent-green); box-shadow: 0 0 20px rgba(16, 185, 129, 0.3); }
        .model-card.active-heavy { opacity: 1.0; transform: scale(1.0); border-color: var(--accent-red); box-shadow: 0 0 20px rgba(239, 68, 68, 0.3); }
        .model-title { font-size: 13px; font-weight: 700; display: flex; justify-content: space-between; align-items: center; }
        .model-specs { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94a3b8; margin-top: 6px; line-height: 1.4; }

        /* Metric Grid */
        .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }
        .metric-pill { background: rgba(10, 15, 29, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 12px; }
        .metric-label { font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
        .metric-val { font-size: 20px; font-weight: 800; font-family: 'JetBrains Mono', monospace; margin-top: 4px; }

        /* Progress Bar */
        .progress-box { background: rgba(10, 15, 29, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 12px; margin-bottom: 12px; }
        .progress-bg { width: 100%; height: 8px; background: rgba(255,255,255,0.08); border-radius: 99px; overflow: hidden; margin-top: 6px; }
        .progress-fill { height: 100%; width: 0%; background: linear-gradient(90deg, #10b981, #38bdf8); border-radius: 99px; transition: width 0.3s; }

        /* Log Ticker */
        .log-box { background: #030712; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 10px; height: 105px; overflow-y: hidden; font-family: 'JetBrains Mono', monospace; font-size: 11px; display: flex; flex-direction: column-reverse; gap: 4px; }
        .log-item { color: #94a3b8; border-left: 2px solid #38bdf8; padding-left: 6px; }

        /* Two Dedicated Charts */
        .charts-container { display: flex; flex-direction: column; gap: 12px; }
        .chart-card { background: rgba(10, 15, 29, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px 12px; height: 135px; transition: all 0.3s ease; }
        .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
        .chart-title { font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
        .chart-legend { font-size: 10px; font-family: 'JetBrains Mono', monospace; font-weight: 700; }
        .chart-wrapper { height: 100px; position: relative; }
    </style>
</head>
<body>
    <div class="header">
        <div style="display:flex; align-items:center; gap:16px;">
            <a href="/" class="nav-back">← All Scenarios</a>
            <div class="brand-logo">⚡ {{ scenario.name }}</div>
        </div>
        <div class="chips-group">
            {% if demo_mode %}
            <div class="demo-chip">⚡ DEMO MODE — SIMULATED CARBON</div>
            {% endif %}
            <div class="live-chip">ASSUMED: {{ scenario.device_profile.name }}</div>
        </div>
    </div>

    <div class="container">
        <!-- Left Column: Video Stream & XAI Rationale -->
        <div class="glass-card" style="display: flex; flex-direction: column; gap: 14px;">
            {% if not video_exists %}
            <div class="alert-missing">
                <strong>⚠️ Missing Footage:</strong> Put your video at <code>{{ scenario.video_path }}</code> to stream live video for this scenario. The app is running smoothly in standby mode.
            </div>
            {% endif %}

            <div class="video-wrapper">
                <img src="/video_feed/{{ scenario.id }}" alt="Connecting Stream..." onerror="this.src='/fallback_frame';" />
            </div>

            <!-- Explainable AI Explanation -->
            <div style="background: rgba(15, 23, 42, 0.85); border-left: 4px solid var(--accent-cyan); padding: 14px; border-radius: 8px;">
                <div style="font-size:11px; font-weight:800; color:var(--accent-cyan); text-transform:uppercase;">🔍 XAI Runtime Decision Rationale</div>
                <div id="xai-text" style="font-size:13px; color:#cbd5e1; margin-top:5px; line-height:1.4;">Evaluating scene load and carbon context...</div>
            </div>

            <!-- Transition Log Feed -->
            <div>
                <div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; margin-bottom:6px;">Runtime Transition Log</div>
                <div class="log-box" id="log-feed">
                    <div class="log-item">System ready. Dedicated scenario pipeline active.</div>
                </div>
            </div>
        </div>

        <!-- Right Column: Specs, Metrics & Dual Charts -->
        <div class="glass-card">
            <!-- Simulated Timeline / Period Indicator -->
            <div class="metric-pill" style="margin-bottom:12px; border-left: 3px solid var(--accent-cyan);">
                <div class="metric-label">Simulated Timeline & Context</div>
                <div class="metric-val" id="sim-hour" style="font-size:15px; color:#38bdf8;">Loading period...</div>
            </div>

            <!-- Dual Model Spec Cards -->
            <div class="model-switcher">
                <div class="model-card" id="card-heavy">
                    <div class="model-title">
                        <span style="color:#ef4444;">HIGH-FIDELITY CORE</span>
                        <span id="badge-heavy" style="font-size:10px; padding:2px 6px; border-radius:4px; background:#334155;">STANDBY</span>
                    </div>
                    <div class="model-specs">
                        Weight: heavy_yolo.pt<br>
                        Params: 25.9M (YOLOv8m)<br>
                        Assumed Power: {{ scenario.device_profile.heavy_power_w }}W
                    </div>
                </div>
                <div class="model-card" id="card-eco">
                    <div class="model-title">
                        <span style="color:#10b981;">ECO-ABATEMENT CORE</span>
                        <span id="badge-eco" style="font-size:10px; padding:2px 6px; border-radius:4px; background:#334155;">STANDBY</span>
                    </div>
                    <div class="model-specs">
                        Weight: light_yolo.pt<br>
                        Params: 3.2M (YOLOv8n)<br>
                        Assumed Power: {{ scenario.device_profile.eco_power_w }}W ({{ scenario.device_profile.power_saved_pct|round|int }}% Saved)
                    </div>
                </div>
            </div>

            <!-- In-Memory Object Pointer Address Proof -->
            <div class="metric-pill" style="margin-bottom:12px;">
                <div class="metric-label">In-Memory Object Pointer Address</div>
                <div class="metric-val" id="mem-ptr" style="font-size:13px; color:#a855f7;">0x000000000000</div>
            </div>

            <!-- Metrics Grid -->
            <div class="metric-grid">
                <div class="metric-pill">
                    <div class="metric-label">Grid Carbon Intensity</div>
                    <div class="metric-val" id="carbon" style="color:#f59e0b;">--</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-label">Detected {{ scenario.object_label|title }}</div>
                    <div class="metric-val" id="objects" style="color:#38bdf8;">--</div>
                </div>
            </div>

            <!-- Progress Box with Configurable Power Assumptions -->
            <div class="progress-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="metric-label">Estimated energy saved (current loop)</div>
                        <div style="font-size:10px; color:#64748b; margin-top:2px;">Assumed Edge Device: {{ scenario.device_profile.name }} ({{ scenario.device_profile.heavy_power_w }}W vs {{ scenario.device_profile.eco_power_w }}W)</div>
                    </div>
                    <span id="energy-pct" style="font-family:'JetBrains Mono'; font-weight:700; color:#10b981; font-size:16px;">0.0%</span>
                </div>
                <div class="progress-bg"><div class="progress-fill" id="energy-bar"></div></div>
            </div>

            <div class="metric-grid">
                <div class="metric-pill">
                    <div class="metric-label">Estimated Carbon Abated (Current Loop)</div>
                    <div class="metric-val" id="carbon-saved" style="color:#10b981;">0.000 g</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-label">Measured Execution Latency & Assumed Power</div>
                    <div class="metric-val" id="power-latency" style="font-size:14px; color:#cbd5e1;">--</div>
                </div>
            </div>

            <!-- Dual Synchronized Real-Time Charts (60 data points window) -->
            <div class="charts-container">
                <!-- Chart A: Grid Carbon vs Limit -->
                <div class="chart-card" id="cardChartCarbon">
                    <div class="chart-header">
                        <span class="chart-title">Chart A: Grid Carbon vs Limit {% if demo_mode %}(Simulated){% endif %}</span>
                        <span class="chart-legend" style="color:#38bdf8;">Carbon vs Limit ({{ carbon_high|round|int }} g)</span>
                    </div>
                    <div class="chart-wrapper">
                        <canvas id="chartCarbon"></canvas>
                    </div>
                </div>

                <!-- Chart B: Detected Objects vs Load Threshold -->
                <div class="chart-card" id="cardChartLoad">
                    <div class="chart-header">
                        <span class="chart-title">Chart B: Detected {{ scenario.object_label|title }} vs Load Threshold</span>
                        <span class="chart-legend" style="color:#10b981;">Count vs Threshold ({{ scenario.load_threshold }})</span>
                    </div>
                    <div class="chart-wrapper">
                        <canvas id="chartLoad"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const WINDOW_SIZE = 60;
        const initialLabels = Array(WINDOW_SIZE).fill('');

        // Chart A: Grid Carbon vs Limit
        const ctxCarbon = document.getElementById('chartCarbon').getContext('2d');
        const chartCarbon = new Chart(ctxCarbon, {
            type: 'line',
            data: {
                labels: initialLabels.slice(),
                datasets: [
                    {
                        label: 'Grid Carbon {% if demo_mode %}(Simulated){% endif %}',
                        data: Array(WINDOW_SIZE).fill(300),
                        borderColor: '#38bdf8',
                        backgroundColor: 'rgba(56, 189, 248, 0.08)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: 'Limit ({{ carbon_high|round|int }})',
                        data: Array(WINDOW_SIZE).fill({{ carbon_high }}),
                        borderColor: '#ef4444',
                        borderDash: [4, 4],
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        min: 220,
                        max: 380,
                        ticks: { color: '#64748b', font: { size: 9 } },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    },
                    x: { display: false }
                },
                plugins: { legend: { display: false } }
            }
        });

        // Chart B: Detected Objects vs Load Threshold
        const ctxLoad = document.getElementById('chartLoad').getContext('2d');
        const chartLoad = new Chart(ctxLoad, {
            type: 'line',
            data: {
                labels: initialLabels.slice(),
                datasets: [
                    {
                        label: 'Detected {{ scenario.object_label|title }}',
                        data: Array(WINDOW_SIZE).fill(0),
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.08)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: 'Load Threshold ({{ scenario.load_threshold }})',
                        data: Array(WINDOW_SIZE).fill({{ scenario.load_threshold }}),
                        borderColor: '#f59e0b',
                        borderDash: [4, 4],
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        min: 0,
                        max: Math.max(20, {{ scenario.load_threshold }} + 10),
                        ticks: { color: '#64748b', font: { size: 9 }, stepSize: 5 },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    },
                    x: { display: false }
                },
                plugins: { legend: { display: false } }
            }
        });

        let lastModel = "";

        setInterval(() => {
            fetch('/telemetry/{{ scenario.id }}')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('carbon').innerText = d.carbon_intensity.toFixed(1) + ' g';
                    document.getElementById('objects').innerText = d.detected_count + ' ' + d.object_label;
                    document.getElementById('sim-hour').innerText = d.simulated_hour_text;
                    document.getElementById('xai-text').innerText = d.xai_explanation;
                    document.getElementById('energy-pct').innerText = d.energy_saved_pct.toFixed(1) + '%';
                    document.getElementById('energy-bar').style.width = Math.min(100, Math.max(0, d.energy_saved_pct)) + '%';
                    document.getElementById('carbon-saved').innerText = d.cum_carbon_saved_g.toFixed(3) + ' g';
                    document.getElementById('power-latency').innerText = d.latency_ms.toFixed(1) + ' ms • ' + d.power_watts.toFixed(0) + ' W';
                    document.getElementById('mem-ptr').innerText = d.model_memory_address;

                    // Dual Model Visual State Update
                    const cardHeavy = document.getElementById('card-heavy');
                    const cardEco = document.getElementById('card-eco');
                    const badgeHeavy = document.getElementById('badge-heavy');
                    const badgeEco = document.getElementById('badge-eco');
                    const cardChartCarbon = document.getElementById('cardChartCarbon');
                    const cardChartLoad = document.getElementById('cardChartLoad');

                    if (d.active_model.includes("YOLOv8n")) {
                        cardEco.className = "model-card active-eco";
                        badgeEco.innerText = "ACTIVE";
                        badgeEco.style.background = "#10b981";

                        cardHeavy.className = "model-card";
                        badgeHeavy.innerText = "STANDBY";
                        badgeHeavy.style.background = "#334155";

                        // Subtle eco-mode chart accent
                        cardChartCarbon.style.borderColor = "rgba(16, 185, 129, 0.3)";
                        cardChartLoad.style.borderColor = "rgba(16, 185, 129, 0.3)";
                    } else {
                        cardHeavy.className = "model-card active-heavy";
                        badgeHeavy.innerText = "ACTIVE";
                        badgeHeavy.style.background = "#ef4444";

                        cardEco.className = "model-card";
                        badgeEco.innerText = "STANDBY";
                        badgeEco.style.background = "#334155";

                        // Subtle safety-override chart accent
                        cardChartCarbon.style.borderColor = "rgba(239, 68, 68, 0.3)";
                        cardChartLoad.style.borderColor = "rgba(239, 68, 68, 0.3)";
                    }

                    // Append event to UI log ticker
                    if (lastModel && lastModel !== d.active_model) {
                        const logBox = document.getElementById('log-feed');
                        const now = new Date().toLocaleTimeString();
                        const newLog = document.createElement('div');
                        newLog.className = "log-item";
                        newLog.innerText = `[${now}] HOT-SWAP: Activated ${d.active_model} (${d.state})`;
                        logBox.prepend(newLog);
                    }
                    lastModel = d.active_model;

                    // Update Chart A (Carbon)
                    chartCarbon.data.datasets[0].data.push(d.carbon_intensity);
                    chartCarbon.data.datasets[0].data.shift();
                    chartCarbon.update('none');

                    // Update Chart B (Detected Objects)
                    chartLoad.data.datasets[0].data.push(d.detected_count);
                    chartLoad.data.datasets[0].data.shift();
                    chartLoad.update('none');
                })
                .catch(err => console.debug("Telemetry fetch pending...", err));
        }, 300);
    </script>
</body>
</html>
"""


# ==============================================================================
# Flask Routes
# ==============================================================================


@app.route("/")
def index():
    """Multi-scenario landing page."""
    scenarios_summary = []
    for s_id, s_cfg in SCENARIOS.items():
        v_exists = bool(s_cfg.get("video_path") and os.path.exists(s_cfg["video_path"]))
        scenarios_summary.append({**s_cfg, "video_exists": v_exists})

    return render_template_string(LANDING_HTML_TEMPLATE, scenarios=scenarios_summary)


@app.route("/scenario/<scenario_id>")
def scenario_dashboard(scenario_id):
    """Shared scenario dashboard parameterized by scenario config."""
    if scenario_id not in SCENARIOS:
        abort(404, description=f"Scenario '{scenario_id}' not found in scenario catalog.")

    pipeline = scenario_manager.get_pipeline(scenario_id)
    if not pipeline:
        abort(404)

    pipeline.touch()
    scenario_cfg = SCENARIOS[scenario_id]
    v_exists = bool(scenario_cfg.get("video_path") and os.path.exists(scenario_cfg["video_path"]))

    return render_template_string(
        DASHBOARD_HTML_TEMPLATE,
        scenario=scenario_cfg,
        video_exists=v_exists,
        carbon_high=CARBON_HIGH,
        carbon_low=CARBON_LOW,
        demo_mode=DEMO_MODE,
    )


def scenario_stream_generator(pipeline):
    """Stream latest frame from memory buffer of the specific scenario pipeline."""
    while True:
        pipeline.touch()
        frame_bytes = pipeline.get_latest_jpeg()
        if frame_bytes is not None:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.033)


@app.route("/video_feed/<scenario_id>")
def video_feed(scenario_id):
    """Per-scenario MJPEG streaming route."""
    if scenario_id not in SCENARIOS:
        abort(404)

    pipeline = scenario_manager.get_pipeline(scenario_id)
    if not pipeline:
        abort(404)

    pipeline.touch()
    return Response(scenario_stream_generator(pipeline), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/telemetry/<scenario_id>")
def scenario_telemetry(scenario_id):
    """Per-scenario JSON telemetry endpoint."""
    if scenario_id not in SCENARIOS:
        abort(404)

    pipeline = scenario_manager.get_pipeline(scenario_id)
    if not pipeline:
        abort(404)

    pipeline.touch()
    with pipeline.telemetry_lock:
        return jsonify(pipeline.telemetry)


@app.route("/telemetry")
def default_telemetry():
    """Backward compatibility route returning default traffic_demo telemetry."""
    return scenario_telemetry("traffic_demo")


@app.route("/fallback_frame")
def fallback_frame():
    """Return an informative offline standby frame."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (15, 20, 30)
    cv2.putText(img, "ECOSTREAM STANDBY", (180, 160), cv2.FONT_HERSHEY_DUPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(img, "Pipeline ready", (240, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.errorhandler(404)
def page_not_found(e):
    return (
        f"""
        <!DOCTYPE html>
        <html lang="en">
        <head><title>404 | Scenario Not Found</title></head>
        <body style="background:#080c14; color:#f1f5f9; font-family:sans-serif; text-align:center; padding-top:100px;">
            <h1 style="color:#ef4444; font-size:48px;">404</h1>
            <p style="color:#94a3b8; font-size:18px; margin-bottom:24px;">{e.description}</p>
            <a href="/" style="color:#38bdf8; text-decoration:none; font-weight:bold; padding:10px 20px; border:1px solid #38bdf8; border-radius:8px;">← Return to All Scenarios</a>
        </body>
        </html>
        """,
        404,
    )


if __name__ == "__main__":
    host = os.environ.get("ECOSTREAM_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ECOSTREAM_PORT", DEFAULT_PORT))

    logger.info("Starting EcoStream web server on http://%s:%d (DEMO_MODE=%s)", host, port, DEMO_MODE)
    app.run(host=host, port=port, debug=False, threaded=True)