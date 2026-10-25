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
from flask import Flask, render_template_string, Response, jsonify

from ultralytics import YOLO
from config import (
    MODELS_DIR,
    DATA_DIR,
    PREDICTOR_PATH,
    HEAVY_MODEL_PATH,
    LIGHT_MODEL_PATH,
    DEFAULT_VIDEO_PATH,
    GRID_DATA_PATH,
    CARBON_HIGH,
    CARBON_LOW,
    DENSITY_THRESHOLD,
    HEAVY_POWER_W,
    ECO_POWER_W,
    POWER_SAVED_PCT,
    VEHICLE_CLASSES,
    ECO_SAFETY_PROBE_INTERVAL,
    DEFAULT_HOST,
    DEFAULT_PORT,
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

# Load Predictor and YOLO Models into memory
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

logger.info("Pre-warming YOLO models into system memory...")
model_heavy = None
model_light = None

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


def get_video_source():
    """Discover available video stream source or fallback."""
    if os.path.exists(DEFAULT_VIDEO_PATH):
        return DEFAULT_VIDEO_PATH

    mp4_files = glob.glob(os.path.join(DATA_DIR, "*.mp4"))
    if mp4_files:
        return mp4_files[0]

    return None


VIDEO_PATH = get_video_source()
logger.info("Streaming target video source: %s", VIDEO_PATH if VIDEO_PATH else "NONE (Offline / Fallback)")

# Shared Telemetry State with Mutex
telemetry_lock = threading.Lock()
current_telemetry = {
    "carbon_intensity": float(CARBON_HIGH),
    "vehicle_count": 0,
    "active_model": "YOLOv8m",
    "active_model_params": "25.9M",
    "model_memory_address": hex(id(model_heavy)) if model_heavy else "0x000000000000",
    "state": "Default Mode",
    "latency_ms": 0.0,
    "power_watts": HEAVY_POWER_W,
    "cum_baseline_energy": 0.0,
    "cum_ecostream_energy": 0.0,
    "energy_saved_pct": 0.0,
    "cum_carbon_saved_g": 0.0,
    "xai_explanation": "Initializing neural orchestration runtime...",
    "switch_logs": [],
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EcoStream AI | Live Runtime Orchestrator</title>
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
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Plus Jakarta Sans', sans-serif; 
            background: radial-gradient(circle at top right, #111e38, var(--bg-dark) 85%); 
            color: #f1f5f9; 
            padding: 20px; 
            min-height: 100vh;
        }
        .header { display: flex; justify-content: space-between; align-items: center; max-width: 1500px; margin: 0 auto 16px auto; }
        .brand-logo { font-size: 24px; font-weight: 800; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .live-chip { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; padding: 4px 12px; border-radius: 99px; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
        
        .container { max-width: 1500px; margin: auto; display: grid; grid-template-columns: 1.85fr 1.15fr; gap: 20px; }
        .glass-card { background: var(--card-bg); backdrop-filter: blur(14px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 18px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        .video-wrapper { position: relative; border-radius: 10px; overflow: hidden; border: 2px solid rgba(255,255,255,0.1); background: #000; min-height: 360px; display: flex; align-items: center; justify-content: center; }
        .video-wrapper img { width: 100%; height: auto; display: block; }
        
        /* Dual Architecture Switcher Cards */
        .model-switcher { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; }
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
        .log-box { background: #030712; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 10px; height: 110px; overflow-y: hidden; font-family: 'JetBrains Mono', monospace; font-size: 11px; display: flex; flex-direction: column-reverse; gap: 4px; }
        .log-item { color: #94a3b8; border-left: 2px solid #38bdf8; padding-left: 6px; }

        .chart-box { height: 150px; background: rgba(10, 15, 29, 0.7); border-radius: 10px; padding: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <div class="brand-logo">⚡ EcoStream: Explainable Runtime Orchestrator</div>
        <div class="live-chip">BACKEND ORCHESTRATION PIPELINE ACTIVE</div>
    </div>

    <div class="container">
        <!-- Left Column: Video Stream & XAI Decisions -->
        <div class="glass-card" style="display: flex; flex-direction: column; gap: 14px;">
            <div class="video-wrapper">
                <img src="{{ url_for('video_feed') }}" alt="Awaiting Stream..." onerror="this.src='/fallback_frame';" />
            </div>

            <!-- Explainable AI Explanation -->
            <div style="background: rgba(15, 23, 42, 0.85); border-left: 4px solid var(--accent-cyan); padding: 14px; border-radius: 8px;">
                <div style="font-size:11px; font-weight:800; color:var(--accent-cyan); text-transform:uppercase;">🔍 XAI Runtime Decision Rationale</div>
                <div id="xai-text" style="font-size:13px; color:#cbd5e1; margin-top:5px; line-height:1.4;">Evaluating frame context...</div>
            </div>

            <!-- Transition Log Feed -->
            <div>
                <div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; margin-bottom:6px;">Runtime Transition Log</div>
                <div class="log-box" id="log-feed">
                    <div class="log-item">System ready. Pipeline initialized.</div>
                </div>
            </div>
        </div>

        <!-- Right Column: Dual Architecture Spec Cards, Metrics & Chart -->
        <div class="glass-card">
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
                        Nominal Power: {{ heavy_power_w }}W
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
                        Throttled Power: {{ eco_power_w }}W ({{ power_saved_pct|round|int }}% Saved)
                    </div>
                </div>
            </div>

            <!-- Pointer Proof Metric Card -->
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
                    <div class="metric-label">Detected Vehicles</div>
                    <div class="metric-val" id="vehicles" style="color:#38bdf8;">--</div>
                </div>
            </div>

            <div class="progress-box">
                <div style="display:flex; justify-content:space-between;">
                    <span class="metric-label">Estimated Energy Saved</span>
                    <span id="energy-pct" style="font-family:'JetBrains Mono'; font-weight:700; color:#10b981;">0.0%</span>
                </div>
                <div class="progress-bg"><div class="progress-fill" id="energy-bar"></div></div>
            </div>

            <div class="metric-grid">
                <div class="metric-pill">
                    <div class="metric-label">Estimated Carbon Abated</div>
                    <div class="metric-val" id="carbon-saved" style="color:#10b981;">0.000 g</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-label">Execution Latency & Power</div>
                    <div class="metric-val" id="power-latency" style="font-size:15px; color:#cbd5e1;">--</div>
                </div>
            </div>

            <!-- Live Chart -->
            <div class="chart-box">
                <canvas id="carbonChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('carbonChart').getContext('2d');
        const carbonChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(15).fill(''),
                datasets: [
                    { 
                        label: 'Grid Carbon', 
                        data: Array(15).fill({{ carbon_high }}), 
                        borderColor: '#38bdf8', 
                        borderWidth: 2, 
                        fill: false, 
                        tension: 0.3, 
                        pointRadius: 1 
                    },
                    { 
                        label: 'Limit ({{ carbon_high|round|int }})', 
                        data: Array(15).fill({{ carbon_high }}), 
                        borderColor: '#ef4444', 
                        borderDash: [4, 4], 
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
                        min: {{ y_min }}, 
                        max: {{ y_max }}, 
                        ticks: { color: '#64748b', font: { size: 9 } }, 
                        grid: { color: 'rgba(255,255,255,0.05)' } 
                    },
                    x: { display: false }
                },
                plugins: { legend: { display: false } }
            }
        });

        let lastModel = "";

        setInterval(() => {
            fetch('/telemetry')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('carbon').innerText = d.carbon_intensity.toFixed(1) + ' g';
                    document.getElementById('vehicles').innerText = d.vehicle_count + ' Cars';
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

                    if (d.active_model.includes("YOLOv8n")) {
                        cardEco.className = "model-card active-eco";
                        badgeEco.innerText = "ACTIVE";
                        badgeEco.style.background = "#10b981";

                        cardHeavy.className = "model-card";
                        badgeHeavy.innerText = "STANDBY";
                        badgeHeavy.style.background = "#334155";
                    } else {
                        cardHeavy.className = "model-card active-heavy";
                        badgeHeavy.innerText = "ACTIVE";
                        badgeHeavy.style.background = "#ef4444";

                        cardEco.className = "model-card";
                        badgeEco.innerText = "STANDBY";
                        badgeEco.style.background = "#334155";
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

                    // Update live chart
                    carbonChart.data.datasets[0].data.push(d.carbon_intensity);
                    carbonChart.data.datasets[0].data.shift();
                    carbonChart.update('none');
                })
                .catch(err => console.debug("Telemetry fetch pending...", err));
        }, 300);
    </script>
</body>
</html>
"""


class VideoStreamWorker(threading.Thread):
    """
    Decoupled single background worker:
    Captures frames, executes carbon prediction & YOLO inference,
    updates shared telemetry, and produces a single latest JPEG buffer.
    All web clients read from this buffer with ZERO multi-viewer duplicate inference lag.
    """

    def __init__(self, video_source):
        super().__init__(daemon=True)
        self.video_source = video_source
        self.latest_jpeg_bytes = None
        self.lock = threading.Lock()
        self.running = True

    def build_synthetic_frame(self, frame_idx, text="STREAM STANDBY"):
        """Generate animated synthetic video frame when no video file is present."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Background gradient
        frame[:] = (20, 24, 34)

        # Draw road lanes
        cv2.rectangle(frame, (0, 100), (640, 280), (35, 40, 52), -1)
        for x in range((frame_idx * 6) % 80 - 80, 640, 80):
            cv2.line(frame, (x, 190), (x + 40, 190), (120, 130, 150), 2)

        # Draw simulated moving vehicles (cars)
        num_sim_cars = 3 if (frame_idx // 60) % 2 == 0 else 6
        for i in range(num_sim_cars):
            x = (frame_idx * (3 + i * 2) + i * 140) % 640
            y = 130 if i % 2 == 0 else 220
            color = (56, 189, 248) if i % 2 == 0 else (16, 185, 129)
            cv2.rectangle(frame, (x, y), (x + 55, y + 28), color, -1)
            cv2.rectangle(frame, (x, y), (x + 55, y + 28), (255, 255, 255), 1)

        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (56, 189, 248), 1, cv2.LINE_AA)
        return frame

    def run(self):
        global current_telemetry

        cap = None
        if self.video_source and os.path.exists(self.video_source):
            cap = cv2.VideoCapture(self.video_source)

        frame_idx = 0
        cached_boxes = []
        cached_count = 0
        cached_model_tag = "YOLOv8m (Heavy Core)"
        cached_state = "Default Mode"
        cached_color = (255, 165, 0)
        current_mode = "heavy_default"
        current_sim_hour = -1
        current_carbon = float(CARBON_HIGH)

        while self.running:
            # 1. Acquire Frame
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.03)
                        continue
            else:
                # Synthetic simulated camera feed
                frame = self.build_synthetic_frame(frame_idx, "ECOSTREAM SYNTHETIC SIMULATION STREAM")

            frame_idx += 1
            frame = cv2.resize(frame, (640, 360))

            # 2. Simulated Clock: hour advances every 20 frames (simulates 24-hour day in ~16s)
            simulated_hour = (frame_idx // 20) % 24

            # Predict carbon intensity ONCE per simulated hour
            if simulated_hour != current_sim_hour:
                current_sim_hour = simulated_hour
                current_carbon = predict_carbon(
                    grid_predictor,
                    hour=simulated_hour,
                    fallback=HOURLY_FALLBACK.get(simulated_hour, 250.0),
                )

            # 3. Decision & Inference every 3rd frame
            if frame_idx % 3 == 0 or frame_idx == 1:
                start_ticks = cv2.getTickCount()

                # Eco-Mode Safety Gap Verification:
                # Periodic heavy model probe every N frames while in eco mode to eliminate nano undercounting
                if current_mode == "eco" and (frame_idx % ECO_SAFETY_PROBE_INTERVAL == 0) and model_heavy:
                    probe_res = model_heavy(frame, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
                    probe_cnt = len(probe_res[0].boxes)
                    if probe_cnt >= DENSITY_THRESHOLD:
                        # Safety override immediately triggered by heavy validation probe
                        cached_count = probe_cnt

                # Determine target operating mode via central orchestrator
                new_mode = choose_mode(current_carbon, cached_count, current_mode)

                # Log transition on mode change
                if new_mode != current_mode:
                    meta = get_mode_metadata(new_mode, current_carbon, cached_count)
                    log_decision(
                        simulated_hour=simulated_hour,
                        carbon=current_carbon,
                        vehicle_count=cached_count,
                        mode=new_mode,
                        reason=meta["state_label"],
                    )
                    logger.info(
                        "[ORCHESTRATOR TRANSITION] Mode: %s | Carbon: %.1f | Vehicles: %d | %s",
                        new_mode,
                        current_carbon,
                        cached_count,
                        meta["state_label"],
                    )
                    current_mode = new_mode

                meta = get_mode_metadata(current_mode, current_carbon, cached_count)
                cached_model_tag = meta["model_tag"]
                cached_state = meta["state_label"]
                cached_color = meta["color_bgr"]
                est_power = meta["power_w"]
                explanation = meta["explanation"]

                # Select active YOLO model
                active_model = model_light if current_mode == "eco" and model_light else model_heavy

                if active_model is not None:
                    results = active_model(frame, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
                    boxes = results[0].boxes
                    cached_boxes = []
                    cached_count = len(boxes)
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        cached_boxes.append(xyxy)
                else:
                    cached_boxes = []

                elapsed_ticks = cv2.getTickCount() - start_ticks
                elapsed_ms = (elapsed_ticks / cv2.getTickFrequency()) * 1000.0

                # Measured energy calculations using measured execution latency
                elapsed_sec = elapsed_ms / 1000.0
                baseline_energy_joules = HEAVY_POWER_W * elapsed_sec
                ecostream_energy_joules = est_power * elapsed_sec

                with telemetry_lock:
                    current_telemetry["cum_baseline_energy"] += baseline_energy_joules
                    current_telemetry["cum_ecostream_energy"] += ecostream_energy_joules

                    delta_joules = baseline_energy_joules - ecostream_energy_joules
                    if delta_joules > 0:
                        delta_carbon_grams = (delta_joules / 3600000.0) * current_carbon
                        current_telemetry["cum_carbon_saved_g"] += delta_carbon_grams

                    joules_saved = (
                        current_telemetry["cum_baseline_energy"] - current_telemetry["cum_ecostream_energy"]
                    )
                    energy_saved_pct = (
                        (joules_saved / current_telemetry["cum_baseline_energy"]) * 100.0
                        if current_telemetry["cum_baseline_energy"] > 0
                        else 0.0
                    )

                    current_telemetry["carbon_intensity"] = current_carbon
                    current_telemetry["vehicle_count"] = cached_count
                    current_telemetry["active_model"] = cached_model_tag
                    current_telemetry["model_memory_address"] = (
                        hex(id(active_model)) if active_model else "0x000000000000"
                    )
                    current_telemetry["state"] = cached_state
                    current_telemetry["latency_ms"] = elapsed_ms
                    current_telemetry["power_watts"] = est_power
                    current_telemetry["energy_saved_pct"] = energy_saved_pct
                    current_telemetry["xai_explanation"] = explanation

            # 4. In-Video Overlays & Bounding Boxes
            cv2.rectangle(frame, (2, 2), (637, 357), cached_color, 2)
            for xyxy in cached_boxes:
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), cached_color, 2)

            # In-video HUD box
            active_ptr = hex(id(model_heavy if "Heavy" in cached_model_tag else model_light))[-6:] if (model_heavy or model_light) else "000000"
            cv2.rectangle(frame, (10, 10), (330, 60), (15, 20, 30), -1)
            cv2.rectangle(frame, (10, 10), (330, 60), cached_color, 1)
            cv2.putText(
                frame,
                f"CORE: {cached_model_tag.split()[0]} [0x{active_ptr}]",
                (20, 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.52,
                cached_color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"STATE: {cached_state}",
                (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (200, 220, 240),
                1,
                cv2.LINE_AA,
            )

            # 5. JPEG Frame Encoding
            ret_enc, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret_enc:
                with self.lock:
                    self.latest_jpeg_bytes = buffer.tobytes()

            # Maintain smooth ~30 FPS frame pacing
            time.sleep(0.018)

        if cap:
            cap.release()


# Single background worker instance
stream_worker = VideoStreamWorker(VIDEO_PATH)


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        carbon_high=CARBON_HIGH,
        carbon_low=CARBON_LOW,
        y_min=100,
        y_max=400,
        heavy_power_w=HEAVY_POWER_W,
        eco_power_w=ECO_POWER_W,
        power_saved_pct=POWER_SAVED_PCT,
    )


def stream_generator():
    """Stream latest frame from memory buffer to client."""
    while True:
        with stream_worker.lock:
            frame_bytes = stream_worker.latest_jpeg_bytes

        if frame_bytes is not None:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.033)  # Deliver at ~30 FPS


@app.route("/video_feed")
def video_feed():
    return Response(stream_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/fallback_frame")
def fallback_frame():
    """Return an informative offline frame if stream drops or video is missing."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (15, 20, 30)
    cv2.putText(
        img,
        "ECOSTREAM STREAM STANDBY",
        (130, 150),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "No video found in data/ (sample_video.mp4)",
        (120, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (148, 163, 184),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Place a video in data/sample_video.mp4 to stream",
        (115, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (56, 189, 248),
        1,
        cv2.LINE_AA,
    )
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/telemetry")
def telemetry():
    with telemetry_lock:
        return jsonify(current_telemetry)


if __name__ == "__main__":
    host = os.environ.get("ECOSTREAM_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ECOSTREAM_PORT", DEFAULT_PORT))

    logger.info("Starting EcoStream background streaming worker...")
    stream_worker.start()

    logger.info("Starting EcoStream web server on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=False, threaded=True)