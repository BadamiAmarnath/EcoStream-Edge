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
    DEMO_MODE,
    DEMO_CARBON_3AM,
    DEMO_CARBON_9AM,
    IMG_SIZE,
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

logger.info("Pre-warming YOLO models into system memory with IMG_SIZE=%d...", IMG_SIZE)
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
    "carbon_intensity": float(DEMO_CARBON_3AM if DEMO_MODE else CARBON_HIGH),
    "simulated_hour": 3 if DEMO_MODE else 0,
    "simulated_hour_text": "03:00 AM (Low Traffic)" if DEMO_MODE else "00:00 (Night)",
    "vehicle_count": 0,
    "active_model": "YOLOv8n (Eco Core)" if DEMO_MODE else "YOLOv8m (Heavy Core)",
    "active_model_params": "3.2M" if DEMO_MODE else "25.9M",
    "model_memory_address": hex(id(model_light)) if DEMO_MODE and model_light else (hex(id(model_heavy)) if model_heavy else "0x000000000000"),
    "state": "Eco-Mode Abatement" if DEMO_MODE else "Default Mode",
    "latency_ms": 0.0,
    "power_watts": ECO_POWER_W if DEMO_MODE else HEAVY_POWER_W,
    "cum_baseline_energy": 0.0,
    "cum_ecostream_energy": 0.0,
    "energy_saved_pct": 0.0,
    "cum_carbon_saved_g": 0.0,
    "xai_explanation": "Initializing neural orchestration runtime...",
    "demo_mode": DEMO_MODE,
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
        .header { display: flex; justify-content: space-between; align-items: center; max-width: 1500px; margin: 0 auto 16px auto; }
        .brand-logo { font-size: 24px; font-weight: 800; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .chips-group { display: flex; gap: 10px; align-items: center; }
        .live-chip { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; padding: 4px 12px; border-radius: 99px; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
        .demo-chip { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); color: #fbbf24; padding: 4px 12px; border-radius: 99px; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
        
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
        <div class="chips-group">
            {% if demo_mode %}
            <div class="demo-chip">⚡ DEMO MODE — SIMULATED CARBON</div>
            {% endif %}
            <div class="live-chip">BACKEND ORCHESTRATION PIPELINE ACTIVE</div>
        </div>
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
                    <div class="log-item">System ready. Playback pipeline decoupled and active.</div>
                </div>
            </div>
        </div>

        <!-- Right Column: Dual Architecture Spec Cards, Metrics & Chart -->
        <div class="glass-card">
            <!-- Simulated Timeline / Hour Indicator -->
            <div class="metric-pill" style="margin-bottom:12px; border-left: 3px solid var(--accent-cyan);">
                <div class="metric-label">Simulated Timeline & Context</div>
                <div class="metric-val" id="sim-hour" style="font-size:15px; color:#38bdf8;">03:00 AM (Low Traffic)</div>
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
                    <span class="metric-label">Estimated Energy Saved (Per Loop)</span>
                    <span id="energy-pct" style="font-family:'JetBrains Mono'; font-weight:700; color:#10b981;">0.0%</span>
                </div>
                <div class="progress-bg"><div class="progress-fill" id="energy-bar"></div></div>
            </div>

            <div class="metric-grid">
                <div class="metric-pill">
                    <div class="metric-label">Estimated Carbon Abated (Per Loop)</div>
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


class AsyncInferenceWorker(threading.Thread):
    """
    Dedicated worker thread that processes YOLO inference asynchronously.
    Decoupled from video playback so inference latency NEVER drops video framerate.
    """

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

        # Output states shared with playback thread
        self.latest_boxes = []
        self.latest_count = 0
        self.latest_model_tag = "YOLOv8n (Eco Core)" if DEMO_MODE else "YOLOv8m (Heavy Core)"
        self.latest_state = "Eco-Mode Abatement" if DEMO_MODE else "Default Mode"
        self.latest_color = (50, 220, 100) if DEMO_MODE else (255, 165, 0)
        self.current_mode = "eco" if DEMO_MODE else "heavy_default"
        self.latest_latency_ms = 0.0
        self.latest_power_w = ECO_POWER_W if DEMO_MODE else HEAVY_POWER_W
        self.active_ptr = "000000"
        self.probe_counter = 0

    def submit_frame(self, frame, carbon, sim_hour):
        """Submit a frame for detection if worker is not busy."""
        with self.lock:
            if self.is_busy:
                return False
            self.pending_frame = frame.copy()
            self.pending_meta = {"carbon": carbon, "hour": sim_hour}
            self.is_busy = True
            self.new_frame_event.set()
            return True

    def run(self):
        global current_telemetry

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

            current_carbon = meta.get("carbon", float(CARBON_HIGH))
            sim_hour = meta.get("hour", 0)

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

            # Decide operating mode via central orchestrator
            new_mode = choose_mode(current_carbon, self.latest_count, self.current_mode)

            if new_mode != self.current_mode:
                mode_meta = get_mode_metadata(new_mode, current_carbon, self.latest_count)
                log_decision(
                    simulated_hour=sim_hour,
                    carbon=current_carbon,
                    vehicle_count=self.latest_count,
                    mode=new_mode,
                    reason=mode_meta["state_label"],
                )
                logger.info(
                    "[ORCHESTRATOR TRANSITION] Mode: %s | Carbon: %.1f | Vehicles: %d | %s",
                    new_mode,
                    current_carbon,
                    self.latest_count,
                    mode_meta["state_label"],
                )
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
            elapsed_sec = elapsed_ms / 1000.0

            # Real measured energy calculations using measured execution latency
            est_power = mode_meta["power_w"]
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
                current_telemetry["simulated_hour"] = sim_hour
                current_telemetry["simulated_hour_text"] = (
                    f"{sim_hour:02d}:00 {'AM' if sim_hour < 12 else 'PM'} "
                    f"({'Low Traffic' if sim_hour == 3 else 'Rush Hour'})"
                )
                current_telemetry["vehicle_count"] = len(boxes)
                current_telemetry["active_model"] = mode_meta["model_tag"]
                current_telemetry["active_model_params"] = "25.9M" if mode_meta["model_key"] == "heavy" else "3.2M"
                current_telemetry["model_memory_address"] = (
                    hex(id(active_model)) if active_model else "0x000000000000"
                )
                current_telemetry["state"] = mode_meta["state_label"]
                current_telemetry["latency_ms"] = elapsed_ms
                current_telemetry["power_watts"] = est_power
                current_telemetry["energy_saved_pct"] = energy_saved_pct
                current_telemetry["xai_explanation"] = mode_meta["explanation"]

            # Update shared inference results for playback loop
            with self.lock:
                self.latest_boxes = boxes
                self.latest_count = len(boxes)
                self.latest_model_tag = mode_meta["model_tag"]
                self.latest_state = mode_meta["state_label"]
                self.latest_color = mode_meta["color_bgr"]
                self.latest_latency_ms = elapsed_ms
                self.latest_power_w = est_power
                self.active_ptr = (
                    hex(id(active_model))[-6:] if active_model else "000000"
                )
                self.is_busy = False


class RealtimePlaybackWorker(threading.Thread):
    """
    Playback Thread:
    Reads frames strictly by wall-clock time against the video's FPS.
    Skips frames using cap.grab() when falling behind wall-clock time.
    Overlays the latest available detections and encodes JPEG.
    Never blocks on inference!
    """

    def __init__(self, video_source, inference_worker):
        super().__init__(daemon=True)
        self.video_source = video_source
        self.inference_worker = inference_worker
        self.latest_jpeg_bytes = None
        self.lock = threading.Lock()
        self.running = True

    def build_synthetic_frame(self, frame_idx, text="STREAM STANDBY"):
        """Generate animated synthetic video frame when no video file is present."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:] = (20, 24, 34)

        # Draw road lanes
        cv2.rectangle(frame, (0, 100), (640, 280), (35, 40, 52), -1)
        for x in range((frame_idx * 6) % 80 - 80, 640, 80):
            cv2.line(frame, (x, 190), (x + 40, 190), (120, 130, 150), 2)

        # Draw simulated moving vehicles
        num_sim_cars = 2 if (frame_idx // 120) % 2 == 0 else 8
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
        video_fps = 24.0
        total_frames = 480.0

        if self.video_source and os.path.exists(self.video_source):
            cap = cv2.VideoCapture(self.video_source)
            if cap.isOpened():
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 480.0

        frame_duration = 1.0 / video_fps
        frame_counter = 0
        start_wall_clock = time.perf_counter()

        while self.running:
            # 1. Wall-clock synchronization & frame skipping with cap.grab()
            now = time.perf_counter()
            expected_elapsed = frame_counter * frame_duration
            actual_elapsed = now - start_wall_clock

            if actual_elapsed - expected_elapsed > frame_duration and cap and cap.isOpened():
                # Skip frames to catch up to real-time wall-clock
                frames_to_skip = int((actual_elapsed - expected_elapsed) / frame_duration)
                for _ in range(min(frames_to_skip, 5)):
                    if not cap.grab():
                        break
                    frame_counter += 1

            # 2. Frame Retrieval
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Video reached end -> loop back seamlessly
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    start_wall_clock = time.perf_counter()
                    frame_counter = 0

                    # Reset cumulative counters per loop so stats reflect the active 20s demo
                    with telemetry_lock:
                        current_telemetry["cum_baseline_energy"] = 0.0
                        current_telemetry["cum_ecostream_energy"] = 0.0
                        current_telemetry["cum_carbon_saved_g"] = 0.0
                        current_telemetry["energy_saved_pct"] = 0.0

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(frame_duration)
                        continue
            else:
                frame = self.build_synthetic_frame(frame_counter, "ECOSTREAM SYNTHETIC SIMULATION STREAM")

            frame_counter += 1
            frame_resized = cv2.resize(frame, (640, 360))

            # 3. Calculate video progress and simulated hour
            if cap and cap.isOpened():
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                progress = pos / total_frames if total_frames > 0 else 0.0
            else:
                progress = ((frame_counter % 480) / 480.0)

            if DEMO_MODE:
                simulated_hour = 3 if progress < 0.5 else 9
                current_carbon = DEMO_CARBON_3AM if progress < 0.5 else DEMO_CARBON_9AM
            else:
                simulated_hour = (frame_counter // 20) % 24
                current_carbon = predict_carbon(
                    grid_predictor,
                    hour=simulated_hour,
                    fallback=HOURLY_FALLBACK.get(simulated_hour, 250.0),
                )

            # 4. Dispatch to background inference worker (non-blocking)
            self.inference_worker.submit_frame(frame_resized, current_carbon, simulated_hour)

            # 5. Retrieve most recent inference results
            with self.inference_worker.lock:
                boxes = self.inference_worker.latest_boxes
                color = self.inference_worker.latest_color
                tag = self.inference_worker.latest_model_tag
                state = self.inference_worker.latest_state
                ptr = self.inference_worker.active_ptr

            # 6. Draw annotations on displayed frame
            cv2.rectangle(frame_resized, (2, 2), (637, 357), color, 2)
            for xyxy in boxes:
                cv2.rectangle(frame_resized, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)

            # In-video HUD with simulated hour and core
            cv2.rectangle(frame_resized, (10, 10), (350, 60), (15, 20, 30), -1)
            cv2.rectangle(frame_resized, (10, 10), (350, 60), color, 1)
            cv2.putText(
                frame_resized,
                f"CORE: {tag.split()[0]} [0x{ptr}] | {simulated_hour:02d}:00",
                (20, 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.48,
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

            # 8. Sleep for remainder of frame interval to match video FPS
            target_next = start_wall_clock + (frame_counter * frame_duration)
            remaining_sleep = target_next - time.perf_counter()
            if remaining_sleep > 0.001:
                time.sleep(remaining_sleep)

        if cap:
            cap.release()


# Global Worker Instances
inference_worker = AsyncInferenceWorker(model_heavy, model_light)
playback_worker = RealtimePlaybackWorker(VIDEO_PATH, inference_worker)


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
        demo_mode=DEMO_MODE,
    )


def stream_generator():
    """Stream latest frame from memory buffer to client."""
    while True:
        with playback_worker.lock:
            frame_bytes = playback_worker.latest_jpeg_bytes

        if frame_bytes is not None:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.033)


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

    logger.info("Starting EcoStream decoupled workers (Inference + Realtime Playback)...")
    inference_worker.start()
    playback_worker.start()

    logger.info("Starting EcoStream web server on http://%s:%d (DEMO_MODE=%s)", host, port, DEMO_MODE)
    app.run(host=host, port=port, debug=False, threaded=True)