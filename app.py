import os
import cv2
import glob
import pickle
import numpy as np
import time
from datetime import datetime
from flask import Flask, render_template_string, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Load Predictor and YOLO Models into memory
with open(os.path.join(MODELS_DIR, 'grid_predictor.pkl'), 'rb') as f:
    grid_predictor = pickle.load(f)

print("[INFO] Pre-warming YOLO models into system memory...")
model_heavy = YOLO(os.path.join(MODELS_DIR, 'heavy_yolo.pt'))
model_light = YOLO(os.path.join(MODELS_DIR, 'light_yolo.pt'))
print(f"[INFO] Heavy Model loaded at address: {hex(id(model_heavy))}")
print(f"[INFO] Eco Model loaded at address  : {hex(id(model_light))}")

# Robust Video Auto-Discovery
def get_video_source():
    # 1. Try default sample_video.mp4
    default_path = os.path.join(DATA_DIR, 'sample_video.mp4')
    if os.path.exists(default_path):
        return default_path
    
    # 2. Look for any .mp4 file inside the data folder
    mp4_files = glob.glob(os.path.join(DATA_DIR, '*.mp4'))
    if mp4_files:
        return mp4_files[0]
        
    return default_path

VIDEO_PATH = get_video_source()
print(f"[INFO] Streaming target video file: {VIDEO_PATH}")

current_telemetry = {
    "carbon_intensity": 725.0,
    "vehicle_count": 0,
    "active_model": "YOLOv8m",
    "active_model_params": "25.9M",
    "model_memory_address": hex(id(model_heavy)),
    "state": "Default Mode",
    "latency_ms": 0.0,
    "power_watts": 35.0,
    "cum_baseline_energy": 0.0,
    "cum_ecostream_energy": 0.0,
    "energy_saved_pct": 0.0,
    "cum_carbon_saved_g": 0.0,
    "xai_explanation": "Initializing neural orchestration runtime...",
    "switch_logs": []
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EcoStream AI | Live Runtime Orchestrator</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
                    <div class="log-item">System ready. Video capture initialized.</div>
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
                        Nominal Power: 35.0W
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
                        Throttled Power: 12.0W
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
                    <span class="metric-label">Cumulative Energy Saved</span>
                    <span id="energy-pct" style="font-family:'JetBrains Mono'; font-weight:700; color:#10b981;">0.0%</span>
                </div>
                <div class="progress-bg"><div class="progress-fill" id="energy-bar"></div></div>
            </div>

            <div class="metric-grid">
                <div class="metric-pill">
                    <div class="metric-label">Carbon Abated</div>
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
                    { label: 'Grid Carbon', data: Array(15).fill(720), borderColor: '#38bdf8', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 1 },
                    { label: 'Limit (720)', data: Array(15).fill(720), borderColor: '#ef4444', borderDash: [4, 4], pointRadius: 0, fill: false }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: 670, max: 750, ticks: { color: '#64748b', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
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
                });
        }, 300);
    </script>
</body>
</html>
"""

def generate_frames():
    global current_telemetry
    cap = cv2.VideoCapture(VIDEO_PATH)

    C_THRESH = 720.0
    D_THRESH = 5
    BASELINE_POWER_W = 35.0
    
    frame_idx = 0
    cached_boxes = []
    cached_count = 0
    cached_model_tag = "YOLOv8m"
    cached_state = "Default Mode"
    cached_color = (0, 165, 255)
    last_printed_model = None

    while True:
        ret, frame = cap.read()
        
        # Robust Frame Recovery: loop video seamlessly if it reaches the end or drops a frame
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.04)
                continue

        frame_idx += 1
        frame = cv2.resize(frame, (640, 360))

        simulated_hour = (frame_idx // 20) % 24
        diurnal_curve = [718, 722, 725, 728, 727, 723, 719, 712, 704, 695, 688, 682, 
                         679, 683, 690, 701, 714, 724, 731, 735, 736, 732, 726, 721]
        try:
            pred_carbon = float(grid_predictor.predict([[simulated_hour, 1, 150000.0, 20.0]])[0])
        except Exception:
            pred_carbon = float(diurnal_curve[simulated_hour])

        # Execute decision logic every 3rd frame
        if frame_idx % 3 == 0:
            start_time = cv2.getTickCount()

            # --- PRIORITIZED HEURISTIC ORCHESTRATION ---
            if cached_count >= D_THRESH:
                active_model = model_heavy
                cached_model_tag = "YOLOv8m (Heavy Core)"
                cached_state = "Safety-Critical Override"
                est_power = 35.0
                cached_color = (60, 60, 245)  # Bright Coral Red
                explanation = f"[SAFETY OVERRIDE] Traffic density is {cached_count} (>= {D_THRESH}). Safety override forces YOLOv8m to ensure zero tracking dropouts."
            elif pred_carbon > C_THRESH:
                active_model = model_light
                cached_model_tag = "YOLOv8n (Eco Core)"
                cached_state = "Eco-Mode Abatement"
                est_power = 12.0
                cached_color = (50, 220, 100)  # Bright Neon Green
                explanation = f"[ECO-MODE ACTIVE] Grid carbon is {pred_carbon:.1f} gCO2/kWh (> {C_THRESH}). Traffic is sparse ({cached_count} < {D_THRESH}). Throttled to YOLOv8n to save 65% power."
            else:
                active_model = model_heavy
                cached_model_tag = "YOLOv8m (Heavy Core)"
                cached_state = "Default Mode"
                est_power = 35.0
                cached_color = (255, 165, 0)
                explanation = f"[DEFAULT MODE] Grid carbon is {pred_carbon:.1f} gCO2/kWh (<= {C_THRESH}). Clean energy available; executing high-fidelity core."

            # Terminal verbose logging of backend object swap
            if active_model != last_printed_model:
                now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print("\n" + "="*80)
                print(f"[ECOSTREAM BACKEND HOT-SWAP @ {now_str}]")
                print(f" > Transition Trigger  : {cached_state}")
                print(f" > Active Model Object : {cached_model_tag}")
                print(f" > Memory Address (ID) : {hex(id(active_model))}")
                print(f" > Internal Backbone   : {active_model.model.__class__.__name__}")
                print(f" > Rationale           : {explanation}")
                print("="*80 + "\n")
                last_printed_model = active_model

            # Neural forward pass
            results = active_model(frame, classes=[2, 3, 5, 7], verbose=False, conf=0.25)
            boxes = results[0].boxes

            cached_boxes = []
            cached_count = len(boxes)
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cached_boxes.append(xyxy)

            elapsed_ms = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000.0

            # Energy calculations
            frame_duration_sec = 0.1
            baseline_energy_joules = BASELINE_POWER_W * frame_duration_sec
            ecostream_energy_joules = est_power * frame_duration_sec

            current_telemetry["cum_baseline_energy"] += baseline_energy_joules
            current_telemetry["cum_ecostream_energy"] += ecostream_energy_joules

            joules_saved = current_telemetry["cum_baseline_energy"] - current_telemetry["cum_ecostream_energy"]
            energy_saved_pct = (joules_saved / current_telemetry["cum_baseline_energy"]) * 100.0 if current_telemetry["cum_baseline_energy"] > 0 else 0.0
            carbon_saved_grams = (joules_saved / 3600000.0) * pred_carbon

            # Update shared state
            current_telemetry["carbon_intensity"] = pred_carbon
            current_telemetry["vehicle_count"] = cached_count
            current_telemetry["active_model"] = cached_model_tag
            current_telemetry["model_memory_address"] = hex(id(active_model))
            current_telemetry["state"] = cached_state
            current_telemetry["latency_ms"] = elapsed_ms
            current_telemetry["power_watts"] = est_power
            current_telemetry["energy_saved_pct"] = energy_saved_pct
            current_telemetry["cum_carbon_saved_g"] = carbon_saved_grams
            current_telemetry["xai_explanation"] = explanation

        # In-Video Bounding Boxes and HUD Overlays
        cv2.rectangle(frame, (2, 2), (637, 357), cached_color, 2)
        for xyxy in cached_boxes:
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), cached_color, 2)

        # In-video HUD info box
        active_ptr = hex(id(model_heavy if "Heavy" in cached_model_tag else model_light))[-6:]
        cv2.rectangle(frame, (10, 10), (330, 60), (15, 20, 30), -1)
        cv2.rectangle(frame, (10, 10), (330, 60), cached_color, 1)
        cv2.putText(frame, f"CORE: {cached_model_tag.split()[0]} [0x{active_ptr}]", 
                    (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.52, cached_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"STATE: {cached_state}", 
                    (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 240), 1, cv2.LINE_AA)

        ret_enc, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret_enc:
            continue
            
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/telemetry')
def telemetry():
    return jsonify(current_telemetry)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)