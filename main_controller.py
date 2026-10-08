import os
import cv2
import pickle
import warnings
import logging
import pandas as pd
from datetime import datetime

# Silence all standard warnings & logs
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
os.environ["YOLO_VERBOSE"] = "False"

from ultralytics import YOLO

print("🧠 Initializing EcoStream Intelligent Controller...")

# Load our time-series carbon forecaster model weights
with open('models/grid_predictor.pkl', 'rb') as f:
    predictor = pickle.load(f)

# Load our dual computer vision detection engines
heavy_model = YOLO('models/heavy_yolo.pt')
light_model = YOLO('models/light_yolo.pt')

# Open our downloaded video stream
cap = cv2.VideoCapture('data/sample_video.mp4')
frame_count = 0

print("🚀 Starting video analysis pipeline...")

# Define feature names to match exactly what the model learned during training
feature_names = ['hour', 'day_of_week', 'month']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    now = datetime.now()
    
    # Wrap input data into a DataFrame with matching feature names to stop warnings
    input_data = pd.DataFrame([[now.hour, now.weekday(), now.month]], columns=feature_names)
    
    # Predict current carbon grid intensity
    pred_carbon = predictor.predict(input_data)[0]
    
    # If carbon intensity is high, swap to light eco-mode model
    is_dirty = pred_carbon > 230
    current_model = light_model if is_dirty else heavy_model
    
    # Run inference on the current frame
    results = current_model(frame, verbose=False)
    
    # Print a clean status log every 30 frames
    mode_str = "LIGHT (Eco-Mode)" if is_dirty else "HEAVY (Performance)"
    if frame_count % 30 == 0:
        print(f"Frame {frame_count} | Grid Carbon: {int(pred_carbon)} gCO2/kWh | Running: {mode_str}")

print("✅ Pipeline processing finished successfully!")
cap.release()