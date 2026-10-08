from ultralytics import YOLO  
import os  
os.makedirs('models', exist_ok=True)  
print("?? Downloading Heavy Model (YOLOv8x)...")  
heavy_model = YOLO('yolov8x.pt')  
os.rename('yolov8x.pt', 'models/heavy_yolo.pt')  
print("?? Downloading Light Model (YOLOv8n)...")  
light_model = YOLO('yolov8n.pt')  
os.rename('yolov8n.pt', 'models/light_yolo.pt')  
print("? Success! Both models saved cleanly in models/") 
