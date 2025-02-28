import cv2
import torch
import yaml
import os
import time
from datetime import datetime
from ultralytics import YOLO

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

device = config["device"] if torch.cuda.is_available() or config["device"] == "cpu" else "cpu"
model = YOLO(config["model_path"]).to(device)

os.makedirs(config["save_dir"], exist_ok=True)

cap = cv2.VideoCapture(config["source"])
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    results = model(frame, imgsz=config["img_size"], conf=config["conf_threshold"], iou=config["iou_threshold"])[0]
    detections = results.boxes.data

    valid_detections = [d for d in detections if d[4] >= config["conf_threshold"]]

    if valid_detections:
        filename = os.path.join(config["save_dir"], f"plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Number Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
