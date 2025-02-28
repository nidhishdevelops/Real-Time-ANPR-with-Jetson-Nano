import cv2
import torch
import yaml
import os
import time
import boto3
from datetime import datetime
from ultralytics import YOLO

# Load config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load AWS credentials from aws_config.yaml
with open("aws_config.yaml", "r") as file:
    aws_config = yaml.safe_load(file)

# Initialize Textract client
textract_client = boto3.client(
    "textract",
    aws_access_key_id=aws_config["aws_access_key_id"],
    aws_secret_access_key=aws_config["aws_secret_access_key"],
    region_name=aws_config["aws_region"]
)

# Load YOLO model
device = config["device"] if torch.cuda.is_available() or config["device"] == "cpu" else "cpu"
model = YOLO(config["model_path"]).to(device)

# Create directories for results
image_dir = os.path.join(config["save_dir"], "images")
text_dir = os.path.join(config["save_dir"], "text")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(text_dir, exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(config["source"])
prev_time = time.time()

def extract_text_from_image(image_path):
    """Extract text from an image using Amazon Textract."""
    with open(image_path, "rb") as file:
        image_bytes = file.read()

    response = textract_client.detect_document_text(
        Document={'Bytes': image_bytes}
    )

    extracted_text = "\n".join([item["Text"] for item in response["Blocks"] if item["BlockType"] == "LINE"])
    return extracted_text

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Run YOLOv8 model
    results = model(frame, imgsz=config["img_size"], conf=config["conf_threshold"], iou=config["iou_threshold"])[0]
    detections = results.boxes.data

    valid_detections = [d for d in detections if d[4] >= config["conf_threshold"]]

    for idx, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box[:4])
        plate_img = frame[y1:y2, x1:x2]  # Crop number plate

        # Save cropped image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = os.path.join(image_dir, f"plate_{timestamp}_{idx}.jpg")
        cv2.imwrite(image_filename, plate_img)

        # Extract text from the cropped image
        extracted_text = extract_text_from_image(image_filename)

        # Save OCR text
        text_filename = os.path.join(text_dir, f"plate_{timestamp}_{idx}.txt")
        with open(text_filename, "w") as text_file:
            text_file.write(extracted_text)

        print(f"Saved Image: {image_filename}")
        print(f"OCR Text: {extracted_text}")

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, extracted_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Display FPS and timestamp
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Number Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
