import cv2
import torch
import yaml
import os
import time
import boto3
import pymysql
from datetime import datetime
from ultralytics import YOLO

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

with open("aws_config.yaml", "r") as file:
    aws_config = yaml.safe_load(file)

device = config["device"] if torch.cuda.is_available() or config["device"] == "cpu" else "cpu"
model = YOLO(config["model_path"]).to(device)

os.makedirs(os.path.join(config["save_dir"], "images"), exist_ok=True)
os.makedirs(os.path.join(config["save_dir"], "text"), exist_ok=True)

s3 = boto3.client("s3", region_name=aws_config["s3_region"])
textract = boto3.client("textract", region_name=aws_config["s3_region"])

try:
    conn = pymysql.connect(
        host=aws_config["rds_host"],
        user=aws_config["rds_user"],
        password=aws_config["rds_password"],
        database="license_plate_detection"
    )
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ocr_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            plate_text VARCHAR(255),
            confidence FLOAT,
            image_s3_url VARCHAR(512)
        )
    """)
    conn.commit()
except pymysql.MySQLError as e:
    print(f"Database Connection Error: {e}")
    exit()

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

    for i, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box[:4])
        plate_crop = frame[y1:y2, x1:x2]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        img_filename = f"plate_{timestamp}.jpg"
        img_path = os.path.join(config["save_dir"], "images", img_filename)
        
        cv2.imwrite(img_path, plate_crop)
        s3.upload_file(img_path, aws_config["s3_bucket"], img_filename)
        s3_url = f"https://{aws_config['s3_bucket']}.s3.{aws_config['s3_region']}.amazonaws.com/{img_filename}"

        with open(img_path, "rb") as img_file:
            response = textract.detect_document_text(Document={"Bytes": img_file.read()})

        ocr_text = ""
        confidence = 0.0
        total_words = len(response.get("Blocks", []))

        if total_words > 0:
            for block in response["Blocks"]:
                if block["BlockType"] == "LINE":
                    ocr_text += block["Text"] + " "
                    confidence += block["Confidence"]
            confidence /= total_words

        text_filename = f"plate_{timestamp}.txt"
        text_path = os.path.join(config["save_dir"], "text", text_filename)
        with open(text_path, "w") as f:
            f.write(f"Detected Text: {ocr_text.strip()}\nConfidence: {confidence:.2f}\n")

        cursor.execute("INSERT INTO ocr_results (plate_text, confidence, image_s3_url) VALUES (%s, %s, %s)", 
                       (ocr_text.strip(), confidence, s3_url))
        conn.commit()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    timestamp_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp_display, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("License Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
