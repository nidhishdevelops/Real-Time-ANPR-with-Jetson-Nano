import cv2
import yaml
import os
import time
import boto3
import pymysql
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from datetime import datetime

# Load configuration files
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

with open("aws_config.yaml", "r") as file:
    aws_config = yaml.safe_load(file)

# Ensure CUDA is available
if config["device"] == "cuda" and not cuda.Device(0):
    print("CUDA device not found. Switching to CPU.")
    config["device"] = "cpu"

# Initialize TensorRT runtime and load engine
TRT_LOGGER = trt.Logger()

def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_data)

engine = load_engine(config["model_path"])
context = engine.create_execution_context()

# Allocate memory
input_shape = (3, config["img_size"], config["img_size"])  # CHW format
input_size = np.prod(input_shape).item() * np.dtype(np.float32).itemsize
output_size = 1000 * 4  # Adjust based on YOLOv8 output
d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)
bindings = [int(d_input), int(d_output)]

# Create directories
os.makedirs(os.path.join(config["save_dir"], "images"), exist_ok=True)
os.makedirs(os.path.join(config["save_dir"], "text"), exist_ok=True)

# Initialize AWS clients
s3 = boto3.client("s3", region_name=aws_config["s3_region"])
textract = boto3.client("textract", region_name=aws_config["s3_region"])

# Connect to MySQL database
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

# Video capture
cap = cv2.VideoCapture(config["source"])
prev_time = time.time()

def preprocess_image(image):
    image = cv2.resize(image, (config["img_size"], config["img_size"]))
    image = image.transpose((2, 0, 1))  # Convert to CHW
    image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
    return image

def run_inference(frame):
    image_data = preprocess_image(frame)
    cuda.memcpy_htod(d_input, image_data)

    context.execute_v2(bindings)

    output = np.empty(1000, dtype=np.float32)  # Adjust based on model output size
    cuda.memcpy_dtoh(output, d_output)

    return process_output(output)  # Function to extract bounding boxes

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    detections = run_inference(frame)  # Run TensorRT inference

    for i, box in enumerate(detections):
        x1, y1, x2, y2, conf = map(int, box[:5])
        if conf < config["conf_threshold"]:
            continue

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
