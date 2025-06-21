import cv2
from ultralytics import YOLO
import numpy as np
import datetime
import csv
import os
import signal
import sys
import sqlite3
import time # Import time for sleep

# === Configuration ===
# Define the path for the file that holds the current active camera location ID
LOCATION_CONFIG_FILE = "current_camera_location.txt"
# Default location if the config file is not found or empty
DEFAULT_CAMERA_LOCATION_ID = "jodhpur" 

# Initialize CAMERA_LOCATION_ID, will be updated periodically
CAMERA_LOCATION_ID = DEFAULT_CAMERA_LOCATION_ID

webcam_index = "http://192.168.31.90:8080/video"
model_path = "yolov8n.pt"
count_line_position = 270
offset = 20

# Function to read the current camera location from the config file
def read_current_location():
    global CAMERA_LOCATION_ID
    try:
        if os.path.exists(LOCATION_CONFIG_FILE):
            with open(LOCATION_CONFIG_FILE, 'r') as f:
                new_location = f.read().strip()
                if new_location and new_location != CAMERA_LOCATION_ID:
                    print(f"🔄 Backend location updated from '{CAMERA_LOCATION_ID}' to '{new_location}'")
                    CAMERA_LOCATION_ID = new_location
    except Exception as e:
        print(f"⚠️ Error reading location config file: {e}")
    # Ensure it's never empty
    if not CAMERA_LOCATION_ID:
        CAMERA_LOCATION_ID = DEFAULT_CAMERA_LOCATION_ID

# === Initialize YOLOv8 Model ===
model = YOLO(model_path)

# === Initialize Webcam Capture ===
cap = cv2.VideoCapture(webcam_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === Logging Setup ===
count_cars = count_bikes = count_trucks = 0
counted_ids = set()

if not os.path.exists("logs"):
    os.makedirs("logs")

# Single CSV filename for all locations
CSV_FILENAME = "logs/vehicle_log_all.csv"
# Check if CSV file exists to write header only once
file_exists = os.path.exists(CSV_FILENAME)
csv_file = open(CSV_FILENAME, mode='a', newline='')
csv_writer = csv.writer(csv_file)
if not file_exists or os.path.getsize(CSV_FILENAME) == 0:
    csv_writer.writerow(["Timestamp", "Vehicle Type", "Vehicle ID", "Location ID"])

# === Connect to SQLite ===
db_conn = sqlite3.connect("vehicle_data.db", check_same_thread=False)
db_cursor = db_conn.cursor()
db_cursor.execute("""
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    vehicle_type TEXT NOT NULL,
    vehicle_id INTEGER,
    location_id TEXT NOT NULL
)
""")
db_conn.commit()

# === Graceful Shutdown ===
def cleanup(*args):
    print("\n🔻 Exiting... Saving data.")
    csv_file.close()
    cap.release()
    db_conn.close()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# Initial read of location
read_current_location()

# === Main Loop ===
frame_count = 0
last_location_check_time = time.time()
LOCATION_CHECK_INTERVAL = 5 # Check for location update every 5 seconds

while True:
    # Periodically check for location updates from the file
    current_time = time.time()
    if current_time - last_location_check_time >= LOCATION_CHECK_INTERVAL:
        read_current_location()
        last_location_check_time = current_time

    ret, frame = cap.read()
    if not ret:
        print(f"❌ Failed to grab frame from webcam_index {webcam_index}. Retrying...")
        continue

    frame_count += 1
    if frame_count % 2 != 0: # Process every other frame
        continue

    results = model.track(frame, persist=True, conf=0.5, tracker="bytetrack.yaml")

    if results[0].boxes.id is not None:
        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for box_id, cls, coord, conf in zip(ids, classes, coords, confs):
            x1, y1, x2, y2 = coord
            center_y = int((y1 + y2) / 2)
            label = model.names[int(cls)]

            if label in ["car", "motorcycle", "truck"] and conf > 0.5:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}-{int(box_id)}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if (count_line_position - offset < center_y < count_line_position + offset and
                        box_id not in counted_ids):
                    counted_ids.add(box_id)
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Log with the currently active CAMERA_LOCATION_ID
                    csv_writer.writerow([timestamp, label, int(box_id), CAMERA_LOCATION_ID])
                    csv_file.flush() # Ensure data is written to disk immediately

                    db_cursor.execute(
                        "INSERT INTO vehicles (timestamp, vehicle_type, vehicle_id, location_id) VALUES (?, ?, ?, ?)",
                        (timestamp, label, int(box_id), CAMERA_LOCATION_ID)
                    )
                    db_conn.commit()

                    print(f"✔ Counted {label}-{int(box_id)} at {timestamp} for location {CAMERA_LOCATION_ID}")

                    if label == "car":
                        count_cars += 1
                    elif label == "motorcycle":
                        count_bikes += 1
                    elif label == "truck":
                        count_trucks += 1

    cv2.line(frame, (0, count_line_position), (frame.shape[1], count_line_position), (0, 0, 255), 2)

    cv2.putText(frame, f"Cars: {count_cars} | Bikes: {count_bikes} | Trucks: {count_trucks}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Vehicle Detection & Counting (Webcam)", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        cleanup()