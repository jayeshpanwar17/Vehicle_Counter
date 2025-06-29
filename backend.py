import cv2
from ultralytics import YOLO
import numpy as np
import datetime
import csv
import os
import signal
import sys
import sqlite3
import time
import threading
from threading import Lock

# === Configuration ===
LOCATION_CONFIG_FILE = "current_camera_location.txt"
DEFAULT_CAMERA_LOCATION_ID = "Basni crossing"
CAMERA_LOCATION_ID = DEFAULT_CAMERA_LOCATION_ID

rtsp_url = "http://192.168.31.90:8080/video"  # Change as needed
model_path = "yolov8n.pt"
count_line_position = 470
offset = 20

# === Global variables for live streaming ===
latest_frame = None
frame_lock = Lock()
camera_active = False
cap = None  # Add global cap variable

# === YOLO Model Load ===
model = YOLO(model_path)

# === CSV and Database Setup ===
count_cars = count_bikes = count_trucks = 0
counted_ids = set()

if not os.path.exists("logs"):
    os.makedirs("logs")

CSV_FILENAME = "logs/vehicle_log_all.csv"
file_exists = os.path.exists(CSV_FILENAME)
csv_file = open(CSV_FILENAME, mode='a', newline='')
csv_writer = csv.writer(csv_file)
if not file_exists or os.path.getsize(CSV_FILENAME) == 0:
    csv_writer.writerow(["Timestamp", "Vehicle Type", "Vehicle ID", "Location ID"])

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
    global camera_active, cap
    camera_active = False
    print("\n🔻 Exiting... Saving data.")
    
    try:
        csv_file.close()
        print("✅ CSV file closed")
    except Exception as e:
        print(f"⚠️ Error closing CSV file: {e}")
    
    # Only try to release cap if it exists
    if cap is not None:
        try:
            cap.release()
            print("✅ Camera released")
        except Exception as e:
            print(f"⚠️ Error releasing camera: {e}")
    
    try:
        db_conn.close()
        print("✅ Database connection closed")
    except Exception as e:
        print(f"⚠️ Error closing database: {e}")
    
    try:
        cv2.destroyAllWindows()
        print("✅ OpenCV windows closed")
    except Exception as e:
        print(f"⚠️ Error closing OpenCV windows: {e}")
    
    print("👋 Cleanup complete. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# === Camera Init Function ===
def init_camera():
    if rtsp_url.startswith("http"):
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

# === Location Reader ===
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
    if not CAMERA_LOCATION_ID:
        CAMERA_LOCATION_ID = DEFAULT_CAMERA_LOCATION_ID

# === Frame Access for Flask ===
def get_latest_frame():
    global latest_frame, frame_lock
    with frame_lock:
        return latest_frame.copy() if latest_frame is not None else None

def is_camera_active():
    global camera_active
    return camera_active

# === Main Detection Loop ===
def run_detection_loop():
    global cap, camera_active, frame_count, last_location_check_time  # Use global cap
    global count_cars, count_bikes, count_trucks
    cap = init_camera()  # Initialize the global cap variable
    read_current_location()

    frame_count = 0
    last_location_check_time = time.time()
    camera_active = True

    while True:
        current_time = time.time()
        if current_time - last_location_check_time >= 5:
            read_current_location()
            last_location_check_time = current_time

        if rtsp_url.startswith("http"):
            ret, frame = cap.read()
        else:
            for _ in range(2):
                cap.grab()
            ret, frame = cap.read()

        if not ret or frame is None or frame.shape[0] == 0:
            print("⚠️ Empty/corrupted frame, trying to reconnect...")
            camera_active = False
            cap.release()
            time.sleep(2)
            cap = init_camera()
            camera_active = True
            continue

        frame_count += 1
        if frame_count % 2 != 0:
            continue

        display_frame = frame.copy()
        results = model.track(display_frame, persist=True, conf=0.5, tracker="bytetrack.yaml")

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
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{label}-{int(box_id)}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    if (count_line_position - offset < center_y < count_line_position + offset and
                            box_id not in counted_ids):
                        counted_ids.add(box_id)
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        csv_writer.writerow([timestamp, label, int(box_id), CAMERA_LOCATION_ID])
                        csv_file.flush()

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

        cv2.line(display_frame, (0, count_line_position), (display_frame.shape[1], count_line_position), (0, 0, 255), 2)
        cv2.putText(display_frame, f"Cars: {count_cars} | Bikes: {count_bikes} | Trucks: {count_trucks}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(display_frame, f"Location: {CAMERA_LOCATION_ID}",
                    (20, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        with frame_lock:
            latest_frame = display_frame.copy()

        cv2.imshow("Vehicle Detection & Counting (Webcam)", display_frame)
        if cv2.waitKey(1) == 27:
            cleanup()

# === Run detection only if backend.py is run directly ===
if __name__ == "__main__":
    run_detection_loop()
