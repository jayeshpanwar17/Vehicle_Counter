import cv2
from ultralytics import YOLO
import numpy as np
import datetime
import csv
import os
import time
import torch
import sqlite3

# === CONFIGURATION ===
VIDEO_PATH = r"clip.mp4"
MODEL_PATH = r"yolov8m.pt"  # Medium model
CSV_FILENAME = "vehicle_log_from_video.csv"
DB_FILENAME = "vehicle_data.db"
LOCATION_CONFIG_FILE = "current_camera_location.txt"

LINE_START = (100, 180)
LINE_END = (700, 50)
OFFSET = 15

CONFIDENCE_THRESHOLD = 0.3
FRAME_SKIP = 3
RESIZE_WIDTH = 960
RESIZE_HEIGHT = 540

# === DATABASE SETUP ===
def init_database():
    """Initialize database with the same schema as app.py"""
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            vehicle_type TEXT NOT NULL,
            vehicle_id INTEGER,
            location_id TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

def get_current_location():
    """Read the current active location from the config file"""
    try:
        if os.path.exists(LOCATION_CONFIG_FILE):
            with open(LOCATION_CONFIG_FILE, 'r') as f:
                location_id = f.read().strip()
                if location_id:
                    return location_id
    except Exception as e:
        print(f"❌ Error reading location config: {e}")
    
    # Default location if file doesn't exist or is empty
    return "Basni Crossing"

def log_vehicle_to_database(vehicle_type, vehicle_id, location_id):
    """Log vehicle detection to SQLite database"""
    try:
        conn = sqlite3.connect(DB_FILENAME)
        cursor = conn.cursor()
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute("""
            INSERT INTO vehicles (timestamp, vehicle_type, vehicle_id, location_id)
            VALUES (?, ?, ?, ?)
        """, (timestamp, vehicle_type, vehicle_id, location_id))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Logged to database: {vehicle_type} (ID: {vehicle_id}) at {location_id}")
        
    except Exception as e:
        print(f"❌ Error logging to database: {e}")

def map_vehicle_class(original_class):
    """Map YOLO classes to dashboard classes"""
    mapping = {
        "car": "car",
        "motorcycle": "motorcycle", 
        "truck": "truck",
        "bus": "bus"
    }
    return mapping.get(original_class, original_class.lower())

# === INIT ===
print("CUDA Available:", torch.cuda.is_available())  # Check if GPU is being used

# Initialize database
init_database()
current_location = get_current_location()
print(f"🎯 Current active location: {current_location}")

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

count_cars = count_bikes = count_trucks = 0
counted_ids = set()
object_memory = {}  # {vehicle_id: (prev_center_x, prev_center_y)}
frame_count = 0

# CSV Logging (keeping for backup)
csv_file = open(CSV_FILENAME, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Vehicle Type", "Vehicle ID", "Location"])

# === CROSS LINE DETECTION FUNCTION ===
def crossed_line(prev, curr, line_start, line_end):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(line_start, prev, curr) != ccw(line_end, prev, curr) and \
           ccw(line_start, line_end, prev) != ccw(line_start, line_end, curr)

# === MAIN LOOP ===
start_time = time.time()
last_location_check = time.time()

print("🚀 Starting vehicle detection...")
print("Press ESC to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        print("🔄 Video ended, restarting...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
        continue

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue  # Skip frames to speed up

    # Check for location changes every 5 seconds
    current_time = time.time()
    if current_time - last_location_check > 5:
        new_location = get_current_location()
        if new_location != current_location:
            current_location = new_location
            print(f"📍 Location changed to: {current_location}")
        last_location_check = current_time

    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # Run YOLO tracking
    results = model.track(frame, persist=True, conf=0.25, tracker="bytetrack.yaml")

    if results[0].boxes.id is not None:
        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for box_id, cls, coord, conf in zip(ids, classes, coords, confs):
            x1, y1, x2, y2 = coord
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            label = model.names[int(cls)]

            if label in ["car", "motorcycle", "truck", "bus"] and conf > 0.25:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}-{int(box_id)}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                prev_center = object_memory.get(box_id, (center_x, center_y))
                object_memory[box_id] = (center_x, center_y)

                if crossed_line(prev_center, (center_x, center_y), LINE_START, LINE_END) and box_id not in counted_ids:
                    counted_ids.add(box_id)
                    
                    # Map the vehicle class for database consistency
                    mapped_class = map_vehicle_class(label)
                    
                    # Log to database
                    log_vehicle_to_database(mapped_class, int(box_id), current_location)
                    
                    # Log to CSV (backup)
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    csv_writer.writerow([timestamp, mapped_class, int(box_id), current_location])
                    csv_file.flush()

                    # Update local counters for display
                    if label == "car":
                        count_cars += 1
                    elif label == "motorcycle":
                        count_bikes += 1
                    elif label == "truck":
                        count_trucks += 1

    # === Draw Line and Info ===
    cv2.line(frame, LINE_START, LINE_END, (0, 0, 255), 2)
    cv2.putText(frame, f"Cars: {count_cars} | Bikes: {count_bikes} | Trucks: {count_trucks}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show current location
    cv2.putText(frame, f"Location: {current_location}", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # FPS counter (debug)
    elapsed_time = time.time() - start_time
    fps = int(frame_count / elapsed_time)
    cv2.putText(frame, f"FPS: {fps}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Vehicle Detection & Counting", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
        break

# === CLEANUP ===
cap.release()
csv_file.close()
cv2.destroyAllWindows()
print("✅ Detection stopped. Data saved to database and CSV file.")
