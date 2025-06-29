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

# Updated RTSP URL for CP Plus camera
rtsp_url = "rtsp://admin:suncity%4013@192.168.1.203:554/cam/realmonitor?channel=1&subtype=0"
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

# === Enhanced Camera Init Function for RTSP ===
def init_camera():
    print(f"🎥 Initializing camera with URL: {rtsp_url}")
    
    if rtsp_url.startswith("rtsp://"):
        # For RTSP streams (CP Plus camera)
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
        cap.set(cv2.CAP_PROP_FPS, 25)        # Set FPS to match camera
        
        # Timeout settings for RTSP
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)   # 10 seconds connection timeout
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)    # 5 seconds read timeout
        
        # Optional: Set codec for better performance
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        
        # Reduce latency settings
        cap.set(cv2.CAP_PROP_PROBESIZE, 1024)
        cap.set(cv2.CAP_PROP_MAX_DELAY, 1)
        
        print("✅ RTSP camera configuration applied")
        
    elif rtsp_url.startswith("http"):
        # For HTTP streams (IP cameras)
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("✅ HTTP camera configuration applied")
        
    else:
        # For other sources (local cameras, etc.)
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("✅ Generic camera configuration applied")
    
    # Test if camera opened successfully
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return None
    else:
        print("✅ Camera opened successfully")
    
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

# === Enhanced Main Detection Loop ===
def run_detection_loop():
    global cap, camera_active, frame_count, last_location_check_time
    global count_cars, count_bikes, count_trucks
    
    # Initialize camera
    cap = init_camera()
    if cap is None:
        print("❌ Failed to initialize camera. Exiting...")
        return
    
    read_current_location()

    frame_count = 0
    last_location_check_time = time.time()
    camera_active = True
    connection_retry_count = 0
    max_retries = 5

    print("🚀 Starting vehicle detection loop...")

    while True:
        current_time = time.time()
        
        # Check location every 5 seconds
        if current_time - last_location_check_time >= 5:
            read_current_location()
            last_location_check_time = current_time

        # Enhanced frame reading based on stream type
        if rtsp_url.startswith("rtsp://"):
            # For RTSP streams, read frame directly
            ret, frame = cap.read()
        elif rtsp_url.startswith("http"):
            # For HTTP streams
            ret, frame = cap.read()
        else:
            # For other sources, use buffer clearing
            for _ in range(2):
                cap.grab()
            ret, frame = cap.read()

        # Handle frame reading errors
        if not ret or frame is None or frame.size == 0:
            print(f"⚠️ Frame reading failed. Retry count: {connection_retry_count}")
            camera_active = False
            
            connection_retry_count += 1
            if connection_retry_count >= max_retries:
                print("❌ Max retries reached. Attempting to reinitialize camera...")
                cap.release()
                time.sleep(5)  # Wait before reconnecting
                cap = init_camera()
                if cap is None:
                    print("❌ Camera reinitialization failed. Exiting...")
                    break
                connection_retry_count = 0
            else:
                time.sleep(2)  # Short wait before retry
            continue

        # Reset retry count on successful frame read
        if connection_retry_count > 0:
            print("✅ Camera connection restored")
            connection_retry_count = 0
        
        camera_active = True

        # Skip every other frame for performance (optional)
        frame_count += 1
        if frame_count % 2 != 0:
            continue

        # Create a copy for display
        display_frame = frame.copy()
        
        # Run YOLO detection
        try:
            results = model.track(display_frame, persist=True, conf=0.5, tracker="bytetrack.yaml")
        except Exception as e:
            print(f"⚠️ YOLO detection error: {e}")
            continue

        # Process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for box_id, cls, coord, conf in zip(ids, classes, coords, confs):
                x1, y1, x2, y2 = coord
                center_y = int((y1 + y2) / 2)
                label = model.names[int(cls)]

                # Only process vehicles with high confidence
                if label in ["car", "motorcycle", "truck"] and conf > 0.5:
                    # Draw bounding box
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    label_text = f"{label}-{int(box_id)} ({conf:.2f})"
                    cv2.putText(display_frame, label_text, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Check if vehicle crossed the counting line
                    if (count_line_position - offset < center_y < count_line_position + offset and
                            box_id not in counted_ids):
                        
                        # Add to counted IDs
                        counted_ids.add(box_id)
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Log to CSV
                        try:
                            csv_writer.writerow([timestamp, label, int(box_id), CAMERA_LOCATION_ID])
                            csv_file.flush()
                        except Exception as e:
                            print(f"⚠️ CSV logging error: {e}")

                        # Log to database
                        try:
                            db_cursor.execute(
                                "INSERT INTO vehicles (timestamp, vehicle_type, vehicle_id, location_id) VALUES (?, ?, ?, ?)",
                                (timestamp, label, int(box_id), CAMERA_LOCATION_ID)
                            )
                            db_conn.commit()
                        except Exception as e:
                            print(f"⚠️ Database logging error: {e}")

                        print(f"✔ Counted {label}-{int(box_id)} at {timestamp} for location {CAMERA_LOCATION_ID}")

                        # Update counters
                        if label == "car":
                            count_cars += 1
                        elif label == "motorcycle":
                            count_bikes += 1
                        elif label == "truck":
                            count_trucks += 1

        # Draw counting line
        cv2.line(display_frame, (0, count_line_position), (display_frame.shape[1], count_line_position), (0, 0, 255), 3)
        cv2.putText(display_frame, "COUNTING LINE", (10, count_line_position - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display counts
        cv2.putText(display_frame, f"Cars: {count_cars} | Bikes: {count_bikes} | Trucks: {count_trucks}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display location
        cv2.putText(display_frame, f"Location: {CAMERA_LOCATION_ID}",
                    (20, display_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display camera status
        status_text = "LIVE" if camera_active else "RECONNECTING"
        status_color = (0, 255, 0) if camera_active else (0, 165, 255)
        cv2.putText(display_frame, status_text, (display_frame.shape[1] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Update the latest frame for streaming
        with frame_lock:
            latest_frame = display_frame.copy()

        # Display the frame
        cv2.imshow("Vehicle Detection & Counting (CP Plus Camera)", display_frame)
        
        # Check for exit key (ESC)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("🔻 ESC key pressed. Exiting...")
            cleanup()

    # Cleanup if loop exits
    cleanup()

# === Run detection only if backend.py is run directly ===
if __name__ == "__main__":
    print("🚀 Starting CP Plus Camera Vehicle Detection System")
    print(f"📹 Camera URL: {rtsp_url}")
    print("Press ESC to exit")
    print("-" * 60)
    
    try:
        run_detection_loop()
    except KeyboardInterrupt:
        print("\n🔻 Keyboard interrupt received")
        cleanup()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        cleanup()
