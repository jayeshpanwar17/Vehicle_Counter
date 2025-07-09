import cv2
from ultralytics import YOLO
import numpy as np
import datetime
import csv
import os
import time

# === CONFIG ===
VIDEO_PATH = r"D:\clips\testclip3.mp4"
MODEL_PATH = r"D:\project_folder\best.pt"  # Your trained model
CSV_FILENAME = "vehicle_log_from_video.csv"

# Line for counting (adjust for your video)
LINE_START = (100, 180)
LINE_END = (700, 50)

# Class Names as per your model
TARGET_CLASSES = ["car", "motorcycle", "truck", "Auto-Rickshaw", "Bus", "HCV", "LCV", "Toto"]

CONFIDENCE_THRESHOLD = 0.3
FRAME_SKIP = 1
RESIZE_WIDTH = 960
RESIZE_HEIGHT = 540

# === INIT ===
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
counted_ids = set()
object_memory = {}

# Vehicle counters
vehicle_counts = {cls: 0 for cls in TARGET_CLASSES}

# Setup CSV
csv_file = open(CSV_FILENAME, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Vehicle Type", "Vehicle ID"])

# Line crossing detection
def crossed_line(prev, curr, line_start, line_end):
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(line_start, prev, curr) != ccw(line_end, prev, curr) and \
           ccw(line_start, line_end, prev) != ccw(line_start, line_end, curr)

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    results = model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, tracker="bytetrack.yaml")

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for box_id, cls, coord, conf in zip(ids, classes, coords, confs):
            label = model.names[int(cls)]
            if label not in TARGET_CLASSES or conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = coord.astype(int)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}-{int(box_id)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Count if cross line
            prev_center = object_memory.get(box_id, (center_x, center_y))
            object_memory[box_id] = (center_x, center_y)

            if crossed_line(prev_center, (center_x, center_y), LINE_START, LINE_END) and box_id not in counted_ids:
                counted_ids.add(box_id)
                vehicle_counts[label] += 1
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_writer.writerow([timestamp, label, int(box_id)])
                csv_file.flush()

    # Draw line
    cv2.line(frame, LINE_START, LINE_END, (0, 0, 255), 2)

    # Show counts
    count_text = " | ".join([f"{cls}: {vehicle_counts[cls]}" for cls in TARGET_CLASSES if vehicle_counts[cls] > 0])
    cv2.putText(frame, count_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # FPS
    fps = int(frame_count / (time.time() - start_time))
    cv2.putText(frame, f"FPS: {fps}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display
    cv2.imshow("Vehicle Detection & Counting", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

# === CLEANUP ===
cap.release()
csv_file.close()
cv2.destroyAllWindows()
