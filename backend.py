import cv2
from ultralytics import YOLO
import numpy as np
import datetime
import csv
import os
import time

# === CONFIGURATION ===
VIDEO_PATH = r"D:\clips\testclip3.mp4"
MODEL_PATH = r"yolov8m.pt"  # Using YOLOv8n for faster performance
CSV_FILENAME = "vehicle_log_from_video.csv"

LINE_START = (100, 180)
LINE_END = (700, 50)
OFFSET = 15

FRAME_SKIP = 1   # Process every frame, no skipping
RESIZE_WIDTH = 960
RESIZE_HEIGHT = 540

# === INIT ===
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

count_cars = count_bikes = count_trucks = 0
counted_ids = set()
object_memory = {}  # {vehicle_id: (prev_center_x, prev_center_y)}
frame_count = 0

# CSV Logging
csv_file = open(CSV_FILENAME, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Vehicle Type", "Vehicle ID"])

# === CROSS LINE DETECTION FUNCTION ===
def crossed_line(prev, curr, line_start, line_end):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(line_start, prev, curr) != ccw(line_end, prev, curr) and \
           ccw(line_start, line_end, prev) != ccw(line_start, line_end, curr)

# === MAIN LOOP ===
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

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

            if label in ["car", "motorcycle", "truck"] and conf > 0.25:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}-{int(box_id)}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                prev_center = object_memory.get(box_id, (center_x, center_y))
                object_memory[box_id] = (center_x, center_y)

                if crossed_line(prev_center, (center_x, center_y), LINE_START, LINE_END) and box_id not in counted_ids:
                    counted_ids.add(box_id)
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    csv_writer.writerow([timestamp, label, int(box_id)])
                    csv_file.flush()

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

    # FPS counter (debug)
    elapsed_time = time.time() - start_time
    fps = int(frame_count / elapsed_time)
    cv2.putText(frame, f"FPS: {fps}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Vehicle Detection & Counting", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

# === CLEANUP ===
cap.release()
csv_file.close()
cv2.destroyAllWindows()
