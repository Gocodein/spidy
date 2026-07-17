"""
EndangeredSpeciesDetector.py
------------------------------------------------------------
Architecture implemented here:
  Stage 1: Generic detector (YOLOv8 pretrained OR fine-tuned) -> animal / human / vehicle
  Stage 2: (placeholder) Species classifier hook -- plug in your
           fine-tuned EfficientNet/ViT model here once trained
  Stage 3: Multi-object tracking (ByteTrack via YOLO's built-in tracker)
  Stage 4: Behavior state estimation from trajectory features
  Stage 5: Human-disturbance detection (proximity between human & animal tracks)
  Stage 6: Structured logging to SQLite for later analysis / dashboarding

Replace 'yolov8n.pt' with your own fine-tuned weights once you have a
custom-trained model on your target endangered species.
------------------------------------------------------------
"""

import cv2
import numpy as np
import sqlite3
import time
from collections import defaultdict, deque
from ultralytics import YOLO

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
VIDEO_SOURCE = 0          # 0 = webcam, or 'path/to/video.mp4'
MODEL_WEIGHTS = "yolov8n.pt"   # swap for your fine-tuned species model later
DB_PATH = "wildlife_events.db"

# COCO class IDs relevant to generic "animal" (Stage 1 fallback until you
# fine-tune your own species classes)
ANIMAL_CLASSES = {15, 16, 17, 18, 19, 20, 21, 22, 23}
HUMAN_CLASS = 0            # COCO 'person'
VEHICLE_CLASSES = {2, 3, 5, 7}   # car, motorcycle, bus, truck

DISTURBANCE_DISTANCE_PX = 250     # proximity threshold to flag disturbance
BEHAVIOR_WINDOW = 15              # frames used to smooth behavior decisions

# ------------------------------------------------------------------
# DATABASE SETUP (Stage 6: logging)
# ------------------------------------------------------------------
def init_db(path):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            track_id INTEGER,
            class_name TEXT,
            behavior TEXT,
            x REAL, y REAL,
            disturbance_flag INTEGER
        )
    """)
    conn.commit()
    return conn

def log_event(conn, track_id, class_name, behavior, x, y, disturbance):
    conn.execute(
        "INSERT INTO events (timestamp, track_id, class_name, behavior, x, y, disturbance_flag) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (time.time(), track_id, class_name, behavior, x, y, int(disturbance))
    )
    conn.commit()

# ------------------------------------------------------------------
# STAGE 4: BEHAVIOR ESTIMATION
# Uses a short history of positions per track to compute speed AND
# direction variance, which is a much better signal than speed alone.
# ------------------------------------------------------------------
class BehaviorEstimator:
    def __init__(self, window=BEHAVIOR_WINDOW):
        self.window = window
        self.history = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id, center):
        self.history[track_id].append(center)

    def classify(self, track_id):
        pts = list(self.history[track_id])
        if len(pts) < 3:
            return "Observing"

        pts_arr = np.array(pts)
        deltas = np.diff(pts_arr, axis=0)
        speeds = np.linalg.norm(deltas, axis=1)
        avg_speed = float(np.mean(speeds))

        # direction variance: high variance + low speed = agitated/pacing
        # low variance + high speed = fleeing/traveling in a line
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        angle_var = float(np.var(angles)) if len(angles) > 1 else 0.0

        if avg_speed < 1.0:
            return "Resting"
        elif avg_speed < 8.0 and angle_var > 1.5:
            return "Alert/Pacing"     # low travel, erratic direction -> possible stress
        elif avg_speed < 15.0:
            return "Walking/Grazing"
        else:
            return "Fleeing/Running"

# ------------------------------------------------------------------
# STAGE 5: HUMAN-DISTURBANCE CHECK
# ------------------------------------------------------------------
def check_disturbance(animal_center, human_centers, threshold=DISTURBANCE_DISTANCE_PX):
    for hc in human_centers:
        dist = np.linalg.norm(np.array(animal_center) - np.array(hc))
        if dist < threshold:
            return True
    return False

# ------------------------------------------------------------------
# STAGE 2 PLACEHOLDER: species classifier hook
# Once you fine-tune a classifier, load it here and call it on the
# cropped animal region instead of relying on the generic COCO label.
# ------------------------------------------------------------------
def classify_species(frame, box):
    """
    Placeholder. Replace with:
        crop = frame[y1:y2, x1:x2]
        species_label, confidence = species_model.predict(crop)
        return species_label
    """
    return None  # falls back to generic COCO class name

# ------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------
def main():
    model = YOLO(MODEL_WEIGHTS)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    conn = init_db(DB_PATH)
    behavior_estimator = BehaviorEstimator()

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

        human_centers = []
        animal_detections = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # First pass: collect human positions for disturbance check
            for box, class_id in zip(boxes, class_ids):
                if class_id == HUMAN_CLASS:
                    x, y, w, h = box
                    human_centers.append((float(x), float(y)))

            # Second pass: process animals
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id not in ANIMAL_CLASSES:
                    continue

                x, y, w, h = box
                center = (float(x), float(y))

                behavior_estimator.update(track_id, center)
                behavior = behavior_estimator.classify(track_id)

                species_override = classify_species(frame, box)
                class_name = species_override or model.names[class_id]

                disturbed = check_disturbance(center, human_centers)

                log_event(conn, track_id, class_name, behavior, center[0], center[1], disturbed)

                # --- Visualization ---
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                color = (0, 0, 255) if disturbed else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name} #{track_id}: {behavior}"
                if disturbed:
                    label += " [HUMAN NEARBY]"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Draw human boxes too, for context
            for box, class_id in zip(boxes, class_ids):
                if class_id == HUMAN_CLASS:
                    x, y, w, h = box
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.putText(frame, "human", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("Endangered Species Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()


if __name__ == "__main__":
    main()
