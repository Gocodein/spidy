import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLOv8 model (pre-trained on COCO dataset)
# 'yolov8n.pt' is the nano model (fastest). Use 'yolov8m.pt' for better accuracy.
model = YOLO('yolov8n.pt')

# COCO dataset classes relevant to animals
ANIMAL_CLASSES = [15, 16, 17, 18, 19, 20, 21, 22, 23]  # cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

# Video source: Replace with 'path/to/video.mp4' or use 0 for webcam
video_path = 0 bh
cap = cv2.VideoCapture(video_path)

# Store track history to calculate speed
track_history = defaultdict(lambda: [])
previous_positions = {}

def get_behavior(speed):
    """Simple heuristic to determine behavior based on speed (pixels/frame)."""
    if speed < 1.0:
        return "Resting"
    elif speed < 15.0:
        return "Walking"
    else:
        return "Running"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:
        # Get the boxes, class IDs, and track IDs
        boxes = results[0].boxes.xywh.cpu()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            # Process only if detected object is an animal
            if class_id in ANIMAL_CLASSES:
                x, y, w, h = box
                center = (float(x), float(y))
                
                # Calculate speed
                speed = 0
                if track_id in previous_positions:
                    prev_center = previous_positions[track_id]
                    # Euclidean distance between current and previous frame center
                    speed = np.linalg.norm(np.array(center) - np.array(prev_center))
                
                # Update previous position
                previous_positions[track_id] = center
                
                # Determine behavior
                behavior = get_behavior(speed)
                
                # --- Visualization ---
                # Draw bounding box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label (Animal Type + Behavior)
                animal_name = model.names[class_id]
                label = f"{animal_name} ({track_id}): {behavior}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Animal Behavior Detection", frame)

    # Press 'q' to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()