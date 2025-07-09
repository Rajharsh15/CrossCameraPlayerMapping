import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.optimize import linear_sum_assignment
import os
import csv

# === CONFIGURATION ===
BROADCAST_VIDEO = 'videos/broadcast.mp4'
TACTICAM_VIDEO = 'videos/tacticam.mp4'
MODEL_PATH = 'weights/yolo11n.pt'  # Ensure this matches the downloaded model name

# === LOAD YOLO MODEL ===
print("[INFO] Loading YOLOv11 model...")
model = YOLO(MODEL_PATH)

# === INITIALIZE DEEP SORT ===
tracker_broadcast = DeepSort(max_age=30)
tracker_tacticam = DeepSort(max_age=30)

# === FUNCTION: HISTOGRAM FEATURE EXTRACTION ===
def get_histogram(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        # Return zero histogram as float32 vector
        return np.zeros((50 * 60,), dtype=np.float32)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


# === FUNCTION: TRACK PLAYERS IN VIDEO ===
def track_players(video_path, model, tracker):
    cap = cv2.VideoCapture(video_path)
    features = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = []
        results = model(frame)[0]

        for r in results.boxes:
            if int(r.cls) == 0:  # class 0 = player
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                conf = float(r.conf)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'player'))

        tracks = tracker.update_tracks(detections, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue
            track_id = t.track_id
            bbox = t.to_ltrb()
            hist = get_histogram(frame, bbox)
            features[track_id] = hist

        frame_count += 1

    cap.release()
    return features

# === RUN TRACKING ===
print("[INFO] Tracking players in broadcast video...")
broadcast_features = track_players(BROADCAST_VIDEO, model, tracker_broadcast)

print("[INFO] Tracking players in tacticam video...")
tacticam_features = track_players(TACTICAM_VIDEO, model, tracker_tacticam)

# === PLAYER MATCHING ===
print("[INFO] Matching players using visual features...")
broadcast_ids = list(broadcast_features.keys())
tacticam_ids = list(tacticam_features.keys())

cost_matrix = np.zeros((len(tacticam_ids), len(broadcast_ids)))

for i, tid2 in enumerate(tacticam_ids):
    for j, tid1 in enumerate(broadcast_ids):
        dist = cv2.compareHist(broadcast_features[tid1], tacticam_features[tid2], cv2.HISTCMP_BHATTACHARYYA)
        cost_matrix[i, j] = dist

row_ind, col_ind = linear_sum_assignment(cost_matrix)

mapping = {tacticam_ids[i]: broadcast_ids[j] for i, j in zip(row_ind, col_ind)}

# === PRINT RESULT ===
print("\n[RESULT] Player ID Mapping (Tacticam → Broadcast):")
for tid2, tid1 in mapping.items():
    print(f"Tacticam ID {tid2} → Broadcast ID {tid1}")

# === SAVE TO CSV ===
with open('player_id_mapping.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Tacticam_ID', 'Broadcast_ID'])
    for tid2, tid1 in mapping.items():
        writer.writerow([tid2, tid1])

print("\n[INFO] Mapping saved to 'player_id_mapping.csv'")