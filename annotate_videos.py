import os
import cv2
import csv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import draw_boxes

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Load YOLOv11 model
model = YOLO('weights/yolo11n.pt')

def load_mapping(path='player_id_mapping.csv'):
    mapping = {}
    try:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[int(row['Tacticam_ID'])] = int(row['Broadcast_ID'])
        print("[INFO] Loaded player ID mapping.")
    except FileNotFoundError:
        print("[ERROR] player_id_mapping.csv file not found.")
    return mapping

def annotate_video(input_path, output_path, mapping, camera_type):
    print(f"[INFO] Starting annotation: {camera_type} video")

    if not os.path.exists(input_path):
        print(f"[ERROR] Input video not found: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0 or width == 0 or height == 0:
        print(f"[ERROR] Failed to read video metadata: {input_path}")
        cap.release()
        return

    # Using XVID codec (widely supported)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    tracker = DeepSort(max_age=30)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect players
        results = model(frame)[0]
        detections = []
        for r in results.boxes:
            if int(r.cls) == 0:  # player class
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                conf = float(r.conf)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'player'))

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw boxes + IDs
        annotated_frame = draw_boxes(frame, tracks, camera_type, mapping)

        # Write frame to output video
        out.write(annotated_frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"[INFO] {camera_type}: Processed {frame_count} frames")

    cap.release()
    out.release()
    print(f"[INFO] Finished annotation for {camera_type}. Saved to {output_path}")

def annotate_all():
    mapping = load_mapping()
    annotate_video('videos/broadcast.mp4', 'output/broadcast_annotated.mp4', mapping, 'broadcast')
    annotate_video('videos/tacticam.mp4', 'output/tacticam_annotated.mp4', mapping, 'tacticam')


if __name__ == "__main__":
    annotate_all()

