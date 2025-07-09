# utils.py

import cv2
import numpy as np

def get_histogram(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return np.zeros((50 * 60,), dtype=np.float32)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)

def draw_boxes(frame, tracks, camera_type, mapping):
    for t in tracks:
        if not t.is_confirmed():
            continue
        tid = t.track_id
        bbox = t.to_ltrb()
        x1, y1, x2, y2 = map(int, bbox)

        if camera_type == 'tacticam':
            consistent_id = mapping.get(tid, tid)
        else:
            consistent_id = tid

        color = (0, 255, 0) if camera_type == 'broadcast' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID: {consistent_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame
