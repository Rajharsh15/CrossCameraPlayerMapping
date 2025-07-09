# Report: Cross-Camera Player Mapping

## ✅ Objective
Match player identities across two videos from different angles using consistent IDs.

## 🔍 Methodology

1. **Detection & Tracking:**
   - Used YOLOv11 (Ultralytics) for player detection.
   - Used Deep SORT for real-time object tracking per video.

2. **Visual Feature Matching:**
   - Extracted HSV histograms of each player.
   - Matched players using Bhattacharyya distance + Hungarian algorithm.

3. **Annotation:**
   - Used OpenCV to draw consistent player IDs and save annotated videos.

## 🧪 Techniques Tried

- Histograms (color-based): worked well for most players.
- IOU + motion-based matching: considered but skipped due to camera angle difference.
- Deep feature embeddings: not used to avoid additional complexity.

## ⚠️ Challenges

- Non-overlapping views make identity matching harder.
- Players with similar uniforms (colors) caused near matches.
- Detection confidence had to be tuned to avoid false positives.

## ✅ Outcome

- Successfully mapped players across both views.
- Annotated videos confirm consistent ID overlays.
- Results saved in CSV and videos — reproducible and interpretable.

