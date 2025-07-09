# Cross-Camera Player ID Mapping System

This project detects and tracks football players across two different camera angles using a YOLOv11 model and matches their identities consistently.

## 📦 Requirements

- Python 3.8+
- pip packages:

## 📁 Files

- `match_players.py`: Detects & tracks players, maps player IDs.
- `annotate_videos.py`: Annotates videos with consistent IDs.
- `utils.py`: Histogram, drawing, helper functions.
- `main.py`: Runs full pipeline.
- `player_id_mapping.csv`: Final ID mapping results.
- `weights/`: Contains `yolo11n.pt` model.
- `videos/`: Input videos.
- `output/`: Annotated results.

## ▶️ How to Run

1. Place `broadcast.mp4` and `tacticam.mp4` in the `videos/` folder.
2. Place `yolo11n.pt` in the `weights/` folder.
3. Run the full pipeline:
4. Check `output/` folder for annotated videos and mapping CSV.

