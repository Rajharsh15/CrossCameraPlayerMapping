# main.py

from match_players import match_players
from annotate_videos import annotate_all

if __name__ == '__main__':
    print("[1] Matching players across both videos...")
    match_players()

    print("[2] Annotating videos with consistent player IDs...")
    annotate_all()

    print("[✓] All tasks completed. Check the output folder.")
