import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
from aruco_detector import run

COLORS = [
    ((91, 32, 0), "Rice University Blue"),
    ((174, 143, 127), "Rice University Blue - Tint: 50%"),
    ((127, 126, 124), "Rice University Gray"),
    ((191, 190, 189), "Rice University Gray - Tint: 50%")]

current_color_idx = 0

def draw_heart(frame, center):
    global current_color_idx
    cx, cy = center
    color, name = COLORS[current_color_idx]

    cv2.putText(frame, "<3", (cx - 20, cy - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

    cv2.putText(frame, f"Color: {name}, (left/right arrow to change)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

def on_pose_detected(rvec, tvec, frame, center):
    global current_color_idx
    draw_heart(frame, center)

    key = cv2.waitKey(1) & 0xFF
    if key == 81 or 2:
        current_color_idx = (current_color_idx - 1) % len(COLORS)
    elif key == 83 or 3:
        current_color_idx = (current_color_idx + 1) % len(COLORS)

run(on_pose_detected)