import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
from aruco_dectector import run

def on_pose_detected(rvec, tvec, frame, center):
    cx, cy = center
    cv2.putText(frame, "<3", (cx - 20, cy - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (91, 32, 0), 4)

run(on_pose_detected)