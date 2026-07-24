import cv2
import numpy as np
import os


# ── CONFIG ──────────────────────────────────────────────────────────────────
SAVE_DIR     = "webcam_calibration/images"
CAMERA_INDEX = 0


CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
SQUARE_LENGTH     = 0.04   # 40mm in meters
MARKER_LENGTH     = 0.03   # 30mm in meters
ARUCO_DICT        = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# ────────────────────────────────────────────────────────────────────────────


os.makedirs(SAVE_DIR, exist_ok=True)


board    = cv2.aruco.CharucoBoard(
    (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
    SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT
)
detector = cv2.aruco.CharucoDetector(board)


cap   = cv2.VideoCapture(CAMERA_INDEX)
count = 0


print("Press SPACE to capture, Q to quit")
print(f"Images will be saved to {SAVE_DIR}")


while True:
    ret, frame = cap.read()
    if not ret:
        continue


    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)


    display = frame.copy()


    if marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
    if charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
        cv2.putText(display, f"Corners: {len(charuco_ids)} — SPACE to capture",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, "No board detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.putText(display, f"Captured: {count}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


    cv2.imshow("Webcam Calibration", display)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and charuco_ids is not None and len(charuco_ids) >= 4:
        filename = os.path.join(SAVE_DIR, f"frame_{count:03d}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
        print(f"Captured {filename} ({count} total)")


cap.release()
cv2.destroyAllWindows()
print(f"Done. {count} images saved.")



