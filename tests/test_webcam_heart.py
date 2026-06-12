import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECTOR   = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())
MARKER_ID  = 0

def draw_heart(frame, center):
    cx, cy = center
    cv2.putText(frame, "<3", (cx - 20, cy - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

cap = cv2.VideoCapture(0)

print("Webcam running — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id != MARKER_ID:
                continue
            center = corners[i][0].mean(axis=0).astype(int)
            draw_heart(frame, center)
    else:
        cv2.putText(frame, "No markers detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Webcam Heart Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()