import cv2
import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    sl = None
    print("[aruco_detector] pyzed not found — ZED functions unavailable (webcam only)")

# ── CONFIG ──────────────────────────────────────────────────────────────────
ARUCO_DICT  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECTOR    = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())

ANCHOR_ID   = 101   # Fixed anchor marker
HMD_ID      = 0     # HMD marker
MARKER_SIZE = 0.1   # meters

CAMERA_IP   = "192.168.50.3"
CAMERA_PORT = 30000
# ────────────────────────────────────────────────────────────────────────────

def get_K_from_zed(zed):
    calib = zed.get_camera_information().camera_configuration.calibration_parameters
    fx, fy = calib.left_cam.fx, calib.left_cam.fy
    cx, cy = calib.left_cam.cx, calib.left_cam.cy
    K    = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(4, dtype=np.float64)
    return K, dist

def get_marker_object_points(size):
    half = size / 2.0
    return np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)

def get_pose(corners, K, dist):
    """Get 4x4 transform matrix from marker corners."""
    obj_points = get_marker_object_points(MARKER_SIZE)
    img_points = corners.reshape(4, 2)
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
    if not success:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tvec.flatten()
    return T

def detect_markers(gray, K, dist):
    """Detect anchor and HMD markers in a grayscale frame.
    Returns (anchor_T, hmd_T) — either can be None."""
    corners, ids, _ = DETECTOR.detectMarkers(gray)

    anchor_T = None
    hmd_T    = None

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            T = get_pose(corners[i], K, dist)
            if T is None:
                continue
            if marker_id == ANCHOR_ID:
                anchor_T = T
            elif marker_id == HMD_ID:
                hmd_T = T

    return anchor_T, hmd_T, corners, ids

def run(on_pose_detected):
    """
    Run live ZED stream detection loop.
    on_pose_detected(anchor_T, hmd_T) called when both markers detected.
    """
    init_params = sl.InitParameters()
    init_params.set_from_stream(CAMERA_IP, CAMERA_PORT)

    zed = sl.Camera()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[aruco_detector] ZED connection failed: {status}")
        return

    K, dist = get_K_from_zed(zed)
    print("[aruco_detector] ZED connected.")

    image          = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    try:
        while True:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame     = image.get_data()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            anchor_T, hmd_T, corners, ids = detect_markers(gray, K, dist)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)

            if anchor_T is not None and hmd_T is not None:
                on_pose_detected(anchor_T, hmd_T)

            cv2.imshow("ZED ArUco", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        zed.close()
        print("[aruco_detector] ZED terminated")


def run_webcam(on_pose_detected, camera_index=0):
    """
    Run detection loop using a webcam instead of ZED (for testing without cameras).
    on_pose_detected(anchor_T, hmd_T) called when both markers detected.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[aruco_detector] Webcam connection failed")
        return

    # 웹캠은 ZED처럼 정확한 K를 모르니까 대략적인 값 사용
    ret, frame = cap.read()
    if not ret:
        print("[aruco_detector] Failed to read frame")
        return

    h, w = frame.shape[:2]
    focal = w  # rough approximation
    K = np.array([[focal, 0, w/2],
                  [0, focal, h/2],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(4, dtype=np.float64)

    print("[aruco_detector] Webcam connected. (approximate K matrix)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            anchor_T, hmd_T, corners, ids = detect_markers(gray, K, dist)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if anchor_T is not None and hmd_T is not None:
                on_pose_detected(anchor_T, hmd_T)

            cv2.imshow("Webcam ArUco", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[aruco_detector] Webcam terminated")