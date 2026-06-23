import pyzed.sl as sl
import cv2
import numpy as np
from pose_calculator import anchor_to_hmd_pose

# ── CONFIG ──────────────────────────────────────────────────────────────────
ARUCO_DICT  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECTOR    = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())

ANCHOR_ID   = 100   # Fixed anchor marker
HMD_ID      = 0     # HMD marker
MARKER_SIZE = 0.1

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

def run(on_pose_detected):
    """
    on_pose_detected: callback called when both markers detected
    receives rel_T (4x4 transform matrix of HMD in anchor space)
    """
    init_params = sl.InitParameters()
    init_params.set_from_stream(CAMERA_IP, CAMERA_PORT)

    zed = sl.Camera()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[aruco_detector] ZED connection failed: {status}")
        return

    K, dist = get_K_from_zed(zed)
    print(f"[aruco_detector] ZED connected.")

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

            if anchor_T is not None and hmd_T is not None:
                rel_T = anchor_to_hmd_pose(anchor_T, hmd_T)
                on_pose_detected(rel_T)

            cv2.imshow("ZED ArUco", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        zed.close()
        print("[aruco_detector] ZED terminated")