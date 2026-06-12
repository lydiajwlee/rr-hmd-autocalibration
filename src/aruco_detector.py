import pyzed.sl as sl
import cv2
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────────────
ARUCO_DICT  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECTOR    = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())
MARKER_ID   = 0       # ArUco marker ID
MARKER_SIZE = 0.1     # meter unit 실제 태그 크기로

CAMERA_IP   = "192.168.50.3"
CAMERA_PORT = 30000   # label code 004
# ────────────────────────────────────────────────────────────────────────────

def get_K_from_zed(zed):
    calib = zed.get_camera_information().camera_configuration.calibration_parameters
    fx = calib.left_cam.fx
    fy = calib.left_cam.fy
    cx = calib.left_cam.cx
    cy = calib.left_cam.cy
    K    = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)
    dist = np.zeros(4, dtype=np.float64)  # ZED는 rectified라서 0
    return K, dist

def get_marker_object_points(size):
    half = size / 2.0
    return np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)

def detect_pose(gray, K, dist):
    corners, ids, _ = DETECTOR.detectMarkers(gray)

    if ids is None:
        return None, None

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id != MARKER_ID:
            continue

        obj_points = get_marker_object_points(MARKER_SIZE)
        img_points = corners[i].reshape(4, 2)

        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
        if not success:
            return None, None

        return rvec, tvec

    return None, None

def run(on_pose_detected):
    """
    on_pose_detected: pose 감지될 때마다 호출되는 콜백 함수
    rvec, tvec을 인자로 받아
    """
    init_params = sl.InitParameters()
    init_params.set_from_stream(CAMERA_IP, CAMERA_PORT)

    zed = sl.Camera()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[aruco_detector] ZED connection failed: {status}")
        return

    K, dist = get_K_from_zed(zed)
    print(f"[aruco_detector] ZED connection success. K matrix:\n{K}")

    image = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    try:
        while True:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            rvec, tvec = detect_pose(gray, K, dist)

            if rvec is not None:
                on_pose_detected(rvec, tvec)

            # deb
            cv2.imshow("ZED ArUco", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        zed.close()
        print("[aruco_detector] ZED terminated")