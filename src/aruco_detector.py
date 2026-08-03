import cv2
import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    sl = None
    print("[aruco_detector] pyzed not found — ZED functions unavailable (webcam only)")

# ── CONFIG ──────────────────────────────────────────────────────────────────
ARUCO_DICT  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECTOR_PARAMS = cv2.aruco.DetectorParameters()
DETECTOR_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

ANCHOR_IDS  = (100, 101)  # Fixed wall markers required for calibration
HMD_ID      = 0           # HMD marker
ANCHOR_MARKER_SIZE = 0.179  # 179 mm, in meters
HMD_MARKER_SIZE    = 0.096  # 96 mm, in meters
DETECTION_ZOOM = 1.2  # centered crop used for detection and preview

CAMERA_IP   = "192.168.50.3"
CAMERA_PORT = 30000
# ────────────────────────────────────────────────────────────────────────────

def zoom_detection_frame(frame, K, zoom=DETECTION_ZOOM):
    """Center-zoom a frame and return the corresponding camera matrix."""
    if zoom < 1.0:
        raise ValueError("Detection zoom must be at least 1.0")
    if zoom == 1.0:
        return frame, K.copy()

    height, width = frame.shape[:2]
    crop_width = max(1, round(width / zoom))
    crop_height = max(1, round(height / zoom))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = frame[top:top + crop_height, left:left + crop_width]
    zoomed = cv2.resize(
        cropped, (width, height), interpolation=cv2.INTER_LINEAR
    )

    scale_x = width / crop_width
    scale_y = height / crop_height
    zoomed_K = K.copy().astype(np.float64)
    zoomed_K[0, 0] *= scale_x
    zoomed_K[1, 1] *= scale_y
    zoomed_K[0, 2] = (K[0, 2] - left) * scale_x
    zoomed_K[1, 2] = (K[1, 2] - top) * scale_y
    return zoomed, zoomed_K

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

def get_pose(corners, marker_size, K, dist):
    """Get 4x4 transform matrix from marker corners."""
    obj_points = get_marker_object_points(marker_size)
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
    """Detect both wall anchors and the HMD marker in one grayscale frame.

    Returns ({anchor_id: camera_T_anchor}, camera_T_hmd, corners, ids).
    The anchor dictionary can be incomplete when one or both anchors are hidden.
    """
    corners, ids, _ = DETECTOR.detectMarkers(gray)

    anchor_transforms = {}
    hmd_T             = None

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in ANCHOR_IDS:
                marker_size = ANCHOR_MARKER_SIZE
            elif marker_id == HMD_ID:
                marker_size = HMD_MARKER_SIZE
            else:
                continue

            T = get_pose(corners[i], marker_size, K, dist)
            if T is None:
                continue
            if marker_id in ANCHOR_IDS:
                anchor_transforms[int(marker_id)] = T
            elif marker_id == HMD_ID:
                hmd_T = T

    return anchor_transforms, hmd_T, corners, ids

def run(on_pose_detected):
    """
    Detect one valid anchor/HMD pair from the ZED stream, then stop.
    on_pose_detected(anchor_transforms, hmd_T) is called exactly once, after
    both configured wall anchors and the HMD are visible in the same frame.
    """
    if sl is None:
        raise RuntimeError("pyzed is required to use the ZED camera")

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
    calibrated     = False

    try:
        while not calibrated:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame     = image.get_data()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            detection_frame, detection_K = zoom_detection_frame(frame_bgr, K)
            gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)

            anchor_transforms, hmd_T, corners, ids = detect_markers(
                gray, detection_K, dist
            )

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(detection_frame, corners, ids)

            anchors_ready = all(
                anchor_id in anchor_transforms for anchor_id in ANCHOR_IDS
            )
            if not calibrated and anchors_ready and hmd_T is not None:
                on_pose_detected(anchor_transforms, hmd_T)
                calibrated = True
                print("[aruco_detector] Calibration captured; stopping detection")

            if not calibrated:
                cv2.imshow("ZED ArUco", detection_frame)
            if not calibrated and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        zed.close()
        print("[aruco_detector] ZED terminated")


def run_webcam(on_pose_detected, camera_index=0, intrinsics_path=None):
    """Detect one valid anchor/HMD pair from a webcam, then stop."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[aruco_detector] Webcam connection failed")
        return

    if intrinsics_path is not None:
        data = np.load(intrinsics_path)
        K    = data['K']
        dist = data['dist']
        print(f"[aruco_detector] Loaded K matrix from {intrinsics_path}")
    else:
        ret, frame = cap.read()
        if not ret:
            print("[aruco_detector] Failed to read frame")
            cap.release()
            return

        h, w = frame.shape[:2]
        K    = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(4, dtype=np.float64)
        print("[aruco_detector] Using approximate K matrix")

    print("[aruco_detector] Webcam connected — press Q to quit")
    calibrated = False

    try:
        while not calibrated:
            ret, frame = cap.read()
            if not ret:
                continue

            detection_frame, detection_K = zoom_detection_frame(frame, K)
            gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
            anchor_transforms, hmd_T, corners, ids = detect_markers(
                gray, detection_K, dist
            )

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(detection_frame, corners, ids)

            anchors_ready = all(
                anchor_id in anchor_transforms for anchor_id in ANCHOR_IDS
            )
            if not calibrated and anchors_ready and hmd_T is not None:
                on_pose_detected(anchor_transforms, hmd_T)
                calibrated = True
                print("[aruco_detector] Calibration captured; stopping detection")

            if not calibrated:
                cv2.imshow("Webcam ArUco", detection_frame)
            if not calibrated and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[aruco_detector] Webcam terminated")
