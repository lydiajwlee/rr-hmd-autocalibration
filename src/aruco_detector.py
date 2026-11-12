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

ANCHOR_IDS  = (100, 101)  # Both wall markers define camera extrinsics
HMD_ID      = 0           # HMD marker
ANCHOR_MARKER_SIZE = 0.179  # 179 mm, in meters
HMD_MARKER_SIZE    = 0.032  # Measured outer black square on the headset, in meters
DETECTION_ZOOM = 1.0  # validate new 4K intrinsics before adding digital zoom

WEBCAM_WIDTH  = 3840
WEBCAM_HEIGHT = 2160

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

    Returns camera transforms, anchor image corners, HMD transform, and raw
    ArUco results. Anchor dictionaries can be incomplete while markers hide.
    """
    corners, ids, _ = DETECTOR.detectMarkers(gray)

    anchor_transforms = {}
    anchor_image_corners = {}
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
                anchor_image_corners[int(marker_id)] = corners[i].reshape(4, 2)
            elif marker_id == HMD_ID:
                hmd_T = T

    return anchor_transforms, anchor_image_corners, hmd_T, corners, ids

def draw_world_pose(frame, world_pose):
    """Draw an HMD world position and quaternion on a camera frame."""
    if world_pose is None:
        return

    position, quaternion = world_pose[:2]
    lines = [
        f"HMD world XYZ: {position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f} m",
        f"HMD world XYZW: {quaternion[0]:.3f}, {quaternion[1]:.3f}, "
        f"{quaternion[2]:.3f}, {quaternion[3]:.3f}",
    ]
    if len(world_pose) > 2:
        lines.append(f"Anchor reprojection error: {world_pose[2]:.3f} px")
    for index, line in enumerate(lines):
        cv2.putText(
            frame, line, (20, 35 + index * 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
        )

def run(on_pose_detected, stop_after_detection=True):
    """
    Detect the configured anchor(s) and HMD from the ZED stream.
    Stop after the first pose by default, or continue for a live test display.
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
    last_world_pose = None

    try:
        while not calibrated:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame     = image.get_data()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            detection_frame, detection_K = zoom_detection_frame(frame_bgr, K)
            gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)

            (
                anchor_transforms,
                anchor_image_corners,
                hmd_T,
                corners,
                ids,
            ) = detect_markers(
                gray, detection_K, dist
            )

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(detection_frame, corners, ids)

            anchors_ready = all(
                anchor_id in anchor_transforms for anchor_id in ANCHOR_IDS
            )
            if not calibrated and anchors_ready and hmd_T is not None:
                last_world_pose = on_pose_detected(
                    anchor_image_corners, hmd_T, detection_K, dist
                )
                pose_is_valid = (
                    last_world_pose is not None
                    and last_world_pose[2] <= 2.0
                )
                if stop_after_detection and pose_is_valid:
                    calibrated = True
                    print("[aruco_detector] Calibration captured; stopping detection")

            draw_world_pose(detection_frame, last_world_pose)

            if not calibrated:
                cv2.imshow("ZED ArUco", detection_frame)
            if not calibrated and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        zed.close()
        print("[aruco_detector] ZED terminated")


def run_webcam(
    on_pose_detected,
    camera_index=0,
    intrinsics_path=None,
    stop_after_detection=True,
):
    """Detect an anchor/HMD pose once, or continuously in live test mode."""
    requested_size = (WEBCAM_WIDTH, WEBCAM_HEIGHT)
    if intrinsics_path is not None:
        with np.load(intrinsics_path) as data:
            if 'image_size' not in data.files:
                raise RuntimeError(
                    "Intrinsics file has no resolution metadata and is from the "
                    "old calibration. Recapture and recalibrate before detection."
                )
            requested_size = tuple(int(value) for value in data['image_size'])
            K = data['K']
            dist = data['dist']
        print(f"[aruco_detector] Loaded K matrix from {intrinsics_path}")

    requested_width, requested_height = requested_size
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)

    if not cap.isOpened():
        print("[aruco_detector] Webcam connection failed")
        return

    if intrinsics_path is None:
        ret, frame = cap.read()
        if not ret:
            print("[aruco_detector] Failed to read frame")
            cap.release()
            return

        h, w = frame.shape[:2]
        K    = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(4, dtype=np.float64)
        print("[aruco_detector] Using approximate K matrix")

    actual_width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"[aruco_detector] Webcam connected at "
        f"{actual_width}x{actual_height} — press Q to quit"
    )
    if (actual_width, actual_height) != requested_size:
        if intrinsics_path is not None:
            cap.release()
            raise RuntimeError(
                f"Intrinsics are for {requested_width}x{requested_height}, "
                f"but webcam is {actual_width}x{actual_height}"
            )
        print(
            f"[aruco_detector] Requested {requested_width}x{requested_height}, "
            f"using camera-provided {actual_width}x{actual_height}"
        )
    calibrated = False
    last_world_pose = None

    try:
        while not calibrated:
            ret, frame = cap.read()
            if not ret:
                continue

            detection_frame, detection_K = zoom_detection_frame(frame, K)
            gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
            (
                anchor_transforms,
                anchor_image_corners,
                hmd_T,
                corners,
                ids,
            ) = detect_markers(
                gray, detection_K, dist
            )

            anchors_ready = all(
                anchor_id in anchor_transforms for anchor_id in ANCHOR_IDS
            )
            if not calibrated and anchors_ready and hmd_T is not None:
                last_world_pose = on_pose_detected(
                    anchor_image_corners, hmd_T, detection_K, dist
                )
                pose_is_valid = (
                    last_world_pose is not None
                    and last_world_pose[2] <= 2.0
                )
                if stop_after_detection and pose_is_valid:
                    calibrated = True
                    print("[aruco_detector] Calibration captured; stopping detection")

            if not calibrated:
                height, width = detection_frame.shape[:2]
                scale = min(1280 / width, 720 / height, 1.0)
                preview = detection_frame
                if scale < 1.0:
                    preview = cv2.resize(
                        detection_frame, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA,
                    )
                if ids is not None:
                    preview_corners = [corner * scale for corner in corners]
                    cv2.aruco.drawDetectedMarkers(preview, preview_corners, ids)
                draw_world_pose(preview, last_world_pose)
                cv2.imshow("Webcam ArUco", preview)
            if not calibrated and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[aruco_detector] Webcam terminated")
