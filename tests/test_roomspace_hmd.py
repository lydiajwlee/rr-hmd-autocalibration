import pyzed.sl as sl
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import time
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
ARUCO_DICT  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECTOR    = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())

ANCHOR_ID   = 101   # Fixed anchor marker at -37, 34.25, 0 inches -> -0.9398, 0.8700, 0 meters
HMD_ID      = 0     # HMD marker

MARKER_SIZE = 0.1   # meters

CAMERA_IP   = "192.168.50.3"
CAMERA_PORT = 30000

# Fixed anchor world poses (Unity convention, meters)
ANCHOR_WORLD_POSES = {
    101: {
        "position": np.array([-0.9398, 0.8700, 0.0]),
        "rotation": Rotation.from_euler('y', 90, degrees=True).as_matrix()
    }
}
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

    # solve PnP
    success, rvec, tvec = cv2.solvePnP(
        obj_points, 
        img_points,
        K, 
        dist)

    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tvec.flatten()

    print(f"Rotation vector:\n{rvec.ravel()}")
    print(f"\nTranslation vector:\n{tvec.ravel()}")
    print(f"\nRotation matrix:\n{R}")

    return T

def relative_pose(anchor_T, hmd_T):
    """Compute HMD pose relative to anchor."""
    return np.linalg.inv(anchor_T) @ hmd_T

# ZED 연결
init_params = sl.InitParameters()
init_params.set_from_stream(CAMERA_IP, CAMERA_PORT)

zed = sl.Camera()
status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    print(f"ZED connection failed: {status}")
    exit()

K, dist = get_K_from_zed(zed)
print(f"ZED connected.\n")

image          = sl.Mat()
runtime_params = sl.RuntimeParameters()

print("Running — press SPACE to capture, Q to quit")

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
            cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)

            for i, marker_id in enumerate(ids.flatten()):
                T = get_pose(corners[i], K, dist)
                if T is None:
                    continue

                if marker_id == ANCHOR_ID:
                    anchor_T = T
                    rvec, _ = cv2.Rodrigues(T[:3, :3])
                    cv2.drawFrameAxes(frame_bgr, K, dist, rvec, T[:3, 3], 0.05)
                    cv2.putText(frame_bgr, f"Anchor (ID {ANCHOR_ID})",
                                corners[i][0][0].astype(int),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                elif marker_id == HMD_ID:
                    hmd_T = T
                    rvec, _ = cv2.Rodrigues(T[:3, :3])
                    cv2.drawFrameAxes(frame_bgr, K, dist, rvec, T[:3, 3], 0.05)
                    cv2.putText(frame_bgr, f"HMD (ID {HMD_ID})",
                                corners[i][0][0].astype(int),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (242, 255, 0), 2)

        if anchor_T is not None and hmd_T is not None:
            rel = relative_pose(anchor_T, hmd_T)
            rel_pos = rel[:3, 3]
            rel_rot = rel[:3, :3]

            # Apply anchor world pose
            anchor_world_pos = ANCHOR_WORLD_POSES[ANCHOR_ID]["position"]
            anchor_world_rot = ANCHOR_WORLD_POSES[ANCHOR_ID]["rotation"]

            # HMD world space position and rotation
            hmd_world_pos = anchor_world_pos + anchor_world_rot @ rel_pos
            hmd_world_rot = anchor_world_rot @ rel_rot

            # Convert to quaternion for Unity
            qx, qy, qz, qw = Rotation.from_matrix(hmd_world_rot).as_quat()

            pos_in = hmd_world_pos / 0.0254
            cv2.putText(frame_bgr,
                f"HMD world pos: ({pos_in[0]:.2f}, {pos_in[1]:.2f}, {pos_in[2]:.2f}) in",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            print(f"HMD world pos: ({hmd_world_pos[0]:.3f}, {hmd_world_pos[1]:.3f}, {hmd_world_pos[2]:.3f})")
            print(f"HMD world quat: ({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")

        elif anchor_T is None and hmd_T is None:
            cv2.putText(frame_bgr, "No markers detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif anchor_T is None:
            cv2.putText(frame_bgr, "Anchor not detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        elif hmd_T is None:
            cv2.putText(frame_bgr, "HMD not detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        display = cv2.resize(frame_bgr, (960, 540))
        cv2.imshow("Relative Pose Test", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            os.makedirs("tests/relative_pose/captures", exist_ok=True)
            filename = f"tests/relative_pose/captures/capture_{int(time.time())}.png"
            cv2.imwrite(filename, frame_bgr)
            print(f"Screenshot saved to {filename}")

finally:
    cv2.destroyAllWindows()
    zed.close()
    print("ZED terminated")