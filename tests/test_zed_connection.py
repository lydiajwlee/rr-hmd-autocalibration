import pyzed.sl as sl
import cv2
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────────────
ARUCO_DICT  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECTOR    = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())
MARKER_ID   = 0       # Target marker ID
MARKER_SIZE = 0.1     # Marker size in meters

CAMERA_IP   = "192.168.50.3"
CAMERA_PORT = 30000
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
    dist = np.zeros(4, dtype=np.float64)  # ZED provides rectified images, so distortion is 0
    return K, dist

def get_marker_object_points(size):
    # 3D coordinates of marker corners in marker space
    half = size / 2.0
    return np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)

# Initialize ZED camera
init_params = sl.InitParameters()
init_params.set_from_stream(CAMERA_IP, CAMERA_PORT)

zed = sl.Camera()
status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    print(f"Failed to connect to ZED: {status}")
    exit()

K, dist = get_K_from_zed(zed)
print(f"K matrix:\n{K}")

image = sl.Mat()
runtime_params = sl.RuntimeParameters()
obj_points = get_marker_object_points(MARKER_SIZE)

print("ZED live — press Q to quit")

while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = DETECTOR.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)

            for j, marker_id in enumerate(ids.flatten()):
                if marker_id != MARKER_ID:
                    continue

                # Compute pose using solvePnP
                img_points = corners[j].reshape(4, 2)
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points, K, dist)

                if success:
                    # Draw axis and display position
                    cv2.drawFrameAxes(frame_bgr, K, dist, rvec, tvec, 0.05)
                    pos = tvec.flatten()
                    cv2.putText(frame_bgr,
                        f"ID {marker_id} pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame_bgr, "No markers detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("ZED ArUco Test", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
zed.close()