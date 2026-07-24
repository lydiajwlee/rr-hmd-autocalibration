import cv2
import numpy as np
import os
import glob


# ── CONFIG ──────────────────────────────────────────────────────────────────
IMAGES_DIR = "webcam_calibration/images"
SAVE_PATH  = "webcam_calibration/intrinsics.npz"


CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
SQUARE_LENGTH     = 0.04
MARKER_LENGTH     = 0.03
ARUCO_DICT        = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# ────────────────────────────────────────────────────────────────────────────


board    = cv2.aruco.CharucoBoard(
    (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
    SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT
)
detector = cv2.aruco.CharucoDetector(board)


all_obj_points = []
all_img_points = []
img_size       = None


for path in glob.glob(os.path.join(IMAGES_DIR, "*.jpg")):
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]


    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)


    if charuco_ids is not None and len(charuco_ids) >= 4:
        obj_points = board.getChessboardCorners()[charuco_ids.flatten()]
        all_obj_points.append(obj_points.astype(np.float32))
        all_img_points.append(charuco_corners.astype(np.float32))
        print(f"  ✓ {path} — {len(charuco_ids)} corners found")
    else:
        print(f"  ✗ {path} — not enough corners, skipping")


if not all_obj_points:
    print("No valid images found.")
    exit()


ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    all_obj_points, all_img_points, img_size, None, None
)


print(f"\nReprojection error: {ret:.4f} (aim for < 1.0)")
print("K matrix:\n", K)


np.savez(SAVE_PATH, K=K, dist=dist)
print(f"\nSaved to {SAVE_PATH}")


