from osc_sender import OSCSender

HMD_ID = 0
sender = OSCSender()

def on_pose_detected(rel_T):
    sender.send_pose(HMD_ID, rel_T)
    print(f"[main] Pose sent")

# ---- test with images (no cameras) ----
from aruco_detector import run_from_image
IMAGE_PATH = "test_image.jpg"
run_from_image(IMAGE_PATH, on_pose_detected)

# ---- zed streaming ----
# from aruco_detector import run
# run(on_pose_detected)