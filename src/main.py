import sys
import os
sys.path.append(os.path.dirname(__file__))

from aruco_detector import (
    ANCHOR_IDS,
    ANCHOR_MARKER_SIZE,
    HMD_ID,
    run,
    run_webcam,
)
from pose_calculator import hmd_world_pose_from_anchor_extrinsics

# ── OSC (uncomment when sending to Unity) ──────────────────────────────────
from osc_sender import OSCSender
sender = OSCSender()
calibration_sent = False
MAX_ANCHOR_REPROJECTION_ERROR = 2.0  # pixels
# ────────────────────────────────────────────────────────────────────────────

def on_pose_detected(anchor_image_corners, hmd_T, K, dist):
    global calibration_sent

    world_pos, quat, reprojection_error = (
        hmd_world_pose_from_anchor_extrinsics(
            anchor_image_corners,
            hmd_T,
            K,
            dist,
            ANCHOR_MARKER_SIZE,
        )
    )

    if (
        not calibration_sent
        and reprojection_error <= MAX_ANCHOR_REPROJECTION_ERROR
    ):
        print(f"[main] anchors={ANCHOR_IDS} "
              f"pos=({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}) "
              f"quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")

        # Send OSC only once while the camera overlay keeps updating.
        sender.send_pose(HMD_ID, world_pos, quat)
        calibration_sent = True
    # ───────────────────────────────────────────────────────────────────────

    return world_pos, quat, reprojection_error

if __name__ == "__main__":
    intrinsics_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "webcam_calibration",
        "intrinsics.npz",
    )
    # ── Pick ONE camera source ──────────────────────────────────────────────
    run_webcam(
        on_pose_detected,
        intrinsics_path=intrinsics_path,
        stop_after_detection=False,  # live joint-extrinsics display test
    )
    # run(on_pose_detected)        # ZED streaming
    # ────────────────────────────────────────────────────────────────────────
