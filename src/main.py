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
CONTINUOUS_OSC_TEST = True  # Move/rotate marker 0 to drive the Unity rig live
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

    pose_is_valid = reprojection_error <= MAX_ANCHOR_REPROJECTION_ERROR
    should_send = CONTINUOUS_OSC_TEST or not calibration_sent

    if pose_is_valid:
        print(
            f"[pose] pos=({world_pos[0]:.3f}, {world_pos[1]:.3f}, "
            f"{world_pos[2]:.3f}) "
            f"quat=({quat[0]:.3f}, {quat[1]:.3f}, "
            f"{quat[2]:.3f}, {quat[3]:.3f}) "
            f"anchor_error={reprojection_error:.3f}px",
            flush=True,
        )

    if pose_is_valid and should_send:
        sender.send_pose(HMD_ID, world_pos, quat)

        if not calibration_sent:
            mode = "continuous test" if CONTINUOUS_OSC_TEST else "one shot"
            print(f"[main] OSC started ({mode}) anchors={ANCHOR_IDS}")

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
