import sys
import os
import time
from collections import deque

import numpy as np
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
CONTINUOUS_OSC_TEST = True  # Move/rotate marker 0 to drive the Unity cube live
MAX_ANCHOR_REPROJECTION_ERROR = 5.0 if CONTINUOUS_OSC_TEST else 2.0  # pixels
recent_positions = deque(maxlen=3)
last_valid_time = None
# ────────────────────────────────────────────────────────────────────────────

def on_pose_detected(anchor_image_corners, hmd_T, K, dist, *, frame_time=None):
    global calibration_sent, last_valid_time

    world_pos, quat, reprojection_error = (
        hmd_world_pose_from_anchor_extrinsics(
            anchor_image_corners,
            hmd_T,
            K,
            dist,
            ANCHOR_MARKER_SIZE,
        )
    )

    frame_position = world_pos
    pose_is_valid = reprojection_error <= MAX_ANCHOR_REPROJECTION_ERROR
    should_send = CONTINUOUS_OSC_TEST or not calibration_sent

    if pose_is_valid and CONTINUOUS_OSC_TEST:
        now = time.monotonic()
        if last_valid_time is None or now - last_valid_time > 0.5:
            recent_positions.clear()
        recent_positions.append(world_pos)
        last_valid_time = now
        if len(recent_positions) == recent_positions.maxlen:
            world_pos = np.median(recent_positions, axis=0)

    status = "valid" if pose_is_valid else "rejected"
    print(
        f"[pose] {status} world_xyz_m=({world_pos[0]:.3f}, {world_pos[1]:.3f}, "
        f"{world_pos[2]:.3f}) "
        f"quat_xyzw=({quat[0]:.3f}, {quat[1]:.3f}, "
        f"{quat[2]:.3f}, {quat[3]:.3f}) "
        f"anchor_error={reprojection_error:.3f}px",
        flush=True,
    )

    if pose_is_valid and should_send:
        sender.send_pose(HMD_ID, world_pos, quat, frame_time, frame_position)

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
        timestamp_frames=True,
    )
    # run(on_pose_detected)        # ZED streaming
    # ────────────────────────────────────────────────────────────────────────
