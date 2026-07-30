import sys
import os
sys.path.append(os.path.dirname(__file__))

from aruco_detector import run, run_webcam, HMD_ID, ANCHOR_IDS
from pose_calculator import averaged_hmd_world_pose

# ── OSC (uncomment when sending to Unity) ──────────────────────────────────
from osc_sender import OSCSender
sender = OSCSender()
# ────────────────────────────────────────────────────────────────────────────

def on_pose_detected(anchor_transforms, hmd_T):

    world_pos, quat = averaged_hmd_world_pose(anchor_transforms, hmd_T)

    print(f"[main] anchors={ANCHOR_IDS} "
          f"pos=({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}) "
          f"quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")

    # ── OSC (uncomment when sending to Unity) ──────────────────────────────
    sender.send_pose(HMD_ID, world_pos, quat)
    # ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Pick ONE camera source ──────────────────────────────────────────────
    run_webcam(on_pose_detected, intrinsics_path="webcam_calibration/intrinsics.npz")   # webcam (no ZED)
    # run(on_pose_detected)        # ZED streaming
    # ────────────────────────────────────────────────────────────────────────
