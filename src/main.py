import sys
import os
sys.path.append(os.path.dirname(__file__))

from aruco_detector import run, run_webcam, HMD_ID, ANCHOR_ID
from pose_calculator import hmd_world_pose

# ── OSC (uncomment when sending to Unity) ──────────────────────────────────
from osc_sender import OSCSender
sender = OSCSender()
# ────────────────────────────────────────────────────────────────────────────

def on_pose_detected(anchor_T, hmd_T):

    world_pos, quat = hmd_world_pose(ANCHOR_ID, anchor_T, hmd_T)

    print(f"[main] anchor={ANCHOR_ID} "
          f"pos=({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}) "
          f"quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")

    # ── OSC (uncomment when sending to Unity) ──────────────────────────────
    sender.send_pose(HMD_ID, world_pos, quat)
    # ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Pick ONE camera source ──────────────────────────────────────────────
    run_webcam(on_pose_detected)   # webcam (no ZED)
    # run(on_pose_detected)        # ZED streaming
    # ────────────────────────────────────────────────────────────────────────