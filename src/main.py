import sys
import os
sys.path.append(os.path.dirname(__file__))

from aruco_detector import run, run_webcam, HMD_ID, ANCHOR_IDS
from pose_calculator import anchor_to_hmd_pose, averaged_hmd_world_pose

# ── OSC (uncomment when sending to Unity) ──────────────────────────────────
from osc_sender import OSCSender
sender = OSCSender()
calibration_sent = False
# ────────────────────────────────────────────────────────────────────────────

def on_pose_detected(anchor_transforms, hmd_T):
    global calibration_sent

    world_pos, quat = averaged_hmd_world_pose(anchor_transforms, hmd_T)
    anchor_id = ANCHOR_IDS[0]
    relative_pos = anchor_to_hmd_pose(
        anchor_transforms[anchor_id], hmd_T
    )[:3, 3]

    if not calibration_sent:
        print(f"[main] anchors={ANCHOR_IDS} "
              f"pos=({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}) "
              f"quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")

        # Send OSC only once while the camera overlay keeps updating.
        sender.send_pose(HMD_ID, world_pos, quat)
        calibration_sent = True
    # ───────────────────────────────────────────────────────────────────────

    return world_pos, quat, relative_pos

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
        stop_after_detection=False,  # live single-anchor pose display test
    )
    # run(on_pose_detected)        # ZED streaming
    # ────────────────────────────────────────────────────────────────────────
