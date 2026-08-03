import numpy as np
from scipy.spatial.transform import Rotation

# Fixed anchor world poses (Unity convention, meters)
ANCHOR_WORLD_POSES = {
    100: {
        "position": np.array([-0.9398, 0.8700, -1.397]),
        "rotation": Rotation.from_euler('y', 0, degrees=True).as_matrix()
    },
    101: {
        # Surveyed in inches as (-37, 95.5, -97), converted to meters.
        # The marker faces the window (Unity +Z), so its yaw is 0 degrees.
        "position": np.array([-0.9398, 2.4257, -2.4638]),
        "rotation": Rotation.from_euler('y', 0, degrees=True).as_matrix()
    }
}

# Marker-local convention established by get_marker_object_points():
# X+ toward the marker's right edge, Y+ toward its top edge, Z+ out of its face.
# ANCHOR_WORLD_POSES rotations map that marker-local frame into the Unity room.

def anchor_to_hmd_pose(anchor_T, hmd_T):
    """
    Compute HMD pose relative to anchor marker.
    anchor_T: 4x4 transform matrix of anchor marker (ID 100+) in camera space
    hmd_T:    4x4 transform matrix of HMD marker (ID 0-99) in camera space
    returns:  4x4 transform matrix of HMD pose in anchor space
    """
    return np.linalg.inv(anchor_T) @ hmd_T

def hmd_world_pose(anchor_id, anchor_T, hmd_T):
    """
    Compute HMD pose in room/world space.
    anchor_id: ID of the detected anchor marker
    anchor_T:  4x4 matrix of anchor in camera space
    hmd_T:     4x4 matrix of HMD in camera space
    returns:   (position np.array[3], quaternion np.array[4] as x,y,z,w)
    """
    rel_T   = np.linalg.inv(anchor_T) @ hmd_T

    anchor_world_pos = ANCHOR_WORLD_POSES[anchor_id]["position"]
    anchor_world_rot = ANCHOR_WORLD_POSES[anchor_id]["rotation"]

    # camera_T cancels in inv(camera_T_anchor) @ camera_T_hmd. The result is
    # already expressed in the anchor marker's local axes, so only the
    # surveyed anchor-to-world transform is needed; no camera-axis flip belongs
    # here.
    world_pos = anchor_world_pos + anchor_world_rot @ rel_T[:3, 3]
    world_rot = anchor_world_rot @ rel_T[:3, :3]

    quat = Rotation.from_matrix(world_rot).as_quat()  # (x, y, z, w)

    return world_pos, quat


def averaged_hmd_world_pose(anchor_transforms, hmd_T):
    """Fuse independent HMD world-pose estimates from all supplied anchors.

    Positions use an arithmetic mean. Rotations use SciPy's quaternion-aware
    rotation mean, which handles the q/-q equivalence correctly.
    """
    if not anchor_transforms:
        raise ValueError("At least one anchor transform is required")

    unknown_ids = set(anchor_transforms) - set(ANCHOR_WORLD_POSES)
    if unknown_ids:
        raise KeyError(f"Missing world poses for anchors: {sorted(unknown_ids)}")

    estimates = [
        hmd_world_pose(anchor_id, anchor_T, hmd_T)
        for anchor_id, anchor_T in sorted(anchor_transforms.items())
    ]
    positions = np.stack([position for position, _ in estimates])
    quaternions = np.stack([quaternion for _, quaternion in estimates])

    averaged_position = positions.mean(axis=0)
    averaged_quaternion = Rotation.from_quat(quaternions).mean().as_quat()
    return averaged_position, averaged_quaternion
