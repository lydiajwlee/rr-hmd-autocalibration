import numpy as np
from scipy.spatial.transform import Rotation

# Fixed anchor world poses (Unity convention, meters)
ANCHOR_WORLD_POSES = {
    100: {
        "position": np.array([-0.9398, 0.8700, -1.397]),
        "rotation": Rotation.from_euler('y', 0, degrees=True).as_matrix()
    },
    101: {
        # Survey values: verify these measurements in the Unity room frame.
        "position": np.array([-0.9398, 0.8700, 0.0]),
        "rotation": Rotation.from_euler('y', 90, degrees=True).as_matrix()
    }
}

# OpenCV coordinates convention:
# X+ left, Y+ up (ceiling), Z+ forward (window)
# Unity coordinates convention:
# X+ right, Y+ up (ceiling), Z+ forward (window)

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

    rel_pos = rel_T[:3, 3]
    rel_rot = rel_T[:3, :3]

    anchor_world_pos = ANCHOR_WORLD_POSES[anchor_id]["position"]
    anchor_world_rot = ANCHOR_WORLD_POSES[anchor_id]["rotation"]

    # The camera-relative X axis runs opposite to the room's Unity X axis.
    rel_pos_unity = rel_pos.copy()
    rel_pos_unity[0] *= -1

    world_pos = anchor_world_pos + anchor_world_rot @ rel_pos_unity

    ROTATION_CONVERSION = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    rel_rot_unity = (
        ROTATION_CONVERSION
        @ rel_rot
        @ ROTATION_CONVERSION
    )

    world_rot = anchor_world_rot @ rel_rot_unity

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
