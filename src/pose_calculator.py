import numpy as np
from scipy.spatial.transform import Rotation

# Fixed anchor world poses (Unity convention, meters)
ANCHOR_WORLD_POSES = {
    100: {
        "position": np.array([-0.9398, 0.8700, -1.397]),
        "rotation": Rotation.from_euler('y', 0, degrees=True).as_matrix()
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

    world_pos = anchor_world_pos + anchor_world_rot @ rel_pos

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