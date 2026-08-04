import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Fixed anchor world poses (Unity convention, meters)
ANCHOR_WORLD_POSES = {
    100: {
        # Mirrored across world X from marker 101; printed face points +Z.
        "position": np.array([0.9398, 2.4257, -2.4638]),
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
MARKER_TO_UNITY = np.eye(3)
UNITY_TO_CV_WORLD = np.diag([1.0, -1.0, 1.0])


def anchor_world_corners_cv(anchor_id, marker_size):
    """Return an anchor's TL/TR/BR/BL corners in a right-handed world frame."""
    pose = ANCHOR_WORLD_POSES[anchor_id]
    center = pose["position"]
    rotation = pose["rotation"]
    half = marker_size / 2.0

    # When the printed face points toward Unity +Z, its visually rightward
    # direction is Unity -X when viewed from in front of the marker.
    printed_x = rotation @ np.array([-1.0, 0.0, 0.0])
    printed_y = rotation @ np.array([0.0, 1.0, 0.0])
    offsets = (
        (-half, half),
        (half, half),
        (half, -half),
        (-half, -half),
    )
    corners_unity = np.array([
        center + x * printed_x + y * printed_y
        for x, y in offsets
    ])
    return (UNITY_TO_CV_WORLD @ corners_unity.T).T


def camera_extrinsics_from_anchors(
    anchor_image_corners, K, dist, marker_size
):
    """Jointly estimate world_cv_T_camera using all detected anchor corners."""
    if len(anchor_image_corners) < 2:
        raise ValueError("At least two anchors are required for joint extrinsics")

    unknown_ids = set(anchor_image_corners) - set(ANCHOR_WORLD_POSES)
    if unknown_ids:
        raise KeyError(f"Missing world poses for anchors: {sorted(unknown_ids)}")

    object_points = []
    image_points = []
    for anchor_id, corners in sorted(anchor_image_corners.items()):
        object_points.extend(anchor_world_corners_cv(anchor_id, marker_size))
        image_points.extend(np.asarray(corners).reshape(4, 2))

    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("Joint anchor extrinsics solve failed")

    camera_R_world_cv, _ = cv2.Rodrigues(rvec)
    camera_T_world_cv = np.eye(4)
    camera_T_world_cv[:3, :3] = camera_R_world_cv
    camera_T_world_cv[:3, 3] = tvec.ravel()
    world_cv_T_camera = np.linalg.inv(camera_T_world_cv)

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    residuals = projected.reshape(-1, 2) - image_points
    reprojection_error = np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))
    return world_cv_T_camera, float(reprojection_error)


def hmd_world_pose_from_anchor_extrinsics(
    anchor_image_corners, hmd_T, K, dist, anchor_marker_size
):
    """Compute the HMD world pose through one joint two-anchor camera solve."""
    world_cv_T_camera, reprojection_error = camera_extrinsics_from_anchors(
        anchor_image_corners, K, dist, anchor_marker_size
    )
    world_cv_T_hmd = world_cv_T_camera @ hmd_T

    world_position = UNITY_TO_CV_WORLD @ world_cv_T_hmd[:3, 3]
    world_rotation = (
        UNITY_TO_CV_WORLD
        @ world_cv_T_hmd[:3, :3]
        @ UNITY_TO_CV_WORLD
    )
    quaternion = Rotation.from_matrix(world_rotation).as_quat()
    return world_position, quaternion, reprojection_error

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

    # camera_T cancels in inv(camera_T_anchor) @ camera_T_hmd. For the current
    # physical installation, the printed marker axes align with the surveyed
    # Unity room axes.
    world_pos = (
        anchor_world_pos
        + anchor_world_rot @ MARKER_TO_UNITY @ rel_T[:3, 3]
    )
    world_rot = (
        anchor_world_rot
        @ MARKER_TO_UNITY
        @ rel_T[:3, :3]
        @ MARKER_TO_UNITY
    )

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
