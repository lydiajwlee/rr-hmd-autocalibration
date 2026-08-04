import pathlib
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

from pose_calculator import (
    ANCHOR_WORLD_POSES,
    anchor_world_corners_cv,
    averaged_hmd_world_pose,
    hmd_world_pose_from_anchor_extrinsics,
    hmd_world_pose,
)


class AveragedHmdWorldPoseTests(unittest.TestCase):
    def test_joint_anchor_extrinsics_recovers_hmd_world_position(self):
        marker_size = 0.179
        K = np.array([
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0],
        ])
        dist = np.zeros(5)
        camera_rvec = np.zeros(3)
        camera_tvec = np.array([0.0, 0.0, 5.0])
        image_corners = {}
        for anchor_id in (100, 101):
            projected, _ = cv2.projectPoints(
                anchor_world_corners_cv(anchor_id, marker_size),
                camera_rvec,
                camera_tvec,
                K,
                dist,
            )
            image_corners[anchor_id] = projected.reshape(4, 2)

        expected_position = np.array([0.2, 1.0, -0.5])
        hmd_T = np.eye(4)
        hmd_T[:3, 3] = np.array([0.2, -1.0, 4.5])

        position, quaternion, error = hmd_world_pose_from_anchor_extrinsics(
            image_corners, hmd_T, K, dist, marker_size
        )

        np.testing.assert_allclose(position, expected_position, atol=1e-6)
        self.assertAlmostEqual(np.linalg.norm(quaternion), 1.0)
        self.assertLess(error, 1e-5)

    def test_anchor_relative_transform_recovers_world_origin(self):
        anchor_position = np.array([-0.9398, 2.4257, -2.4638])
        anchor_T = np.eye(4)
        hmd_T = np.eye(4)
        hmd_T[:3, 3] = -anchor_position
        synthetic_pose = {
            101: {"position": anchor_position, "rotation": np.eye(3)}
        }

        with patch.dict(ANCHOR_WORLD_POSES, synthetic_pose, clear=True):
            position, _ = hmd_world_pose(101, anchor_T, hmd_T)

        np.testing.assert_allclose(position, np.zeros(3), atol=1e-12)

    def test_identical_estimates_remain_unchanged(self):
        identity = np.eye(4)
        anchor_transforms = {100: identity, 101: identity}
        synthetic_poses = {
            anchor_id: {
                "position": np.array([1.0, 2.0, 3.0]),
                "rotation": np.eye(3),
            }
            for anchor_id in anchor_transforms
        }

        with patch.dict(ANCHOR_WORLD_POSES, synthetic_poses, clear=True):
            position, quaternion = averaged_hmd_world_pose(
                anchor_transforms, identity
            )

        np.testing.assert_allclose(position, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(np.linalg.norm(quaternion), 1.0)

    def test_rotation_average_uses_quaternion_mean(self):
        identity = np.eye(4)
        synthetic_poses = {
            100: {"position": np.zeros(3), "rotation": np.eye(3)},
            101: {
                "position": np.zeros(3),
                "rotation": Rotation.from_euler(
                    "y", 90, degrees=True
                ).as_matrix(),
            },
        }
        with patch.dict(ANCHOR_WORLD_POSES, synthetic_poses, clear=True):
            _, quaternion = averaged_hmd_world_pose(
                {100: identity, 101: identity}, identity
            )

        averaged_yaw = Rotation.from_quat(quaternion).as_euler(
            "xyz", degrees=True
        )[1]
        self.assertAlmostEqual(averaged_yaw, 45.0)


if __name__ == "__main__":
    unittest.main()
