import pathlib
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

from pose_calculator import averaged_hmd_world_pose


class AveragedHmdWorldPoseTests(unittest.TestCase):
    def test_identical_estimates_remain_unchanged(self):
        identity = np.eye(4)
        anchor_transforms = {100: identity, 101: identity}

        position, quaternion = averaged_hmd_world_pose(
            anchor_transforms, identity
        )

        expected_position = np.array([-0.9398, 0.8700, -0.6985])
        np.testing.assert_allclose(position, expected_position)
        self.assertAlmostEqual(np.linalg.norm(quaternion), 1.0)

    def test_rotation_average_uses_quaternion_mean(self):
        identity = np.eye(4)
        _, quaternion = averaged_hmd_world_pose(
            {100: identity, 101: identity}, identity
        )

        averaged_yaw = Rotation.from_quat(quaternion).as_euler(
            "xyz", degrees=True
        )[1]
        self.assertAlmostEqual(averaged_yaw, 45.0)


if __name__ == "__main__":
    unittest.main()
