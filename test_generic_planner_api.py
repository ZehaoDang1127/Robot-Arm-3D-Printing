from __future__ import annotations

import unittest

from robotic_printing_platform.robots.franka_panda import IKConfig, solve_path_ik
from robotic_printing_platform.robots.generic import URDFIKConfig, URDFRobotPlanner, solve_urdf_path
from robotic_printing_platform.robots.ur5 import make_ur5_ik_config


class GenericPlannerApiTests(unittest.TestCase):
    def test_ur5_uses_the_neutral_urdf_configuration_and_planner_api(self):
        config = make_ur5_ik_config()

        self.assertIsInstance(config, URDFIKConfig)
        self.assertEqual(config.robot_model, "ur5")
        self.assertIsInstance(URDFRobotPlanner(config), URDFRobotPlanner)

    def test_legacy_panda_names_remain_compatible(self):
        self.assertIsInstance(IKConfig(), URDFIKConfig)
        self.assertNotEqual(solve_urdf_path.__module__, solve_path_ik.__module__)


if __name__ == "__main__":
    unittest.main()
