from __future__ import annotations

import unittest
from pathlib import Path

from robotic_printing_platform.config import load_planner_config
from robotic_printing_platform.robots.franka_panda import IKConfig, solve_path_ik
from robotic_printing_platform.robots.generic import URDFIKConfig, URDFRobotPlanner, solve_urdf_path
from robotic_printing_platform.robots.ur5 import make_ur5_ik_config


class GenericPlannerApiTests(unittest.TestCase):
    ROBOT_CONFIG_FACTORIES = {
        "franka_panda": IKConfig,
        "ur5": make_ur5_ik_config,
    }

    def test_all_robot_packages_use_the_generic_urdf_planner_api(self):
        for robot_model, make_config in self.ROBOT_CONFIG_FACTORIES.items():
            with self.subTest(robot=robot_model):
                config = make_config()

                self.assertIsInstance(config, URDFIKConfig)
                self.assertEqual(config.robot_model, robot_model)
                self.assertIsInstance(URDFRobotPlanner(config), URDFRobotPlanner)

    def test_legacy_panda_names_remain_compatible(self):
        self.assertIsInstance(IKConfig(), URDFIKConfig)
        self.assertNotEqual(solve_urdf_path.__module__, solve_path_ik.__module__)

    def test_ur5e_package_uses_ur5e_geometry_and_tighter_ik_tolerance(self):
        config = load_planner_config(
            "planner_config.json",
            robot_config_dir=Path(
                "robotic_printing_platform/robots/robot_configs/ur5e"
            ),
        )

        self.assertEqual(config.robot.model, "ur5e")
        self.assertAlmostEqual(config.make_ik_config().pos_tol_m, 0.002)
        urdf_path = Path(config.robot.urdf_path)
        self.assertEqual(urdf_path.name, "robot.urdf")
        self.assertEqual(urdf_path.parent.name, "ur5e")


if __name__ == "__main__":
    unittest.main()
