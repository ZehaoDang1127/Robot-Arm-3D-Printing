from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from robotic_printing_platform.config import load_planner_config
from robotic_printing_platform.robots.franka_panda import IKConfig, solve_path_ik
from robotic_printing_platform.robots.generic import (
    URDFIKConfig,
    URDFRobotPlanner,
    solve_urdf_path,
    urdf_fk,
)
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
        np.testing.assert_allclose(
            config.nozzle_tcp.flange_to_nozzle_xyz_m,
            np.array([-0.0284345792, 0.00774216, 0.146409252]),
            atol=1e-12,
        )
        urdf_path = Path(config.robot.urdf_path)
        self.assertEqual(urdf_path.name, "robot.urdf")
        self.assertEqual(urdf_path.parent.name, "ur5e")

        tool_pose, _ = urdf_fk(
            np.zeros(6),
            tool_tcp_xyz_m=(0.0, 0.0, 0.0),
            urdf_path=config.robot.urdf_path,
            base_link=config.robot.base_link,
            end_link=config.robot.end_link,
        )
        np.testing.assert_allclose(
            tool_pose[:3, 3],
            np.array([0.8172, 0.2329, 0.0628]),
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
