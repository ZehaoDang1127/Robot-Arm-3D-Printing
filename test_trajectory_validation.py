from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from robotic_printing_platform.robots.franka_panda import IKConfig, IKReport, RobotTrajectory, TrajectoryPoint
from robotic_printing_platform.trajectory import retime_trajectory
from robotic_printing_platform.validation import validate_trajectory


def _point(index: int, q: np.ndarray, x: float, volume: float) -> TrajectoryPoint:
    return TrajectoryPoint(
        index=index,
        q=q,
        p=np.array([x, 0.0, 0.0], dtype=float),
        yaw=0.0,
        is_print=True,
        layer=0,
        seg_id=0,
        feed_m_s=1.0,
        de=1.0,
        material="PLA",
        extrusion_volume_mm3=volume,
        extrusion_mass_g=volume / 1000.0,
        pos_error_m=0.001 * (index + 1),
        rot_error_rad=0.01 * (index + 1),
    )


class TrajectoryValidationTests(unittest.TestCase):
    @patch("robotic_printing_platform.validation.report.urdf_geometric_jacobian")
    def test_collects_required_metrics(self, jacobian):
        jacobian.return_value = np.diag([3.0, 2.0, 1.0, 1.0, 1.0, 0.5])
        config = IKConfig(joint_names=[f"joint_{i}" for i in range(7)])
        q0 = np.array([0.0, -0.45, 0.0, -2.35, 0.0, 2.05, 0.75])
        q1 = q0 + 0.1
        q2 = q1 + 0.1
        trajectory = RobotTrajectory(
            points=[_point(0, q0, 0.0, 0.2), _point(1, q1, 0.1, 0.3), _point(2, q2, 0.2, 0.4)],
            report=IKReport(True, 4, 3, [7], ["waypoint 2: link sample below bed clearance"], 0.0, 0.0),
            config=config,
        )
        timed = retime_trajectory(trajectory, np.full(7, 2.0), np.full(7, 20.0))

        report = validate_trajectory(timed, singularity_threshold=0.6)

        self.assertEqual(report.waypoints, 3)
        self.assertEqual(report.ik_success_rate, 0.75)
        self.assertAlmostEqual(report.position_error_m.mean, 0.002)
        self.assertGreater(report.maximum_joint_step_rad, 0.0)
        self.assertGreater(report.maximum_joint_velocity_rad_s, 0.0)
        self.assertGreater(report.maximum_joint_acceleration_rad_s2, 0.0)
        self.assertEqual(report.joint_velocity_violation_count, 0)
        self.assertEqual(report.joint_acceleration_violation_count, 0)
        self.assertIsNotNone(report.minimum_joint_limit_margin_rad)
        self.assertEqual(report.minimum_jacobian_manipulability, 3.0)
        self.assertEqual(report.minimum_jacobian_singular_value, 0.5)
        self.assertEqual(report.near_singular_waypoint_indices, [0, 1, 2])
        self.assertEqual(len(report.collision_warnings), 1)
        self.assertAlmostEqual(report.estimated_print_time_s, timed.points[-1].time_from_start_s)
        self.assertAlmostEqual(report.extrusion_volume_mm3, 0.9)


if __name__ == "__main__":
    unittest.main()
