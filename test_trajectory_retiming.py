from __future__ import annotations

import unittest

import numpy as np

from robotic_printing_platform.robots.franka_panda import IKConfig, IKReport, RobotTrajectory, TrajectoryPoint
from robotic_printing_platform.trajectory import retime_trajectory


def _point(index: int, q: float, x: float, feed: float) -> TrajectoryPoint:
    return TrajectoryPoint(
        index=index,
        q=np.array([q], dtype=float),
        p=np.array([x, 0.0, 0.0], dtype=float),
        yaw=0.0,
        is_print=True,
        layer=0,
        seg_id=0,
        feed_m_s=feed,
        de=0.0,
        material="PLA",
        extrusion_volume_mm3=0.0,
        extrusion_mass_g=0.0,
        pos_error_m=0.0,
        rot_error_rad=0.0,
    )


def _trajectory(points: list[TrajectoryPoint]) -> RobotTrajectory:
    return RobotTrajectory(
        points=points,
        report=IKReport(True, len(points), len(points), [], [], 0.0, 0.0),
        config=IKConfig(joint_names=["joint_1"]),
    )


class RetimingTests(unittest.TestCase):
    def test_enforces_cartesian_and_joint_velocity_limits(self):
        trajectory = _trajectory([_point(0, 0.0, 0.0, 1.0), _point(1, 1.0, 0.1, 1.0), _point(2, 2.0, 0.2, 1.0)])

        timed = retime_trajectory(trajectory, np.array([2.0]), np.array([100.0]))

        self.assertAlmostEqual(timed.points[1].time_from_start_s, 0.5, places=9)
        self.assertAlmostEqual(timed.points[2].time_from_start_s, 1.0, places=9)
        self.assertLessEqual(abs(timed.points[1].joint_velocity_rad_s[0]), 2.0)
        self.assertLessEqual(abs(timed.points[2].joint_velocity_rad_s[0]), 2.0)
        self.assertEqual(trajectory.points[1].time_from_start_s, 0.0)

    def test_expands_duration_to_satisfy_acceleration_limit(self):
        trajectory = _trajectory([_point(0, 0.0, 0.0, 100.0), _point(1, 1.0, 0.0, 100.0), _point(2, 1.0, 0.0, 100.0)])

        timed = retime_trajectory(trajectory, np.array([100.0]), np.array([1.0]))

        self.assertGreaterEqual(timed.points[1].time_from_start_s, 1.0 - 1e-9)
        self.assertGreaterEqual(timed.points[2].time_from_start_s - timed.points[1].time_from_start_s, 1.0 - 1e-9)
        self.assertLessEqual(abs(timed.points[1].joint_acceleration_rad_s2[0]), 1.0 + 1e-9)
        self.assertLessEqual(abs(timed.points[2].joint_acceleration_rad_s2[0]), 1.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
