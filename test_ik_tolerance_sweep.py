from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from robotic_printing_platform.robots.franka_panda import IKConfig, IKReport, RobotTrajectory, TrajectoryPoint
from robotic_printing_platform.validation.tolerance_sweep import sweep_position_tolerances


def _trajectory(tolerance_m: float) -> RobotTrajectory:
    point = TrajectoryPoint(
        index=0,
        q=np.zeros(7),
        p=np.zeros(3),
        yaw=0.0,
        is_print=True,
        layer=0,
        seg_id=0,
        feed_m_s=0.01,
        de=0.0,
        material="PLA",
        extrusion_volume_mm3=0.0,
        extrusion_mass_g=0.0,
        pos_error_m=tolerance_m / 2.0,
        rot_error_rad=0.0,
        ik_iterations=12,
    )
    return RobotTrajectory(
        points=[point],
        report=IKReport(True, 2, 1, [1], [], 0.0, 0.0),
        config=IKConfig(),
    )


class IKToleranceSweepTests(unittest.TestCase):
    @patch("robotic_printing_platform.validation.tolerance_sweep.solve_urdf_path")
    def test_runs_each_requested_tolerance_and_reports_metrics(self, solve):
        solve.side_effect = lambda _path, cfg: _trajectory(cfg.pos_tol_m)

        report = sweep_position_tolerances(object(), IKConfig(), [8.0, 3.0, 1.0])

        self.assertEqual([entry.position_tolerance_m for entry in report.entries], [0.008, 0.003, 0.001])
        self.assertEqual([entry.ik_success_rate for entry in report.entries], [0.5, 0.5, 0.5])
        self.assertEqual([entry.average_ik_iterations for entry in report.entries], [12.0, 12.0, 12.0])
        self.assertEqual(solve.call_count, 3)


if __name__ == "__main__":
    unittest.main()
