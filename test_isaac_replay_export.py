from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotic_printing_platform.exporters.isaac import export_isaac_bundle
from robotic_printing_platform.robots.franka_panda import IKConfig, IKReport, RobotTrajectory, TrajectoryPoint


class IsaacReplayExportTests(unittest.TestCase):
    def test_generates_time_interpolated_position_target_replay(self):
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
            pos_error_m=0.0,
            rot_error_rad=0.0,
            time_from_start_s=0.0,
        )
        trajectory = RobotTrajectory(
            points=[point],
            report=IKReport(True, 1, 1, 1, [], [], 0.0, 0.0),
            config=IKConfig(),
        )

        with tempfile.TemporaryDirectory() as directory:
            bundle = export_isaac_bundle(trajectory, Path(directory))
            source = bundle["isaac_script"].read_text(encoding="utf-8")
            compile(source, str(bundle["isaac_script"]), "exec")

        self.assertIn("time_from_start_s", source)
        self.assertIn("interpolate_joint_target", source)
        self.assertIn("controller.apply_action(ArticulationAction", source)
        self.assertIn("joint_indices=joint_indices", source)
        self.assertIn("robot.get_dof_index(name)", source)
        self.assertIn("SETTLING_TIME_S", source)
        self.assertIn("TRACKING_PLOT_SAMPLE_STRIDE", source)
        self.assertIn("joint_tracking.svg", source)
        self.assertIn("maximum_tracking_error_rad", source)
        self.assertNotIn("robot.set_joint_positions", source)


if __name__ == "__main__":
    unittest.main()
