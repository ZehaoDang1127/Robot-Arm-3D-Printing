from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_pipeline import run


class EndToEndSmokeTests(unittest.TestCase):
    ROBOT_MODELS = ("franka_panda", "ur5")

    def test_real_gcode_smoke_exports_validated_trajectories_for_all_robots(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, trajectories, bundles, _ = run(
                "strong_universal_wall_hook_vcd.gcode",
                lo=0,
                hi=1,
                robot="both",
                max_seg_len_mm=20,
                max_ik_waypoints=10,
                ik_selection_mode="greedy",
                output_dir=directory,
            )

            for robot_model in self.ROBOT_MODELS:
                with self.subTest(robot=robot_model):
                    trajectory = trajectories[robot_model]
                    validation_path = bundles[robot_model]["validation_report"]
                    csv_path = bundles[robot_model]["csv"]
                    trajectory_json_path = bundles[robot_model]["json"]
                    validation = json.loads(Path(validation_path).read_text(encoding="utf-8"))
                    trajectory_json = json.loads(Path(trajectory_json_path).read_text(encoding="utf-8"))

                    self.assertGreater(trajectory.report.generated, 0)
                    self.assertGreater(trajectory.report.successful, 0)
                    self.assertTrue(
                        all(
                            earlier.time_from_start_s <= later.time_from_start_s
                            for earlier, later in zip(trajectory.points, trajectory.points[1:])
                        )
                    )
                    self.assertEqual(validation["joint_velocity_violation_count"], 0)
                    self.assertEqual(validation["joint_acceleration_violation_count"], 0)
                    self.assertEqual(trajectory_json["report"]["generated"], trajectory.report.generated)
                    self.assertEqual(trajectory_json["report"]["successful"], trajectory.report.successful)
                    self.assertTrue(csv_path.is_file())
                    self.assertTrue(validation_path.is_file())


if __name__ == "__main__":
    unittest.main()
