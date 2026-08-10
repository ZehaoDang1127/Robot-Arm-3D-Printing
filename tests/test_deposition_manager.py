from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np

from robotic_printing_platform.extrusion import (
    DepositionManager,
    FlowInterval,
    FlowSchedule,
    TcpPose,
)


class RecordingSink:
    def __init__(self) -> None:
        self.segments = []
        self.points = []
        self.reset_calls = []
        self.flush_count = 0

    def emit_segment(self, start_pose, end_pose, volume_mm3, duration_s):
        self.segments.append((start_pose, end_pose, volume_mm3, duration_s))

    def emit_point(self, pose, volume_mm3, duration_s):
        self.points.append((pose, volume_mm3, duration_s))

    def reset(self, clear_geometry=False):
        self.reset_calls.append(clear_geometry)
        if clear_geometry:
            self.segments.clear()
            self.points.clear()

    def flush(self):
        self.flush_count += 1


def pose(x: float, y: float = 0.0, z: float = 0.0) -> TcpPose:
    return TcpPose(np.array([x, y, z], dtype=float))


@dataclass
class SchedulePoint:
    time_from_start_s: float
    extrusion_volume_mm3: float
    is_print: bool


class TcpPoseTests(unittest.TestCase):
    def test_validates_copies_and_normalizes_pose_arrays(self):
        position = np.array([1.0, 2.0, 3.0])
        sample = TcpPose(position, np.array([0.0, 0.0, 0.0, 2.0]))
        position[0] = 99.0

        np.testing.assert_allclose(sample.position_m, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(sample.orientation_xyzw, [0.0, 0.0, 0.0, 1.0])
        self.assertFalse(sample.position_m.flags.writeable)

    def test_interpolation_uses_shortest_quaternion_representation(self):
        first = TcpPose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        same_rotation = TcpPose([2.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0])

        middle = first.interpolate(same_rotation, 0.5)

        np.testing.assert_allclose(middle.position_m, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(first.angular_distance_rad(middle), 0.0)

    def test_rejects_non_finite_pose(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            TcpPose([math.nan, 0.0, 0.0])


class FlowScheduleTests(unittest.TestCase):
    def test_builds_rates_from_incoming_trajectory_volume(self):
        schedule = FlowSchedule.from_trajectory_points(
            [
                SchedulePoint(0.0, 0.0, False),
                SchedulePoint(1.0, 0.0, False),
                SchedulePoint(3.0, 8.0, True),
                SchedulePoint(4.0, 3.0, True),
            ]
        )

        self.assertEqual([item.material_flow_mm3_s for item in schedule.intervals], [0.0, 4.0, 3.0])
        self.assertAlmostEqual(schedule.volume_between(0.0, 4.0), 11.0)

    def test_split_covers_gaps_and_exact_boundaries(self):
        schedule = FlowSchedule(
            [FlowInterval(1.0, 2.0, 3.0), FlowInterval(3.0, 4.0, 5.0)]
        )

        pieces = schedule.split(0.5, 3.5)

        self.assertEqual(
            [
                (item.start_time_s, item.end_time_s, item.material_flow_mm3_s)
                for item in pieces
            ],
            [
                (0.5, 1.0, 0.0),
                (1.0, 2.0, 3.0),
                (2.0, 3.0, 0.0),
                (3.0, 3.5, 5.0),
            ],
        )
        self.assertAlmostEqual(schedule.volume_between(0.5, 3.5), 5.5)

    def test_rejects_ambiguous_positive_volume_at_zero_duration(self):
        with self.assertRaisesRegex(ValueError, "positive segment duration"):
            FlowSchedule.from_trajectory_points(
                [
                    SchedulePoint(0.0, 0.0, False),
                    SchedulePoint(0.0, 1.0, True),
                ]
            )

    def test_rejects_positive_volume_on_non_print_point(self):
        with self.assertRaisesRegex(ValueError, "non-print"):
            FlowSchedule.from_trajectory_points(
                [
                    SchedulePoint(0.0, 0.0, False),
                    SchedulePoint(1.0, 1.0, False),
                ]
            )


class DepositionManagerTests(unittest.TestCase):
    def test_first_sample_anchors_then_flow_times_elapsed_time_emits_segment(self):
        sink = RecordingSink()
        manager = DepositionManager(sink, max_tcp_step_m=1.0)

        anchor = manager.update(pose(0.0), 10.0, 99.0)
        emitted = manager.update(pose(0.2), 10.5, 4.0)

        self.assertEqual(anchor.kind, "anchor")
        self.assertEqual(emitted.kind, "segment")
        self.assertAlmostEqual(emitted.commanded_volume_mm3, 2.0)
        self.assertAlmostEqual(sink.segments[0][2], 2.0)
        self.assertAlmostEqual(sink.segments[0][3], 0.5)
        self.assertAlmostEqual(manager.statistics.emitted_volume_mm3, 2.0)

    def test_inactive_travel_refreshes_anchor_and_prevents_a_bridge(self):
        sink = RecordingSink()
        manager = DepositionManager(sink, max_tcp_step_m=2.0)
        manager.update(pose(0.0), 0.0, 0.0)
        manager.update(pose(1.0), 1.0, 0.0)

        manager.update(pose(1.2), 2.0, 1.0)

        np.testing.assert_allclose(sink.segments[0][0].position_m, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(sink.segments[0][1].position_m, [1.2, 0.0, 0.0])

    def test_schedule_update_splits_a_physics_step_and_interpolates_tcp(self):
        sink = RecordingSink()
        manager = DepositionManager(sink, max_tcp_step_m=2.0)
        schedule = FlowSchedule(
            [FlowInterval(0.0, 0.5, 0.0), FlowInterval(0.5, 1.0, 4.0)]
        )
        manager.update_from_schedule(pose(0.0), 0.0, schedule)

        updates = manager.update_from_schedule(pose(1.0), 1.0, schedule)

        self.assertEqual([update.kind for update in updates], ["inactive", "segment"])
        self.assertAlmostEqual(sum(update.commanded_volume_mm3 for update in updates), 2.0)
        np.testing.assert_allclose(sink.segments[0][0].position_m, [0.5, 0.0, 0.0])
        np.testing.assert_allclose(sink.segments[0][1].position_m, [1.0, 0.0, 0.0])
        self.assertEqual(manager.statistics.observed_samples, 2)

    def test_discontinuity_skips_volume_without_drawing_and_reanchors(self):
        sink = RecordingSink()
        manager = DepositionManager(sink, max_tcp_step_m=0.25)
        manager.update(pose(0.0), 0.0, 0.0)

        skipped = manager.update(pose(1.0), 1.0, 3.0)
        emitted = manager.update(pose(1.1), 2.0, 2.0)

        self.assertEqual(skipped.kind, "discontinuity")
        self.assertEqual(skipped.reason, "tcp_position_jump")
        self.assertAlmostEqual(skipped.unrepresented_volume_mm3, 3.0)
        self.assertEqual(emitted.kind, "segment")
        self.assertEqual(len(sink.segments), 1)
        np.testing.assert_allclose(sink.segments[0][0].position_m, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(manager.statistics.unrepresented_volume_mm3, 3.0)

    def test_schedule_does_not_hide_a_measured_pose_jump(self):
        sink = RecordingSink()
        manager = DepositionManager(sink, max_tcp_step_m=0.4)
        schedule = FlowSchedule(
            [
                FlowInterval(0.0, 0.25, 1.0),
                FlowInterval(0.25, 0.5, 1.0),
                FlowInterval(0.5, 0.75, 1.0),
                FlowInterval(0.75, 1.0, 1.0),
            ]
        )
        manager.update_from_schedule(pose(0.0), 0.0, schedule)

        updates = manager.update_from_schedule(pose(1.0), 1.0, schedule)

        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].reason, "tcp_position_jump")
        self.assertAlmostEqual(updates[0].unrepresented_volume_mm3, 1.0)
        self.assertFalse(sink.segments)

    def test_stationary_positive_flow_emits_a_point(self):
        sink = RecordingSink()
        manager = DepositionManager(sink)
        manager.update(pose(0.0), 0.0, 0.0)

        result = manager.update(pose(0.0), 2.0, 1.5)

        self.assertEqual(result.kind, "point")
        self.assertAlmostEqual(sink.points[0][1], 3.0)
        self.assertAlmostEqual(manager.statistics.emitted_volume_mm3, 3.0)

    def test_time_regression_breaks_continuity_without_clearing_geometry(self):
        sink = RecordingSink()
        manager = DepositionManager(sink, max_tcp_step_m=1.0)
        manager.update(pose(0.0), 2.0, 0.0)

        reset_boundary = manager.update(pose(0.5), 1.0, 10.0)
        manager.update(pose(0.6), 2.0, 1.0)

        self.assertEqual(reset_boundary.reason, "time_regression")
        self.assertEqual(manager.statistics.time_resets, 1)
        self.assertEqual(len(sink.segments), 1)
        np.testing.assert_allclose(sink.segments[0][0].position_m, [0.5, 0.0, 0.0])

    def test_disabled_flow_is_accounted_as_unrepresented(self):
        sink = RecordingSink()
        manager = DepositionManager(sink, max_tcp_step_m=1.0)
        manager.update(pose(0.0), 0.0, 0.0)

        result = manager.update(pose(0.1), 2.0, 3.0, enabled=False)

        self.assertEqual(result.kind, "skipped")
        self.assertAlmostEqual(result.unrepresented_volume_mm3, 6.0)
        self.assertFalse(sink.segments)

    def test_reset_controls_geometry_and_close_flushes_once(self):
        sink = RecordingSink()
        manager = DepositionManager(sink)
        manager.update(pose(0.0), 0.0, 0.0)
        manager.update(pose(0.0), 1.0, 1.0)

        manager.reset(clear_geometry=True, pose=pose(2.0), simulation_time_s=5.0)
        manager.close()
        manager.close()

        self.assertEqual(sink.reset_calls, [True])
        self.assertFalse(sink.points)
        self.assertEqual(manager.statistics.commanded_volume_mm3, 0.0)
        self.assertEqual(sink.flush_count, 1)

    def test_negative_flow_is_rejected(self):
        manager = DepositionManager(RecordingSink())
        with self.assertRaisesRegex(ValueError, "non-negative"):
            manager.update(pose(0.0), 0.0, -1.0)


if __name__ == "__main__":
    unittest.main()
