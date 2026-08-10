from __future__ import annotations

import unittest

import numpy as np

from robotic_printing_platform.robots.franka_panda import IKConfig, _IKCandidate, _select_candidates_dp


def _candidate(q: float, yaw: float, unary_cost: float) -> _IKCandidate:
    return _IKCandidate(
        q=np.array([q], dtype=float),
        yaw=yaw,
        success=True,
        pos_error_m=0.0,
        rot_error_rad=0.0,
        unary_cost=unary_cost,
    )


class GlobalIKOptimizationTests(unittest.TestCase):
    def test_dp_prefers_globally_smoother_yaw_sequence_over_local_unary_minimum(self):
        cfg = IKConfig(
            global_dp_motion_weight=1.0,
            global_dp_smoothness_weight=0.0,
        )
        candidates = [
            [_candidate(0.0, 0.0, 0.5), _candidate(1.0, 1.0, 0.0)],
            [_candidate(0.0, 0.0, 0.0), _candidate(1.0, 1.0, 0.0)],
        ]

        selected = _select_candidates_dp(candidates, np.array([0.0]), cfg)

        self.assertEqual([candidate.yaw for candidate in selected], [0.0, 0.0])
        self.assertLess(
            sum((selected[i].q[0] - selected[i - 1].q[0]) ** 2 for i in range(1, len(selected))),
            1.0,
        )

    def test_dp_penalizes_change_in_joint_delta(self):
        cfg = IKConfig(
            global_dp_motion_weight=0.0,
            global_dp_smoothness_weight=2.0,
        )
        candidates = [
            [_candidate(0.0, 0.0, 0.0)],
            [_candidate(1.0, 0.0, 0.0)],
            [_candidate(2.0, 0.0, 0.0), _candidate(1.0, 1.0, 0.0)],
        ]

        selected = _select_candidates_dp(candidates, np.array([-1.0]), cfg)

        self.assertEqual(selected[-1].q[0], 2.0)


if __name__ == "__main__":
    unittest.main()
