"""Universal Robots UR5 planner defaults.

The platform's NumPy IK implementation is shared with the Franka module and
loads all arm geometry from the URDF passed in :class:`IKConfig`. This module
provides UR5-specific defaults while keeping the common robot-planner interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robotic_printing_platform.path_planning import PathPrep
from robotic_printing_platform.robots.generic import RobotTrajectory, URDFIKConfig, solve_urdf_path
from robotic_printing_platform.robots.base import RobotPlanner


DEFAULT_UR5_CONFIG_DIR = Path(__file__).resolve().parent / "robot_configs" / "ur5"
DEFAULT_UR5_URDF_PATH = DEFAULT_UR5_CONFIG_DIR / "robot.urdf"
DEFAULT_UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
UR5_JOINT_LIMITS = np.tile(np.array([[-2.0 * np.pi, 2.0 * np.pi]], dtype=float), (6, 1))
UR5_HOME = np.array([0.0, -np.pi / 2.0, 0.0, -np.pi / 2.0, 0.0, 0.0], dtype=float)


def make_ur5_ik_config(**overrides) -> URDFIKConfig:
    """Return a URDFIKConfig populated with UR5-specific defaults."""
    values = {
        "robot_model": "ur5",
        "urdf_path": str(DEFAULT_UR5_URDF_PATH),
        "base_link": "base_link",
        "end_link": "tool0",
        "joint_names": DEFAULT_UR5_JOINT_NAMES.copy(),
        "joint_limits": UR5_JOINT_LIMITS.copy(),
        "q_home": UR5_HOME.copy(),
        "max_reach_m": 0.85,
        "collision_skip_frames": 4,
    }
    values.update(overrides)
    return URDFIKConfig(**values)


@dataclass(frozen=True)
class UR5Planner(RobotPlanner):
    """URDF-backed planner for the original Universal Robots UR5."""

    config: URDFIKConfig | None = None

    def solve(self, path: PathPrep) -> RobotTrajectory:
        return solve_urdf_path(path, self.config or make_ur5_ik_config())
