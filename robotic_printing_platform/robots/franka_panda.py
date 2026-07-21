"""Franka Panda defaults and backward-compatible planner exports.

The generic URDF solver, trajectory types, dynamic programming, and collision
integration live in :mod:`robotic_printing_platform.robots.generic`.  This
module owns only Panda-specific geometry, limits, and compatibility names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robotic_printing_platform.path_planning import PathPrep
from robotic_printing_platform.robots.generic import (
    IKReport,
    RobotTrajectory,
    TrajectoryPoint,
    URDFIKConfig,
    URDFRobotPlanner,
    _IKCandidate,
    _select_candidates_dp,
    jacobian_quality,
    solve_urdf_ik,
    solve_urdf_path,
    urdf_fk,
    urdf_geometric_jacobian,
    urdf_joint_frames,
)


PANDA_JOINT_LIMITS = np.array(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=float,
)
PANDA_HOME = np.array([0.0, -0.45, 0.0, -2.35, 0.0, 2.05, 0.75], dtype=float)
DEFAULT_ROBOT_CONFIG_DIR = Path(__file__).resolve().parent / "robot_configs" / "franka_panda"
DEFAULT_PANDA_URDF_PATH = DEFAULT_ROBOT_CONFIG_DIR / "robot.urdf"
DEFAULT_PANDA_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


@dataclass
class IKConfig(URDFIKConfig):
    """URDF configuration initialized with the platform's Panda defaults."""

    robot_model: str = "franka_panda"
    urdf_path: str = str(DEFAULT_PANDA_URDF_PATH)
    base_link: str = "panda_link0"
    end_link: str = "panda_link8"
    joint_names: list[str] = field(default_factory=lambda: DEFAULT_PANDA_JOINT_NAMES.copy())
    joint_limits: np.ndarray = field(default_factory=lambda: PANDA_JOINT_LIMITS.copy())
    q_home: np.ndarray = field(default_factory=lambda: PANDA_HOME.copy())
    isaac_usd_path: str = "/Isaac/Robots/Franka/franka.usd"


@dataclass(frozen=True)
class FrankaPandaPlanner(URDFRobotPlanner):
    """Panda adapter retaining the original planner entry point."""

    config: URDFIKConfig | None = None

    def solve(self, path: PathPrep) -> RobotTrajectory:
        return solve_urdf_path(path, self.config or IKConfig())


def panda_fk(
    q: np.ndarray,
    tool_length_m: float = 0.115,
    tool_tcp_xyz_m: tuple[float, float, float] | None = None,
    tool_tcp_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Compatibility wrapper for Panda forward kinematics defaults."""
    return urdf_fk(
        q, tool_length_m, tool_tcp_xyz_m, tool_tcp_rpy_rad,
        DEFAULT_PANDA_URDF_PATH, "panda_link0", "panda_link8",
    )


def panda_joint_frames(q: np.ndarray):
    """Compatibility wrapper for Panda joint-frame defaults."""
    return urdf_joint_frames(q, DEFAULT_PANDA_URDF_PATH, "panda_link0", "panda_link8")


def geometric_jacobian(
    q: np.ndarray,
    tool_length_m: float = 0.115,
    tool_tcp_xyz_m: tuple[float, float, float] | None = None,
    tool_tcp_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Compatibility wrapper for Panda Jacobian defaults."""
    return urdf_geometric_jacobian(
        q, tool_length_m, tool_tcp_xyz_m, tool_tcp_rpy_rad,
        DEFAULT_PANDA_URDF_PATH, "panda_link0", "panda_link8",
    )


solve_ik = solve_urdf_ik


def solve_path_ik(path: PathPrep, cfg: URDFIKConfig | None = None) -> RobotTrajectory:
    """Solve with Panda defaults when legacy callers omit a configuration."""
    return solve_urdf_path(path, cfg or IKConfig())


__all__ = [
    "DEFAULT_PANDA_JOINT_NAMES",
    "DEFAULT_PANDA_URDF_PATH",
    "DEFAULT_ROBOT_CONFIG_DIR",
    "FrankaPandaPlanner",
    "IKConfig",
    "IKReport",
    "PANDA_HOME",
    "PANDA_JOINT_LIMITS",
    "RobotTrajectory",
    "TrajectoryPoint",
    "URDFIKConfig",
    "_IKCandidate",
    "_select_candidates_dp",
    "geometric_jacobian",
    "jacobian_quality",
    "panda_fk",
    "panda_joint_frames",
    "solve_ik",
    "solve_path_ik",
]
