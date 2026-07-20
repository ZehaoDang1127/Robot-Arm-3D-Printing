"""Neutral public API for configurable serial-chain robot planning.

The planner is driven entirely by :class:`URDFIKConfig`: a robot adapter only
needs to provide a URDF path, link names, joint metadata, limits, and runtime
asset path. Panda-specific compatibility names remain in ``franka_panda`` for
existing users, while new integrations should import this module.
"""

from .franka_panda import (
    IKReport,
    RobotTrajectory,
    TrajectoryPoint,
    URDFIKConfig,
    URDFRobotPlanner,
    jacobian_quality,
    solve_urdf_ik,
    solve_urdf_path,
    urdf_fk,
    urdf_geometric_jacobian,
    urdf_joint_frames,
)

__all__ = [
    "IKReport",
    "RobotTrajectory",
    "TrajectoryPoint",
    "URDFIKConfig",
    "URDFRobotPlanner",
    "jacobian_quality",
    "solve_urdf_ik",
    "solve_urdf_path",
    "urdf_fk",
    "urdf_geometric_jacobian",
    "urdf_joint_frames",
]
