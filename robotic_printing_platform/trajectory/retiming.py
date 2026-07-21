"""Time-parameterize IK trajectories against Cartesian and joint limits."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from robotic_printing_platform.robots.generic import RobotTrajectory


_EPS = 1e-12
_MIN_SEGMENT_DT_S = 1e-6


def _as_positive_limits(values: np.ndarray, n_joints: int, name: str) -> np.ndarray:
    limits = np.asarray(values, dtype=float)
    if limits.shape != (n_joints,):
        raise ValueError(f"{name} must have shape ({n_joints},), got {limits.shape}")
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
        raise ValueError(f"{name} must contain finite values greater than zero")
    return limits


def _segment_velocities(q: np.ndarray, durations_s: np.ndarray) -> np.ndarray:
    return np.diff(q, axis=0) / durations_s[:, None]


def _waypoint_accelerations(segment_velocities: np.ndarray, durations_s: np.ndarray) -> np.ndarray:
    """Approximate acceleration at each waypoint with zero endpoint velocity.

    Internal acceleration uses the centered difference between adjacent segment
    velocities.  The first and final terms are the acceleration/deceleration
    needed to enter and leave the path at rest.
    """
    n_segments, n_joints = segment_velocities.shape
    accelerations = np.zeros((n_segments + 1, n_joints), dtype=float)
    accelerations[0] = 2.0 * segment_velocities[0] / durations_s[0]
    if n_segments > 1:
        accelerations[1:-1] = 2.0 * (segment_velocities[1:] - segment_velocities[:-1]) / (
            durations_s[1:, None] + durations_s[:-1, None]
        )
    accelerations[-1] = -2.0 * segment_velocities[-1] / durations_s[-1]
    return accelerations


def _enforce_acceleration_limits(
    q: np.ndarray,
    durations_s: np.ndarray,
    acceleration_limits_rad_s2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use forward/backward duration passes to satisfy centered acceleration.

    Increasing both segments adjacent to an internal waypoint preserves their
    relative timing while lowering its centered acceleration.  Repeating a
    forward and backward pass also propagates the zero-velocity endpoint
    constraints through the path.
    """
    durations_s = durations_s.copy()
    n_segments = len(durations_s)
    for _ in range(80):
        changed = False
        for waypoint_indices in (range(n_segments + 1), range(n_segments, -1, -1)):
            for waypoint_index in waypoint_indices:
                segment_velocities = _segment_velocities(q, durations_s)
                acceleration = _waypoint_accelerations(segment_velocities, durations_s)[waypoint_index]
                scale = float(np.sqrt(np.max(np.abs(acceleration) / acceleration_limits_rad_s2)))
                if scale <= 1.0 + 1e-10:
                    continue
                if waypoint_index == 0:
                    affected = slice(0, 1)
                elif waypoint_index == n_segments:
                    affected = slice(n_segments - 1, n_segments)
                else:
                    affected = slice(waypoint_index - 1, waypoint_index + 1)
                durations_s[affected] *= scale
                changed = True
        segment_velocities = _segment_velocities(q, durations_s)
        accelerations = _waypoint_accelerations(segment_velocities, durations_s)
        if np.all(np.abs(accelerations) <= acceleration_limits_rad_s2 + 1e-9):
            return durations_s, accelerations
        if not changed:
            break
    raise RuntimeError("could not retime trajectory within joint acceleration limits")


def retime_trajectory(
    trajectory: RobotTrajectory,
    joint_velocity_limits: np.ndarray,
    joint_acceleration_limits: np.ndarray,
) -> RobotTrajectory:
    """Return a time-parameterized copy of ``trajectory``.

    Segment time starts with the G-code Cartesian duration ``distance / feed``
    and is enlarged for joint velocity limits.  A forward/backward pass then
    enforces centered waypoint acceleration while starting and ending at zero
    joint velocity.  The returned trajectory does not mutate the IK result.
    """
    if not trajectory.points:
        return replace(trajectory, points=[])

    q = np.asarray([point.q for point in trajectory.points], dtype=float)
    p = np.asarray([point.p for point in trajectory.points], dtype=float)
    if q.ndim != 2 or p.shape != (len(trajectory.points), 3):
        raise ValueError("trajectory points must contain consistent joint vectors and 3D positions")

    n_points, n_joints = q.shape
    velocity_limits = _as_positive_limits(joint_velocity_limits, n_joints, "joint_velocity_limits")
    acceleration_limits = _as_positive_limits(
        joint_acceleration_limits, n_joints, "joint_acceleration_limits"
    )
    if n_points == 1:
        only = trajectory.points[0]
        return replace(
            trajectory,
            points=[
                replace(
                    only,
                    q=only.q.copy(),
                    p=only.p.copy(),
                    time_from_start_s=0.0,
                    joint_velocity_rad_s=np.zeros(n_joints, dtype=float),
                    joint_acceleration_rad_s2=np.zeros(n_joints, dtype=float),
                )
            ],
            joint_velocity_limits_rad_s=velocity_limits.copy(),
            joint_acceleration_limits_rad_s2=acceleration_limits.copy(),
        )

    dq = np.diff(q, axis=0)
    distances_m = np.linalg.norm(np.diff(p, axis=0), axis=1)
    feeds_m_s = np.asarray([point.feed_m_s for point in trajectory.points[1:]], dtype=float)
    cartesian_durations_s = np.divide(
        distances_m,
        feeds_m_s,
        out=np.zeros_like(distances_m),
        where=(distances_m > _EPS) & (feeds_m_s > _EPS),
    )
    velocity_durations_s = np.max(np.abs(dq) / velocity_limits[None, :], axis=1)
    durations_s = np.maximum.reduce(
        [cartesian_durations_s, velocity_durations_s, np.full(n_points - 1, _MIN_SEGMENT_DT_S)]
    )
    durations_s, accelerations = _enforce_acceleration_limits(q, durations_s, acceleration_limits)
    segment_velocities = _segment_velocities(q, durations_s)
    if np.any(np.abs(segment_velocities) - velocity_limits > 1e-9):
        raise RuntimeError("retiming produced a joint-velocity limit violation")
    if np.any(np.abs(accelerations) - acceleration_limits > 1e-8):
        raise RuntimeError("retiming produced a joint-acceleration limit violation")

    times_s = np.concatenate(([0.0], np.cumsum(durations_s)))
    waypoint_velocities = np.vstack((np.zeros(n_joints, dtype=float), segment_velocities))
    waypoint_velocities[-1] = 0.0
    points = [
        replace(
            point,
            q=point.q.copy(),
            p=point.p.copy(),
            time_from_start_s=float(times_s[i]),
            joint_velocity_rad_s=waypoint_velocities[i].copy(),
            joint_acceleration_rad_s2=accelerations[i].copy(),
        )
        for i, point in enumerate(trajectory.points)
    ]
    return replace(
        trajectory,
        points=points,
        joint_velocity_limits_rad_s=velocity_limits.copy(),
        joint_acceleration_limits_rad_s2=acceleration_limits.copy(),
    )
