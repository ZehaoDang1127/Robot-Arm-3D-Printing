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


def _acceleration_safe(
    dt_s: float,
    dq_rad: np.ndarray,
    previous_velocity_rad_s: np.ndarray,
    acceleration_limits_rad_s2: np.ndarray,
) -> bool:
    velocity = dq_rad / dt_s
    return bool(
        np.all(
            np.abs(velocity - previous_velocity_rad_s)
            <= acceleration_limits_rad_s2 * dt_s + 1e-12
        )
    )


def _duration_for_acceleration(
    lower_bound_s: float,
    dq_rad: np.ndarray,
    previous_velocity_rad_s: np.ndarray,
    acceleration_limits_rad_s2: np.ndarray,
) -> float:
    """Find the shortest duration satisfying all per-joint acceleration limits."""
    if _acceleration_safe(
        lower_bound_s, dq_rad, previous_velocity_rad_s, acceleration_limits_rad_s2
    ):
        return lower_bound_s

    upper_bound_s = lower_bound_s
    for _ in range(80):
        upper_bound_s *= 2.0
        if _acceleration_safe(
            upper_bound_s, dq_rad, previous_velocity_rad_s, acceleration_limits_rad_s2
        ):
            break
    else:
        raise RuntimeError("could not find a finite duration satisfying acceleration limits")

    lower = lower_bound_s
    upper = upper_bound_s
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        if _acceleration_safe(
            midpoint, dq_rad, previous_velocity_rad_s, acceleration_limits_rad_s2
        ):
            upper = midpoint
        else:
            lower = midpoint
    return upper


def retime_trajectory(
    trajectory: RobotTrajectory,
    joint_velocity_limits: np.ndarray,
    joint_acceleration_limits: np.ndarray,
) -> RobotTrajectory:
    """Return a time-parameterized copy of ``trajectory``.

    Segment time starts with the G-code Cartesian duration ``distance / feed``
    and is then enlarged to satisfy per-joint velocity and acceleration limits.
    The returned trajectory adds cumulative time, joint velocity, and joint
    acceleration to every point without mutating the IK result passed in.
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

    times_s = np.zeros(n_points, dtype=float)
    velocities = np.zeros_like(q)
    accelerations = np.zeros_like(q)

    for i in range(1, n_points):
        dq = q[i] - q[i - 1]
        distance_m = float(np.linalg.norm(p[i] - p[i - 1]))
        feed_m_s = float(trajectory.points[i].feed_m_s)
        cartesian_dt_s = distance_m / feed_m_s if distance_m > _EPS and feed_m_s > _EPS else 0.0
        velocity_dt_s = float(np.max(np.abs(dq) / velocity_limits))
        lower_bound_s = max(cartesian_dt_s, velocity_dt_s, _MIN_SEGMENT_DT_S)
        dt_s = _duration_for_acceleration(
            lower_bound_s, dq, velocities[i - 1], acceleration_limits
        )

        velocities[i] = dq / dt_s
        accelerations[i] = (velocities[i] - velocities[i - 1]) / dt_s
        times_s[i] = times_s[i - 1] + dt_s

    if np.any(np.abs(velocities) - velocity_limits > 1e-9):
        raise RuntimeError("retiming produced a joint-velocity limit violation")
    if np.any(np.abs(accelerations) - acceleration_limits > 1e-8):
        raise RuntimeError("retiming produced a joint-acceleration limit violation")

    points = [
        replace(
            point,
            q=point.q.copy(),
            p=point.p.copy(),
            time_from_start_s=float(times_s[i]),
            joint_velocity_rad_s=velocities[i].copy(),
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
