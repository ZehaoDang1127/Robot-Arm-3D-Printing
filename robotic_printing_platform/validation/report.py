"""Kinematic and path-quality validation for retimed robot trajectories."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from robotic_printing_platform.robots.generic import RobotTrajectory, urdf_geometric_jacobian


@dataclass(frozen=True)
class ErrorStatistics:
    mean: float
    maximum: float
    p95: float


@dataclass(frozen=True)
class TrajectoryValidationReport:
    waypoints: int
    ik_success_rate: float
    ik_selection_mode: str
    yaw_discontinuity_count: int
    position_error_m: ErrorStatistics
    rotation_error_rad: ErrorStatistics
    total_joint_motion_rad: float
    maximum_joint_step_rad: float
    maximum_joint_velocity_rad_s: float
    maximum_joint_acceleration_rad_s2: float
    joint_velocity_violation_count: int
    joint_acceleration_violation_count: int
    minimum_joint_limit_margin_rad: float | None
    minimum_jacobian_manipulability: float | None
    minimum_jacobian_singular_value: float | None
    near_singular_waypoint_indices: list[int]
    collision_warnings: list[str]
    warnings: list[str]
    estimated_print_time_s: float
    extrusion_volume_mm3: float
    jacobian_sample_stride: int

    def summary(self) -> str:
        margin = "n/a" if self.minimum_joint_limit_margin_rad is None else f"{self.minimum_joint_limit_margin_rad:.6f} rad"
        singularity = (
            "n/a"
            if self.minimum_jacobian_singular_value is None
            else f"{self.minimum_jacobian_singular_value:.6g}"
        )
        manipulability = (
            "n/a"
            if self.minimum_jacobian_manipulability is None
            else f"{self.minimum_jacobian_manipulability:.6g}"
        )
        return (
            "Trajectory validation\n"
            f"waypoints                 : {self.waypoints}\n"
            f"IK success rate           : {self.ik_success_rate:.2%}\n"
            f"IK selection              : {self.ik_selection_mode} "
            f"({self.yaw_discontinuity_count} yaw discontinuities)\n"
            f"position error (m)        : mean={self.position_error_m.mean:.6g}, "
            f"max={self.position_error_m.maximum:.6g}, p95={self.position_error_m.p95:.6g}\n"
            f"rotation error (rad)      : mean={self.rotation_error_rad.mean:.6g}, "
            f"max={self.rotation_error_rad.maximum:.6g}, p95={self.rotation_error_rad.p95:.6g}\n"
            f"joint motion (rad)        : {self.total_joint_motion_rad:.6g}\n"
            f"max joint step (rad)      : {self.maximum_joint_step_rad:.6g}\n"
            f"max joint velocity (rad/s): {self.maximum_joint_velocity_rad_s:.6g}\n"
            f"max joint accel (rad/s^2) : {self.maximum_joint_acceleration_rad_s2:.6g}\n"
            f"velocity violations        : {self.joint_velocity_violation_count}\n"
            f"acceleration violations    : {self.joint_acceleration_violation_count}\n"
            f"min joint-limit margin    : {margin}\n"
            f"min Jacobian manipulability: {manipulability}\n"
            f"min Jacobian singular val.: {singularity}\n"
            f"collision warnings         : {len(self.collision_warnings)}\n"
            f"estimated print time (s)  : {self.estimated_print_time_s:.6g}\n"
            f"extrusion volume (mm^3)   : {self.extrusion_volume_mm3:.6g}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def _error_statistics(values: np.ndarray) -> ErrorStatistics:
    if values.size == 0:
        return ErrorStatistics(mean=0.0, maximum=0.0, p95=0.0)
    return ErrorStatistics(
        mean=float(np.mean(values)),
        maximum=float(np.max(values)),
        p95=float(np.percentile(values, 95)),
    )


def _trajectory_derivatives(trajectory: RobotTrajectory, field_name: str) -> np.ndarray:
    if not trajectory.points:
        return np.empty((0, 0), dtype=float)
    q = np.asarray([point.q for point in trajectory.points], dtype=float)
    rows = []
    for point in trajectory.points:
        values = np.asarray(getattr(point, field_name), dtype=float)
        if values.size == 0:
            values = np.zeros_like(point.q)
        if values.shape != point.q.shape:
            raise ValueError(f"{field_name} has shape {values.shape}, expected {point.q.shape}")
        rows.append(values)
    result = np.asarray(rows, dtype=float)
    if result.shape != q.shape:
        raise ValueError(f"{field_name} shape does not match trajectory joint vectors")
    return result


def _joint_limit_margin(trajectory: RobotTrajectory) -> float | None:
    if not trajectory.points:
        return None
    q = np.asarray([point.q for point in trajectory.points], dtype=float)
    limits = np.asarray(trajectory.config.joint_limits, dtype=float)
    if limits.shape != (q.shape[1], 2):
        raise ValueError(
            f"joint_limits has shape {limits.shape}, expected ({q.shape[1]}, 2)"
        )
    margins = np.minimum(q - limits[:, 0], limits[:, 1] - q)
    return float(np.min(margins))


def _derivative_violation_count(values: np.ndarray, limits: np.ndarray, name: str) -> int:
    if values.size == 0 or limits.size == 0:
        return 0
    if limits.shape != (values.shape[1],):
        raise ValueError(f"{name} has shape {limits.shape}, expected ({values.shape[1]},)")
    return int(np.count_nonzero(np.abs(values) > limits + 1e-9))


def _jacobian_quality_metrics(
    trajectory: RobotTrajectory,
    singularity_threshold: float,
    sample_stride: int,
) -> tuple[float | None, float | None, list[int]]:
    if not trajectory.points:
        return None, None, []
    minimum_manipulability = float("inf")
    minimum_singular_value = float("inf")
    near_singular = []
    cfg = trajectory.config
    for point in trajectory.points[::sample_stride]:
        if point.jacobian_manipulability is None or point.jacobian_min_singular_value is None:
            jacobian = urdf_geometric_jacobian(
                point.q,
                cfg.tool_length_m,
                cfg.tool_tcp_xyz_m,
                cfg.tool_tcp_rpy_rad,
                cfg.urdf_path,
                cfg.base_link,
                cfg.end_link,
            )
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            manipulability = float(np.prod(singular_values))
            sigma_min = float(np.min(singular_values))
        else:
            manipulability = point.jacobian_manipulability
            sigma_min = point.jacobian_min_singular_value
        minimum_manipulability = min(minimum_manipulability, manipulability)
        minimum_singular_value = min(minimum_singular_value, sigma_min)
        if sigma_min < singularity_threshold:
            near_singular.append(point.index)
    return minimum_manipulability, minimum_singular_value, near_singular


def validate_trajectory(
    trajectory: RobotTrajectory,
    *,
    singularity_threshold: float = 0.02,
    jacobian_sample_stride: int = 1,
) -> TrajectoryValidationReport:
    """Compute trajectory quality metrics for a retimed IK result.

    ``jacobian_sample_stride`` trades report resolution for speed on very large
    trajectories. A stride of one evaluates every exported waypoint.
    """
    if jacobian_sample_stride < 1:
        raise ValueError("jacobian_sample_stride must be at least one")

    points = trajectory.points
    q = np.asarray([point.q for point in points], dtype=float) if points else np.empty((0, 0))
    position_errors = np.asarray([point.pos_error_m for point in points], dtype=float)
    rotation_errors = np.asarray([point.rot_error_rad for point in points], dtype=float)
    velocities = _trajectory_derivatives(trajectory, "joint_velocity_rad_s")
    accelerations = _trajectory_derivatives(trajectory, "joint_acceleration_rad_s2")

    steps = np.diff(q, axis=0) if len(points) >= 2 else np.empty((0, q.shape[1]))
    maximum_step = float(np.max(np.abs(steps))) if steps.size else 0.0
    total_motion = float(np.linalg.norm(steps, axis=1).sum()) if steps.size else 0.0
    maximum_velocity = float(np.max(np.abs(velocities))) if velocities.size else 0.0
    maximum_acceleration = float(np.max(np.abs(accelerations))) if accelerations.size else 0.0
    velocity_violations = _derivative_violation_count(
        velocities,
        np.asarray(trajectory.joint_velocity_limits_rad_s, dtype=float),
        "joint_velocity_limits_rad_s",
    )
    acceleration_violations = _derivative_violation_count(
        accelerations,
        np.asarray(trajectory.joint_acceleration_limits_rad_s2, dtype=float),
        "joint_acceleration_limits_rad_s2",
    )
    minimum_manipulability, minimum_singular_value, near_singular = _jacobian_quality_metrics(
        trajectory, singularity_threshold, jacobian_sample_stride
    )
    collision_warnings = [
        warning
        for warning in trajectory.report.warnings
        if "collision" in warning.lower() or "clearance" in warning.lower()
    ]
    ik_success_rate = (
        float(trajectory.report.solved / trajectory.report.attempted)
        if trajectory.report.attempted
        else 0.0
    )
    estimated_time = float(points[-1].time_from_start_s) if points else 0.0
    extrusion_volume = float(sum(point.extrusion_volume_mm3 for point in points if point.is_print))

    return TrajectoryValidationReport(
        waypoints=len(points),
        ik_success_rate=ik_success_rate,
        ik_selection_mode=trajectory.report.selection_mode,
        yaw_discontinuity_count=trajectory.report.yaw_discontinuity_count,
        position_error_m=_error_statistics(position_errors),
        rotation_error_rad=_error_statistics(rotation_errors),
        total_joint_motion_rad=total_motion,
        maximum_joint_step_rad=maximum_step,
        maximum_joint_velocity_rad_s=maximum_velocity,
        maximum_joint_acceleration_rad_s2=maximum_acceleration,
        joint_velocity_violation_count=velocity_violations,
        joint_acceleration_violation_count=acceleration_violations,
        minimum_joint_limit_margin_rad=_joint_limit_margin(trajectory),
        minimum_jacobian_manipulability=minimum_manipulability,
        minimum_jacobian_singular_value=minimum_singular_value,
        near_singular_waypoint_indices=near_singular,
        collision_warnings=collision_warnings,
        warnings=list(trajectory.report.warnings),
        estimated_print_time_s=estimated_time,
        extrusion_volume_mm3=extrusion_volume,
        jacobian_sample_stride=jacobian_sample_stride,
    )
