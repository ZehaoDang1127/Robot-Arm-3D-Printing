"""Generic URDF IK, trajectory, and collision-aware planning primitives."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from robotic_printing_platform.path_planning import PathPrep
from robotic_printing_platform.robots.base import RobotPlanner
from robotic_printing_platform.robots.urdf_kinematics import URDFKinematicChain, load_urdf_chain
from robotic_printing_platform.validation.collision import (
    AxisAlignedBox,
    LinkCapsule,
    box_from_point,
    collision_warnings as capsule_collision_warnings,
    merge_boxes,
)


@dataclass
class URDFIKConfig:
    robot_model: str = "generic_urdf"
    urdf_path: str = ""
    base_link: str = ""
    end_link: str = ""
    joint_names: list[str] = field(default_factory=list)
    tool_length_m: float = 0.115
    tool_tcp_xyz_m: tuple[float, float, float] | None = None
    tool_tcp_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # The bundled NumPy-only model is intended for planning/simulation export.
    # Keep these tolerances visible in output files; tighten them after
    # calibrating the actual nozzle TCP and robot model in Isaac/Pinocchio.
    pos_tol_m: float = 0.008
    rot_tol_rad: float = 0.08
    max_iters: int = 180
    damping: float = 0.035
    orientation_weight: float = 0.35
    nullspace_weight: float = 0.015
    max_joint_step_rad: float = 0.10
    yaw_samples: int = 5
    ik_selection_mode: str = "greedy"  # "greedy" or "global_dp"
    global_dp_motion_weight: float = 10.0
    global_dp_smoothness_weight: float = 0.15
    global_dp_ik_error_weight: float = 25.0
    global_dp_singularity_weight: float = 0.01
    global_dp_collision_penalty: float = 0.0
    yaw_discontinuity_threshold_rad: float = math.pi / 2.0
    joint_limits: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    q_home: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    bed_z_m: float = 0.10
    bed_center_xy_m: tuple[float, float] = (0.45, 0.0)
    bed_half_extents_xy_m: tuple[float, float] = (0.15, 0.15)
    bed_thickness_m: float = 0.02
    min_clearance_m: float = 0.006
    link_capsule_radius_m: float = 0.04
    tool_capsule_radius_m: float = 0.015
    printed_volume_padding_m: float = 0.001
    nozzle_print_clearance_m: float = 0.0005
    max_reach_m: float = 0.855
    isaac_usd_path: str = ""
    collision_skip_frames: int = 2
    ik_stride: int = 1
    max_waypoints: int | None = None


@dataclass
class TrajectoryPoint:
    index: int
    q: np.ndarray
    p: np.ndarray
    yaw: float
    is_print: bool
    layer: int
    seg_id: int
    feed_m_s: float
    de: float
    material: str
    extrusion_volume_mm3: float
    extrusion_mass_g: float | None
    pos_error_m: float
    rot_error_rad: float
    time_from_start_s: float = 0.0
    joint_velocity_rad_s: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    joint_acceleration_rad_s2: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    ik_iterations: int = 0
    jacobian_manipulability: float | None = None
    jacobian_min_singular_value: float | None = None


def _derivative_for_export(values: np.ndarray, q: np.ndarray, name: str) -> np.ndarray:
    """Keep pre-retiming exports compatible while rejecting malformed vectors."""
    derivative = np.asarray(values, dtype=float)
    if derivative.size == 0:
        return np.zeros_like(q)
    if derivative.shape != q.shape:
        raise ValueError(f"{name} has shape {derivative.shape}, expected {q.shape}")
    return derivative


@dataclass
class IKReport:
    success: bool
    attempted: int
    generated: int
    successful: int
    failed_indices: list[int]
    warnings: list[str]
    total_joint_motion_rad: float
    estimated_cartesian_length_m: float
    selection_mode: str = "greedy"
    yaw_discontinuity_count: int = 0

    def summary(self) -> str:
        state = "success" if self.success else "incomplete"
        return (
            f"IK {state}: {self.successful}/{self.attempted} waypoints within tolerance "
            f"({self.generated} trajectory points generated)\n"
            f"outside-tolerance indices: {self.failed_indices[:12]}"
            f"{' ...' if len(self.failed_indices) > 12 else ''}\n"
            f"joint motion : {self.total_joint_motion_rad:.3f} rad\n"
            f"path length   : {self.estimated_cartesian_length_m:.3f} m\n"
            f"IK selection  : {self.selection_mode} "
            f"({self.yaw_discontinuity_count} yaw discontinuities)\n"
            f"warnings     : {len(self.warnings)}"
        )

    @property
    def solved(self) -> int:
        """Backward-compatible alias for the number of generated trajectory points."""
        return self.generated


@dataclass
class RobotTrajectory:
    points: list[TrajectoryPoint]
    report: IKReport
    config: URDFIKConfig
    joint_velocity_limits_rad_s: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    joint_acceleration_limits_rad_s2: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    timing_is_preview: bool = False

    def q_array(self) -> np.ndarray:
        return np.array([pt.q for pt in self.points], dtype=float)

    def export_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            joint_names = self.config.joint_names or [f"q{i + 1}" for i in range(len(self.points[0].q) if self.points else 0)]
            writer.writerow(
                [
                    "index",
                    "x_m",
                    "y_m",
                    "z_m",
                    "yaw_rad",
                    "is_print",
                    "layer",
                    "seg_id",
                    "feed_m_s",
                    "de",
                    "material",
                    "extrusion_volume_mm3",
                    "extrusion_mass_g",
                    "pos_error_m",
                    "rot_error_rad",
                    "ik_iterations",
                    "jacobian_manipulability",
                    "jacobian_min_singular_value",
                    "time_from_start_s",
                    *joint_names,
                    *[f"{name}_velocity_rad_s" for name in joint_names],
                    *[f"{name}_acceleration_rad_s2" for name in joint_names],
                ]
            )
            for pt in self.points:
                velocity = _derivative_for_export(
                    pt.joint_velocity_rad_s, pt.q, "joint_velocity_rad_s"
                )
                acceleration = _derivative_for_export(
                    pt.joint_acceleration_rad_s2, pt.q, "joint_acceleration_rad_s2"
                )
                writer.writerow(
                    [
                        pt.index,
                        *pt.p.tolist(),
                        pt.yaw,
                        int(pt.is_print),
                        pt.layer,
                        pt.seg_id,
                        pt.feed_m_s,
                        pt.de,
                        pt.material,
                        pt.extrusion_volume_mm3,
                        "" if pt.extrusion_mass_g is None else pt.extrusion_mass_g,
                        pt.pos_error_m,
                        pt.rot_error_rad,
                        pt.ik_iterations,
                        "" if pt.jacobian_manipulability is None else pt.jacobian_manipulability,
                        "" if pt.jacobian_min_singular_value is None else pt.jacobian_min_singular_value,
                        pt.time_from_start_s,
                        *pt.q.tolist(),
                        *velocity.tolist(),
                        *acceleration.tolist(),
                    ]
                )

    def export_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "joint_names": self.config.joint_names,
            "success": self.report.success,
            "report": {
                "attempted": self.report.attempted,
                "generated": self.report.generated,
                "successful": self.report.successful,
                "failed_indices": self.report.failed_indices,
                "warnings": self.report.warnings,
                "total_joint_motion_rad": self.report.total_joint_motion_rad,
                "estimated_cartesian_length_m": self.report.estimated_cartesian_length_m,
                "selection_mode": self.report.selection_mode,
                "yaw_discontinuity_count": self.report.yaw_discontinuity_count,
            },
            "points": [
                {
                    "index": pt.index,
                    "q": pt.q.tolist(),
                    "position_m": pt.p.tolist(),
                    "yaw_rad": pt.yaw,
                    "is_print": pt.is_print,
                    "layer": pt.layer,
                    "seg_id": pt.seg_id,
                    "feed_m_s": pt.feed_m_s,
                    "de": pt.de,
                    "material": pt.material,
                    "extrusion_volume_mm3": pt.extrusion_volume_mm3,
                    "extrusion_mass_g": pt.extrusion_mass_g,
                    "pos_error_m": pt.pos_error_m,
                    "rot_error_rad": pt.rot_error_rad,
                    "ik_iterations": pt.ik_iterations,
                    "jacobian_manipulability": pt.jacobian_manipulability,
                    "jacobian_min_singular_value": pt.jacobian_min_singular_value,
                    "time_from_start_s": pt.time_from_start_s,
                    "joint_velocity_rad_s": _derivative_for_export(
                        pt.joint_velocity_rad_s, pt.q, "joint_velocity_rad_s"
                    ).tolist(),
                    "joint_acceleration_rad_s2": _derivative_for_export(
                        pt.joint_acceleration_rad_s2, pt.q, "joint_acceleration_rad_s2"
                    ).tolist(),
                }
                for pt in self.points
            ],
        }
        path.write_text(json.dumps(payload, indent=2))


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return Rz @ Ry @ Rx


def _tool_transform(
    tool_length_m: float,
    tool_tcp_xyz_m: tuple[float, float, float] | None = None,
    tool_tcp_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Nozzle TCP transform from final robot link frame."""
    xyz = tool_tcp_xyz_m if tool_tcp_xyz_m is not None else (0.0, 0.0, tool_length_m)
    T = np.eye(4)
    T[:3, :3] = _rpy_matrix(*tool_tcp_rpy_rad)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


@lru_cache(maxsize=16)
def _load_cached_chain(urdf_path: str, base_link: str, end_link: str) -> URDFKinematicChain:
    return load_urdf_chain(urdf_path, base_link=base_link, end_link=end_link)


def _urdf_chain(
    urdf_path: str | Path | None = None,
    base_link: str = "",
    end_link: str = "",
) -> URDFKinematicChain:
    if not urdf_path or not base_link or not end_link:
        raise ValueError("generic URDF planning requires urdf_path, base_link, and end_link")
    return _load_cached_chain(str(Path(urdf_path)), base_link, end_link)


def urdf_fk(
    q: np.ndarray,
    tool_length_m: float = 0.115,
    tool_tcp_xyz_m: tuple[float, float, float] | None = None,
    tool_tcp_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
    urdf_path: str | Path | None = None,
    base_link: str = "",
    end_link: str = "",
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return tool-center pose and all intermediate link transforms.

    The arm geometry is loaded from a URDF chain. The list of transforms contains
    the base frame, each URDF joint/link frame, and the final TCP.
    """
    q = np.asarray(q, dtype=float)
    chain = _urdf_chain(urdf_path, base_link, end_link)
    if q.shape != (len(chain.active_joints),):
        raise ValueError(f"expected {len(chain.active_joints)} joints, got shape {q.shape}")
    return chain.fk(q, _tool_transform(tool_length_m, tool_tcp_xyz_m, tool_tcp_rpy_rad))


def urdf_joint_frames(
    q: np.ndarray,
    urdf_path: str | Path | None = None,
    base_link: str = "",
    end_link: str = "",
) -> list[np.ndarray]:
    """Return base-frame transforms of the active URDF joint axes."""
    q = np.asarray(q, dtype=float)
    chain = _urdf_chain(urdf_path, base_link, end_link)
    if q.shape != (len(chain.active_joints),):
        raise ValueError(f"expected {len(chain.active_joints)} joints, got shape {q.shape}")
    frames = []
    for joint, frame in zip(chain.joints, chain.joint_frames(q)):
        if joint.active:
            frames.append(frame)
    return frames


def urdf_geometric_jacobian(
    q: np.ndarray,
    tool_length_m: float,
    tool_tcp_xyz_m: tuple[float, float, float] | None = None,
    tool_tcp_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
    urdf_path: str | Path | None = None,
    base_link: str = "",
    end_link: str = "",
) -> np.ndarray:
    chain = _urdf_chain(urdf_path, base_link, end_link)
    tcp_t = _tool_transform(tool_length_m, tool_tcp_xyz_m, tool_tcp_rpy_rad)
    return chain.jacobian(q, tcp_t)


def jacobian_quality(
    q: np.ndarray,
    cfg: URDFIKConfig,
) -> tuple[float, float]:
    """Return Yoshikawa manipulability and the smallest Jacobian singular value."""
    jacobian = urdf_geometric_jacobian(
        q,
        cfg.tool_length_m,
        cfg.tool_tcp_xyz_m,
        cfg.tool_tcp_rpy_rad,
        cfg.urdf_path,
        cfg.base_link,
        cfg.end_link,
    )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    # prod(singular_values) equals sqrt(det(J J^T)) but is numerically stable.
    return float(np.prod(singular_values)), float(np.min(singular_values))


def _rotvec_error(R_cur: np.ndarray, R_goal: np.ndarray) -> np.ndarray:
    R_err = R_goal @ R_cur.T
    v = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    )
    cos_angle = (np.trace(R_err) - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, float(cos_angle)))
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return 0.5 * v
    return angle / (2.0 * math.sin(angle)) * v


def _within_limits(q: np.ndarray, limits: np.ndarray) -> bool:
    return bool(np.all(q >= limits[:, 0] - 1e-6) and np.all(q <= limits[:, 1] + 1e-6))


def _clamp_to_limits(q: np.ndarray, limits: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(q, limits[:, 0]), limits[:, 1])


def solve_urdf_ik(
    target_T: np.ndarray,
    seed_q: np.ndarray,
    cfg: URDFIKConfig,
) -> tuple[bool, np.ndarray, float, float, int]:
    q = _clamp_to_limits(np.asarray(seed_q, dtype=float).copy(), cfg.joint_limits)
    lam2 = cfg.damping * cfg.damping
    W = np.diag([1.0, 1.0, 1.0, cfg.orientation_weight, cfg.orientation_weight, cfg.orientation_weight])
    best_q = q.copy()
    best_score = float("inf")
    best_pos = float("inf")
    best_rot = float("inf")

    for iteration in range(1, cfg.max_iters + 1):
        cur_T, _ = urdf_fk(
            q,
            cfg.tool_length_m,
            cfg.tool_tcp_xyz_m,
            cfg.tool_tcp_rpy_rad,
            cfg.urdf_path,
            cfg.base_link,
            cfg.end_link,
        )
        ep = target_T[:3, 3] - cur_T[:3, 3]
        er = _rotvec_error(cur_T[:3, :3], target_T[:3, :3])
        pos_err = float(np.linalg.norm(ep))
        rot_err = float(np.linalg.norm(er))
        score = pos_err + cfg.orientation_weight * rot_err
        if score < best_score:
            best_score, best_q, best_pos, best_rot = score, q.copy(), pos_err, rot_err
        if pos_err <= cfg.pos_tol_m and rot_err <= cfg.rot_tol_rad and _within_limits(q, cfg.joint_limits):
            return True, q, pos_err, rot_err, iteration

        e = np.concatenate([ep, cfg.orientation_weight * er])
        J = W @ urdf_geometric_jacobian(
            q,
            cfg.tool_length_m,
            cfg.tool_tcp_xyz_m,
            cfg.tool_tcp_rpy_rad,
            cfg.urdf_path,
            cfg.base_link,
            cfg.end_link,
        )
        JJt = J @ J.T + lam2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, e)

        # Project the home-pose pull into the Jacobian nullspace. Directly
        # adding it biases a six-axis arm away from its Cartesian target.
        J_pinv = J.T @ np.linalg.solve(JJt, np.eye(6))
        dq += cfg.nullspace_weight * (np.eye(len(q)) - J_pinv @ J) @ (cfg.q_home - q)
        step_norm = float(np.linalg.norm(dq, ord=np.inf))
        if step_norm > cfg.max_joint_step_rad:
            dq *= cfg.max_joint_step_rad / step_norm
        q = _clamp_to_limits(q + dq, cfg.joint_limits)

    return False, best_q, best_pos, best_rot, cfg.max_iters


def _yaw_candidates(n: int) -> list[float]:
    if n <= 1:
        return [0.0]
    vals = np.linspace(-math.pi, math.pi, n, endpoint=False)
    return sorted([float(v) for v in vals], key=abs)


def _estimate_length(points: list[TrajectoryPoint]) -> float:
    if len(points) < 2:
        return 0.0
    P = np.array([pt.p for pt in points])
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def _joint_motion(points: list[TrajectoryPoint]) -> float:
    if len(points) < 2:
        return 0.0
    Q = np.array([pt.q for pt in points])
    return float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())


def _link_capsules(q: np.ndarray, cfg: URDFIKConfig) -> list[LinkCapsule]:
    _, Ts = urdf_fk(
        q,
        cfg.tool_length_m,
        cfg.tool_tcp_xyz_m,
        cfg.tool_tcp_rpy_rad,
        cfg.urdf_path,
        cfg.base_link,
        cfg.end_link,
    )
    # Use active-joint frames only. The raw FK list includes fixed URDF helper
    # frames; treating those as links creates overlapping duplicate capsules.
    active_joint_frames = urdf_joint_frames(
        q,
        cfg.urdf_path,
        cfg.base_link,
        cfg.end_link,
    )
    # Robot packages tune this to omit base-mounted frames that are below the
    # print plane by construction (not an arm/bed collision).
    arm_frames = active_joint_frames[cfg.collision_skip_frames:]
    capsules = [
        LinkCapsule(
            name=f"link_{i}",
            start_m=first[:3, 3],
            end_m=second[:3, 3],
            radius_m=cfg.link_capsule_radius_m,
        )
        for i, (first, second) in enumerate(zip(arm_frames, arm_frames[1:]))
    ]
    if arm_frames and Ts:
        capsules.append(
            LinkCapsule(
                name="tool",
                start_m=arm_frames[-1][:3, 3],
                end_m=Ts[-1][:3, 3],
                radius_m=cfg.tool_capsule_radius_m,
            )
        )
    return capsules


def _collision_warnings(
    q: np.ndarray,
    cfg: URDFIKConfig,
    index: int,
    printed_volume: AxisAlignedBox | None = None,
) -> list[str]:
    half_x, half_y = cfg.bed_half_extents_xy_m
    center_x, center_y = cfg.bed_center_xy_m
    bed_box = AxisAlignedBox(
        minimum_m=np.array([center_x - half_x, center_y - half_y, cfg.bed_z_m - cfg.bed_thickness_m]),
        maximum_m=np.array([center_x + half_x, center_y + half_y, cfg.bed_z_m]),
    )
    return [
        f"waypoint {index}: {warning}"
        for warning in capsule_collision_warnings(
            _link_capsules(q, cfg),
            bed_box=bed_box,
            bed_z_m=cfg.bed_z_m,
            bed_clearance_m=cfg.min_clearance_m,
            printed_volume=printed_volume,
            nozzle_clearance_m=cfg.nozzle_print_clearance_m,
        )
    ]


@dataclass
class _IKCandidate:
    q: np.ndarray
    yaw: float
    success: bool
    pos_error_m: float
    rot_error_rad: float
    unary_cost: float
    ik_iterations: int = 0


def _candidate_unary_cost(
    q: np.ndarray,
    success: bool,
    pos_error_m: float,
    rot_error_rad: float,
    cfg: URDFIKConfig,
) -> float:
    cost = cfg.global_dp_ik_error_weight * (pos_error_m + cfg.orientation_weight * rot_error_rad)
    if not success:
        cost += 10.0
    if cfg.global_dp_singularity_weight > 0.0:
        _, sigma_min = jacobian_quality(q, cfg)
        cost += cfg.global_dp_singularity_weight / (sigma_min + 1e-9)
    return cost


def _ik_candidates_for_waypoint(wp, seed_q: np.ndarray, cfg: URDFIKConfig) -> list[_IKCandidate]:
    """Solve an independent yaw/IK candidate set for one waypoint."""
    yaws = _yaw_candidates(cfg.yaw_samples) if wp.yaw_free else [wp.yaw]
    candidates = []
    for yaw in yaws:
        ok, q, pos_err, rot_err, iterations = solve_urdf_ik(wp.pose_matrix(yaw), seed_q, cfg)
        unary = _candidate_unary_cost(q, ok, pos_err, rot_err, cfg)
        if cfg.global_dp_collision_penalty > 0.0:
            unary += cfg.global_dp_collision_penalty * len(_collision_warnings(q, cfg, -1))
        candidates.append(_IKCandidate(q, yaw, ok, pos_err, rot_err, unary, iterations))
    return candidates


def _select_candidates_dp(
    candidate_layers: list[list[_IKCandidate]],
    seed_q: np.ndarray,
    cfg: URDFIKConfig,
    segment_durations_s: np.ndarray | None = None,
) -> list[_IKCandidate]:
    """Use second-order dynamic programming to choose a smooth IK sequence.

    The DP state is an ordered pair of adjacent candidates.  This retains the
    previous joint delta, allowing the transition cost to penalize changes in
    joint velocity rather than only point-to-point joint motion.
    """
    if not candidate_layers:
        return []

    motion_weight = cfg.global_dp_motion_weight
    smoothness_weight = cfg.global_dp_smoothness_weight
    if segment_durations_s is None:
        segment_durations_s = np.ones(len(candidate_layers), dtype=float)
    else:
        segment_durations_s = np.asarray(segment_durations_s, dtype=float)
        if segment_durations_s.shape != (len(candidate_layers),):
            raise ValueError("segment_durations_s must have one positive duration per candidate layer")
        if not np.all(np.isfinite(segment_durations_s)) or np.any(segment_durations_s <= 0.0):
            raise ValueError("segment_durations_s must contain finite values greater than zero")
    if len(candidate_layers) == 1:
        only = candidate_layers[0]
        best = min(
            only,
            key=lambda candidate: candidate.unary_cost
            + motion_weight * float(np.sum((candidate.q - seed_q) ** 2)),
        )
        return [best]

    q0 = np.asarray([candidate.q for candidate in candidate_layers[0]])
    q1 = np.asarray([candidate.q for candidate in candidate_layers[1]])
    unary0 = np.asarray([candidate.unary_cost for candidate in candidate_layers[0]])
    unary1 = np.asarray([candidate.unary_cost for candidate in candidate_layers[1]])
    initial_velocity = (q0 - seed_q) / segment_durations_s[0]
    current_velocity = (q1[None, :, :] - q0[:, None, :]) / segment_durations_s[1]
    dp = (
        unary0[:, None]
        + unary1[None, :]
        + motion_weight * np.sum(initial_velocity**2, axis=1)[:, None]
        + motion_weight * np.sum(current_velocity**2, axis=2)
        + smoothness_weight * np.sum((current_velocity - initial_velocity[:, None, :]) ** 2, axis=2)
    )
    backpointers: list[np.ndarray | None] = [None, None]

    for i in range(2, len(candidate_layers)):
        q_prevprev = np.asarray([candidate.q for candidate in candidate_layers[i - 2]])
        q_prev = np.asarray([candidate.q for candidate in candidate_layers[i - 1]])
        q_current = np.asarray([candidate.q for candidate in candidate_layers[i]])
        unary_current = np.asarray([candidate.unary_cost for candidate in candidate_layers[i]])
        previous_velocity = (q_prev[None, :, :] - q_prevprev[:, None, :]) / segment_durations_s[i - 1]
        next_dp = np.empty((len(q_prev), len(q_current)), dtype=float)
        back = np.empty((len(q_prev), len(q_current)), dtype=int)
        for current_index, current_q in enumerate(q_current):
            next_velocity = (current_q[None, None, :] - q_prev[None, :, :]) / segment_durations_s[i]
            transition = (
                dp
                + motion_weight * np.sum(next_velocity**2, axis=2)
                + smoothness_weight * np.sum((next_velocity - previous_velocity) ** 2, axis=2)
            )
            best_previous = np.argmin(transition, axis=0)
            next_dp[:, current_index] = unary_current[current_index] + transition[best_previous, np.arange(len(q_prev))]
            back[:, current_index] = best_previous
        dp = next_dp
        backpointers.append(back)

    previous_index, current_index = np.unravel_index(int(np.argmin(dp)), dp.shape)
    selected_indices = [0] * len(candidate_layers)
    selected_indices[-2] = int(previous_index)
    selected_indices[-1] = int(current_index)
    for i in range(len(candidate_layers) - 1, 1, -1):
        back = backpointers[i]
        assert back is not None
        prior_index = int(back[previous_index, current_index])
        selected_indices[i - 2] = prior_index
        previous_index, current_index = prior_index, previous_index
    return [candidate_layers[i][index] for i, index in enumerate(selected_indices)]


def _point_from_candidate(index: int, wp, candidate: _IKCandidate, cfg: URDFIKConfig) -> TrajectoryPoint:
    manipulability, sigma_min = jacobian_quality(candidate.q, cfg)
    return TrajectoryPoint(
        index=index,
        q=candidate.q,
        p=wp.p,
        yaw=candidate.yaw,
        is_print=wp.is_print,
        layer=wp.layer,
        seg_id=wp.seg_id,
        feed_m_s=wp.feed_m_s,
        de=wp.de,
        material=wp.material,
        extrusion_volume_mm3=wp.extrusion_volume_mm3,
        extrusion_mass_g=wp.extrusion_mass_g,
        pos_error_m=candidate.pos_error_m,
        rot_error_rad=candidate.rot_error_rad,
        ik_iterations=candidate.ik_iterations,
        jacobian_manipulability=manipulability,
        jacobian_min_singular_value=sigma_min,
    )


def _reach_warning(wp, index: int, cfg: URDFIKConfig) -> str | None:
    reach = float(np.linalg.norm(wp.p))
    if reach > cfg.max_reach_m + 0.08:
        return f"waypoint {index}: target reach {reach:.3f} m is outside nominal robot range"
    return None


def _solve_greedy_waypoint(index: int, wp, seed_q: np.ndarray, cfg: URDFIKConfig) -> _IKCandidate:
    candidates = _ik_candidates_for_waypoint(wp, seed_q, cfg)
    return min(
        candidates,
        key=lambda candidate: candidate.unary_cost + float(np.linalg.norm(candidate.q - seed_q)),
    )


def _solve_deposition_run_dp(
    run: list[tuple[int, object]], seed_q: np.ndarray, cfg: URDFIKConfig
) -> list[_IKCandidate]:
    """Generate yaw/IK candidates, then optimize one deposition run globally."""
    candidate_layers = []
    reference_q = seed_q.copy()
    for _, wp in run:
        candidates = _ik_candidates_for_waypoint(wp, reference_q, cfg)
        candidate_layers.append(candidates)
        # This reference only improves numerical convergence while generating
        # candidates; final choices are made by the DP over the complete run.
        reference_q = min(
            candidates,
            key=lambda candidate: candidate.unary_cost + float(np.linalg.norm(candidate.q - reference_q)),
        ).q
    durations_s = np.ones(len(run), dtype=float)
    for i in range(1, len(run)):
        previous_wp = run[i - 1][1]
        current_wp = run[i][1]
        distance_m = float(np.linalg.norm(current_wp.p - previous_wp.p))
        durations_s[i] = max(distance_m / max(float(current_wp.feed_m_s), 1e-12), 1e-6)
    if len(durations_s) > 1:
        durations_s[0] = durations_s[1]
    return _select_candidates_dp(candidate_layers, seed_q, cfg, durations_s)


def _yaw_discontinuity_count(points: list[TrajectoryPoint], threshold_rad: float) -> int:
    count = 0
    for previous, current in zip(points, points[1:]):
        if previous.seg_id != current.seg_id:
            continue
        delta = (current.yaw - previous.yaw + math.pi) % (2.0 * math.pi) - math.pi
        if abs(delta) > threshold_rad:
            count += 1
    return count


def _advance_printed_volume(
    active_layer: int | None,
    current_layer_volume: AxisAlignedBox | None,
    completed_volume: AxisAlignedBox | None,
    next_layer: int,
) -> tuple[int, AxisAlignedBox | None, AxisAlignedBox | None]:
    """Move completed deposition from lower layers into the collision volume."""
    if active_layer is None:
        return next_layer, current_layer_volume, completed_volume
    if next_layer > active_layer:
        return next_layer, None, merge_boxes(completed_volume, current_layer_volume)
    return active_layer, current_layer_volume, completed_volume


def solve_urdf_path(path: PathPrep, cfg: URDFIKConfig | None = None) -> RobotTrajectory:
    cfg = cfg or URDFIKConfig()
    if cfg.ik_selection_mode not in {"greedy", "global_dp"}:
        raise ValueError("ik_selection_mode must be 'greedy' or 'global_dp'")
    points: list[TrajectoryPoint] = []
    failed: list[int] = []
    warnings: list[str] = []
    q_prev = cfg.q_home.copy()
    stride = max(1, int(cfg.ik_stride))
    indexed_waypoints = list(enumerate(path.waypoints[::stride]))
    if cfg.max_waypoints is not None:
        indexed_waypoints = indexed_waypoints[:cfg.max_waypoints]
    timing_is_preview = stride > 1 or len(indexed_waypoints) < len(path.waypoints)
    if timing_is_preview:
        warnings.append(
            "IK sampling or waypoint capping is active: timing is a preview and not a full-path print-time estimate"
        )

    cursor = 0
    active_layer: int | None = None
    current_layer_volume: AxisAlignedBox | None = None
    completed_print_volume: AxisAlignedBox | None = None
    while cursor < len(indexed_waypoints):
        local_i, wp = indexed_waypoints[cursor]
        index = local_i * stride
        active_layer, current_layer_volume, completed_print_volume = _advance_printed_volume(
            active_layer,
            current_layer_volume,
            completed_print_volume,
            wp.layer,
        )
        if cfg.ik_selection_mode == "global_dp" and wp.is_print:
            end = cursor + 1
            while (
                end < len(indexed_waypoints)
                and indexed_waypoints[end][1].is_print
                and indexed_waypoints[end][1].seg_id == wp.seg_id
            ):
                end += 1
            run = [(local * stride, run_wp) for local, run_wp in indexed_waypoints[cursor:end]]
            selected = _solve_deposition_run_dp(run, q_prev, cfg)
            for (run_index, run_wp), candidate in zip(run, selected):
                reach_warning = _reach_warning(run_wp, run_index, cfg)
                if reach_warning:
                    warnings.append(reach_warning)
                if not candidate.success:
                    failed.append(run_index)
                warnings.extend(
                    _collision_warnings(candidate.q, cfg, run_index, completed_print_volume)
                )
                points.append(_point_from_candidate(run_index, run_wp, candidate, cfg))
                current_layer_volume = merge_boxes(
                    current_layer_volume,
                    box_from_point(run_wp.p, cfg.printed_volume_padding_m),
                )
            q_prev = selected[-1].q
            cursor = end
            continue

        reach_warning = _reach_warning(wp, index, cfg)
        if reach_warning:
            warnings.append(reach_warning)
        candidate = _solve_greedy_waypoint(index, wp, q_prev, cfg)
        if not candidate.success:
            failed.append(index)
        warnings.extend(_collision_warnings(candidate.q, cfg, index, completed_print_volume))
        points.append(_point_from_candidate(index, wp, candidate, cfg))
        if wp.is_print:
            current_layer_volume = merge_boxes(
                current_layer_volume,
                box_from_point(wp.p, cfg.printed_volume_padding_m),
            )
        q_prev = candidate.q
        cursor += 1

    successful = len(indexed_waypoints) - len(set(failed))
    report = IKReport(
        success=(successful == len(indexed_waypoints) and len(points) == len(indexed_waypoints)),
        attempted=len(indexed_waypoints),
        generated=len(points),
        successful=successful,
        failed_indices=failed,
        warnings=warnings,
        total_joint_motion_rad=_joint_motion(points),
        estimated_cartesian_length_m=_estimate_length(points),
        selection_mode=cfg.ik_selection_mode,
        yaw_discontinuity_count=_yaw_discontinuity_count(points, cfg.yaw_discontinuity_threshold_rad),
    )
    return RobotTrajectory(points=points, report=report, config=cfg, timing_is_preview=timing_is_preview)


@dataclass(frozen=True)
class URDFRobotPlanner(RobotPlanner):
    """Generic planner configured by a serial-chain URDF and URDFIKConfig."""

    config: URDFIKConfig | None = None

    def solve(self, path: PathPrep) -> RobotTrajectory:
        return solve_urdf_path(path, self.config)
