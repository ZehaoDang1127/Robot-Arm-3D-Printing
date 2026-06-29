"""
Stage 3: Franka Panda inverse kinematics and continuity optimization.

This module intentionally stays dependency-light: NumPy only. It implements a
practical Panda kinematic chain, damped least-squares IK, joint-limit checks,
simple table/nozzle collision checks, and redundancy handling by trying several
tool-yaw candidates while always seeding from the previous accepted joint state.

The optimizer preserves the G-code print order. It reduces robot motion through
continuous IK seeding and yaw redundancy, which is the safe optimization knob for
extrusion paths because it does not reorder deposition segments.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from franka_panda_parameters import MODIFIED_DH_LINKS, STANDARD_PANDA_JOINT_LIMITS_RAD
from stage2_pathprep import PathPrep, Waypoint


PANDA_JOINT_LIMITS = STANDARD_PANDA_JOINT_LIMITS_RAD
PANDA_HOME = np.array([0.0, -0.45, 0.0, -2.35, 0.0, 2.05, 0.75], dtype=float)


@dataclass
class IKConfig:
    tool_length_m: float = 0.115
    tool_tcp_xyz_m: tuple[float, float, float] | None = None
    tool_tcp_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # The bundled NumPy-only model is intended for planning/simulation export.
    # Keep these tolerances visible in output files; tighten them after
    # calibrating the actual nozzle TCP and Panda model in Isaac/Pinocchio.
    pos_tol_m: float = 0.008
    rot_tol_rad: float = 0.08
    max_iters: int = 180
    damping: float = 0.035
    orientation_weight: float = 0.35
    nullspace_weight: float = 0.015
    max_joint_step_rad: float = 0.10
    yaw_samples: int = 13
    joint_limits: np.ndarray = field(default_factory=lambda: PANDA_JOINT_LIMITS.copy())
    q_home: np.ndarray = field(default_factory=lambda: PANDA_HOME.copy())
    bed_z_m: float = 0.10
    min_clearance_m: float = 0.006
    max_reach_m: float = 0.855
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
    pos_error_m: float
    rot_error_rad: float


@dataclass
class IKReport:
    success: bool
    attempted: int
    solved: int
    failed_indices: list[int]
    warnings: list[str]
    total_joint_motion_rad: float
    estimated_cartesian_length_m: float

    def summary(self) -> str:
        state = "success" if self.success else "incomplete"
        return (
            f"IK {state}: {self.solved}/{self.attempted} trajectory points generated\n"
            f"outside-tolerance indices: {self.failed_indices[:12]}"
            f"{' ...' if len(self.failed_indices) > 12 else ''}\n"
            f"joint motion : {self.total_joint_motion_rad:.3f} rad\n"
            f"path length   : {self.estimated_cartesian_length_m:.3f} m\n"
            f"warnings     : {len(self.warnings)}"
        )


@dataclass
class RobotTrajectory:
    points: list[TrajectoryPoint]
    report: IKReport
    config: IKConfig

    def q_array(self) -> np.ndarray:
        return np.array([pt.q for pt in self.points], dtype=float)

    def export_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
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
                    "pos_error_m",
                    "rot_error_rad",
                    "q1",
                    "q2",
                    "q3",
                    "q4",
                    "q5",
                    "q6",
                    "q7",
                ]
            )
            for pt in self.points:
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
                        pt.pos_error_m,
                        pt.rot_error_rad,
                        *pt.q.tolist(),
                    ]
                )

    def export_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "joint_names": [f"panda_joint{i}" for i in range(1, 8)],
            "success": self.report.success,
            "report": {
                "attempted": self.report.attempted,
                "solved": self.report.solved,
                "failed_indices": self.report.failed_indices,
                "warnings": self.report.warnings,
                "total_joint_motion_rad": self.report.total_joint_motion_rad,
                "estimated_cartesian_length_m": self.report.estimated_cartesian_length_m,
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
                    "pos_error_m": pt.pos_error_m,
                    "rot_error_rad": pt.rot_error_rad,
                }
                for pt in self.points
            ],
        }
        path.write_text(json.dumps(payload, indent=2))


def _modified_dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Peter Corke modified-DH transform: Rx(alpha) Tx(a) Rz(theta) Tz(d)."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _modified_dh_joint_frame(a: float, alpha: float) -> np.ndarray:
    """Transform from previous link frame to the current modified-DH joint axis."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [1.0, 0.0, 0.0, a],
            [0.0, ca, -sa, 0.0],
            [0.0, sa, ca, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


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
    """Nozzle TCP transform from final Panda link frame."""
    xyz = tool_tcp_xyz_m if tool_tcp_xyz_m is not None else (0.0, 0.0, tool_length_m)
    T = np.eye(4)
    T[:3, :3] = _rpy_matrix(*tool_tcp_rpy_rad)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def panda_fk(
    q: np.ndarray,
    tool_length_m: float = 0.115,
    tool_tcp_xyz_m: tuple[float, float, float] | None = None,
    tool_tcp_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return tool-center pose and all intermediate link transforms.

    The arm geometry is loaded from `franka_panda_parameters.MODIFIED_DH_LINKS`,
    extracted from the referenced MATLAB `panda7dof_robot_gen.m` file. The list
    of transforms contains the base frame, each link frame, and the final TCP.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (7,):
        raise ValueError(f"expected 7 Panda joints, got shape {q.shape}")
    T = np.eye(4)
    Ts = [T.copy()]
    for qi, link in zip(q, MODIFIED_DH_LINKS):
        theta = float(qi + link.theta_offset_rad)
        T = T @ _modified_dh(theta, link.d_m, link.a_m, link.alpha_rad)
        Ts.append(T.copy())
    T = T @ _tool_transform(tool_length_m, tool_tcp_xyz_m, tool_tcp_rpy_rad)
    Ts.append(T.copy())
    return T, Ts


def panda_joint_frames(q: np.ndarray) -> list[np.ndarray]:
    """Return base-frame transforms of the seven modified-DH joint axes."""
    q = np.asarray(q, dtype=float)
    if q.shape != (7,):
        raise ValueError(f"expected 7 Panda joints, got shape {q.shape}")

    T = np.eye(4)
    frames: list[np.ndarray] = []
    for qi, link in zip(q, MODIFIED_DH_LINKS):
        joint_T = T @ _modified_dh_joint_frame(link.a_m, link.alpha_rad)
        frames.append(joint_T.copy())
        theta = float(qi + link.theta_offset_rad)
        T = T @ _modified_dh(theta, link.d_m, link.a_m, link.alpha_rad)
    return frames


def geometric_jacobian(
    q: np.ndarray,
    tool_length_m: float,
    tool_tcp_xyz_m: tuple[float, float, float] | None = None,
    tool_tcp_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    T_tool, _ = panda_fk(q, tool_length_m, tool_tcp_xyz_m, tool_tcp_rpy_rad)
    pe = T_tool[:3, 3]
    joint_frames = panda_joint_frames(q)
    J = np.zeros((6, 7), dtype=float)
    for i, joint_T in enumerate(joint_frames):
        zi = joint_T[:3, 2]
        pi = joint_T[:3, 3]
        J[:3, i] = np.cross(zi, pe - pi)
        J[3:, i] = zi
    return J


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


def solve_ik(
    target_T: np.ndarray,
    seed_q: np.ndarray,
    cfg: IKConfig,
) -> tuple[bool, np.ndarray, float, float]:
    q = _clamp_to_limits(np.asarray(seed_q, dtype=float).copy(), cfg.joint_limits)
    lam2 = cfg.damping * cfg.damping
    W = np.diag([1.0, 1.0, 1.0, cfg.orientation_weight, cfg.orientation_weight, cfg.orientation_weight])
    best_q = q.copy()
    best_score = float("inf")
    best_pos = float("inf")
    best_rot = float("inf")

    for _ in range(cfg.max_iters):
        cur_T, _ = panda_fk(q, cfg.tool_length_m, cfg.tool_tcp_xyz_m, cfg.tool_tcp_rpy_rad)
        ep = target_T[:3, 3] - cur_T[:3, 3]
        er = _rotvec_error(cur_T[:3, :3], target_T[:3, :3])
        pos_err = float(np.linalg.norm(ep))
        rot_err = float(np.linalg.norm(er))
        score = pos_err + cfg.orientation_weight * rot_err
        if score < best_score:
            best_score, best_q, best_pos, best_rot = score, q.copy(), pos_err, rot_err
        if pos_err <= cfg.pos_tol_m and rot_err <= cfg.rot_tol_rad and _within_limits(q, cfg.joint_limits):
            return True, q, pos_err, rot_err

        e = np.concatenate([ep, cfg.orientation_weight * er])
        J = W @ geometric_jacobian(q, cfg.tool_length_m, cfg.tool_tcp_xyz_m, cfg.tool_tcp_rpy_rad)
        JJt = J @ J.T + lam2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, e)

        # Mild nullspace pull toward home keeps the arm away from awkward limits.
        dq += cfg.nullspace_weight * (cfg.q_home - q)
        step_norm = float(np.linalg.norm(dq, ord=np.inf))
        if step_norm > cfg.max_joint_step_rad:
            dq *= cfg.max_joint_step_rad / step_norm
        q = _clamp_to_limits(q + dq, cfg.joint_limits)

    return False, best_q, best_pos, best_rot


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


def _collision_warnings(q: np.ndarray, cfg: IKConfig, index: int) -> list[str]:
    warnings = []
    _, Ts = panda_fk(q, cfg.tool_length_m, cfg.tool_tcp_xyz_m, cfg.tool_tcp_rpy_rad)
    min_z = min(float(T[2, 3]) for T in Ts[1:])
    if min_z < cfg.bed_z_m + cfg.min_clearance_m:
        warnings.append(
            f"waypoint {index}: link sample below bed clearance "
            f"({min_z:.4f} m < {cfg.bed_z_m + cfg.min_clearance_m:.4f} m)"
        )
    return warnings


def solve_path_ik(path: PathPrep, cfg: IKConfig | None = None) -> RobotTrajectory:
    cfg = cfg or IKConfig()
    points: list[TrajectoryPoint] = []
    failed: list[int] = []
    warnings: list[str] = []
    q_prev = cfg.q_home.copy()
    stride = max(1, int(cfg.ik_stride))
    indexed_waypoints = list(enumerate(path.waypoints[::stride]))
    if cfg.max_waypoints is not None:
        indexed_waypoints = indexed_waypoints[:cfg.max_waypoints]

    for local_i, wp in indexed_waypoints:
        i = local_i * stride
        reach = float(np.linalg.norm(wp.p))
        if reach > cfg.max_reach_m + 0.08:
            warnings.append(f"waypoint {i}: target reach {reach:.3f} m is outside nominal Panda range")

        candidates: list[tuple[float, bool, np.ndarray, float, float, float]] = []
        yaws = _yaw_candidates(cfg.yaw_samples) if wp.yaw_free else [wp.yaw]
        # Prefer the previous yaw for continuity when available.
        if points and points[-1].yaw not in yaws:
            yaws = [points[-1].yaw] + yaws

        for yaw in yaws:
            target_T = wp.pose_matrix(yaw)
            ok, q, pos_err, rot_err = solve_ik(target_T, q_prev, cfg)
            motion = float(np.linalg.norm(q - q_prev))
            penalty = motion + 25.0 * pos_err + cfg.orientation_weight * rot_err
            if not ok:
                penalty += 10.0
            candidates.append((penalty, ok, q, yaw, pos_err, rot_err))

        if not candidates:
            failed.append(i)
            continue

        candidates.sort(key=lambda item: item[0])
        _, ok_best, q_best, yaw_best, pos_err, rot_err = candidates[0]
        if not ok_best:
            failed.append(i)
        warnings.extend(_collision_warnings(q_best, cfg, i))
        q_prev = q_best
        points.append(
            TrajectoryPoint(
                index=i,
                q=q_best,
                p=wp.p,
                yaw=yaw_best,
                is_print=wp.is_print,
                layer=wp.layer,
                seg_id=wp.seg_id,
                feed_m_s=wp.feed_m_s,
                de=wp.de,
                pos_error_m=pos_err,
                rot_error_rad=rot_err,
            )
        )

    report = IKReport(
        success=(len(failed) == 0 and len(points) == len(indexed_waypoints)),
        attempted=len(indexed_waypoints),
        solved=len(points),
        failed_indices=failed,
        warnings=warnings,
        total_joint_motion_rad=_joint_motion(points),
        estimated_cartesian_length_m=_estimate_length(points),
    )
    return RobotTrajectory(points=points, report=report, config=cfg)
