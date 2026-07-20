"""Dependency-light URDF kinematics and damped least-squares IK.

The parser supports one serial chain containing fixed, revolute, continuous,
and prismatic joints. It is intentionally NumPy-only so the platform can run
small analysis and export jobs without ROS or simulator dependencies.
"""

from __future__ import annotations

import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class URDFJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float

    @property
    def active(self) -> bool:
        return self.joint_type in {"revolute", "continuous", "prismatic"}


@dataclass(frozen=True)
class IKResult:
    success: bool
    q: np.ndarray
    pos_error_m: float
    rot_error_rad: float
    iterations: int


class URDFKinematicChain:
    def __init__(self, joints: list[URDFJoint], base_link: str, end_link: str):
        self.joints = joints
        self.base_link = base_link
        self.end_link = end_link
        self.active_joints = [joint for joint in joints if joint.active]

    @property
    def joint_names(self) -> list[str]:
        return [joint.name for joint in self.active_joints]

    @property
    def joint_limits(self) -> np.ndarray:
        return np.array([[joint.lower, joint.upper] for joint in self.active_joints], dtype=float)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        limits = self.joint_limits
        return np.minimum(np.maximum(q, limits[:, 0]), limits[:, 1])

    def fk(self, q: np.ndarray, tcp_transform: np.ndarray | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.active_joints),):
            raise ValueError(f"expected {len(self.active_joints)} joints, got shape {q.shape}")

        transform = np.eye(4)
        frames = [transform.copy()]
        active_index = 0
        for joint in self.joints:
            transform = transform @ _transform(_rpy_matrix(joint.origin_rpy), joint.origin_xyz)
            if joint.active:
                value = float(q[active_index])
                active_index += 1
                if joint.joint_type == "prismatic":
                    transform = transform @ _transform(translation=joint.axis * value)
                else:
                    transform = transform @ _transform(_axis_angle_matrix(joint.axis, value))
            frames.append(transform.copy())
        if tcp_transform is not None:
            transform = transform @ tcp_transform
            frames.append(transform.copy())
        return transform, frames

    def joint_frames(self, q: np.ndarray) -> list[np.ndarray]:
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.active_joints),):
            raise ValueError(f"expected {len(self.active_joints)} joints, got shape {q.shape}")

        transform = np.eye(4)
        frames: list[np.ndarray] = []
        active_index = 0
        for joint in self.joints:
            transform = transform @ _transform(_rpy_matrix(joint.origin_rpy), joint.origin_xyz)
            if joint.active:
                frames.append(transform.copy())
                value = float(q[active_index])
                active_index += 1
                if joint.joint_type == "prismatic":
                    transform = transform @ _transform(translation=joint.axis * value)
                else:
                    transform = transform @ _transform(_axis_angle_matrix(joint.axis, value))
            else:
                frames.append(transform.copy())
        return frames

    def jacobian(self, q: np.ndarray, tcp_transform: np.ndarray | None = None) -> np.ndarray:
        tool_t, _ = self.fk(q, tcp_transform)
        pe = tool_t[:3, 3]
        joint_frames = self.joint_frames(q)
        jacobian = np.zeros((6, len(self.active_joints)), dtype=float)
        active_index = 0
        for joint, frame in zip(self.joints, joint_frames):
            if not joint.active:
                continue
            axis = frame[:3, :3] @ _unit(joint.axis)
            origin = frame[:3, 3]
            if joint.joint_type == "prismatic":
                jacobian[:3, active_index] = axis
            else:
                jacobian[:3, active_index] = np.cross(axis, pe - origin)
                jacobian[3:, active_index] = axis
            active_index += 1
        return jacobian

    def solve_ik(
        self,
        target_t: np.ndarray,
        seed_q: np.ndarray,
        *,
        pos_tol_m: float = 0.008,
        rot_tol_rad: float = 0.08,
        max_iters: int = 180,
        damping: float = 0.035,
        orientation_weight: float = 0.35,
        nullspace_weight: float = 0.015,
        max_joint_step_rad: float = 0.10,
        q_home: np.ndarray | None = None,
        tcp_transform: np.ndarray | None = None,
    ) -> IKResult:
        q = self.clamp(np.asarray(seed_q, dtype=float).copy())
        home = np.zeros_like(q) if q_home is None else np.asarray(q_home, dtype=float)
        lam2 = damping * damping
        weights = np.diag([1.0, 1.0, 1.0, orientation_weight, orientation_weight, orientation_weight])
        best_q = q.copy()
        best_score = float("inf")
        best_pos = float("inf")
        best_rot = float("inf")

        for iteration in range(1, max_iters + 1):
            cur_t, _ = self.fk(q, tcp_transform)
            ep = target_t[:3, 3] - cur_t[:3, 3]
            er = _rotvec_error(cur_t[:3, :3], target_t[:3, :3])
            pos_err = float(np.linalg.norm(ep))
            rot_err = float(np.linalg.norm(er))
            score = pos_err + orientation_weight * rot_err
            if score < best_score:
                best_score = score
                best_q = q.copy()
                best_pos = pos_err
                best_rot = rot_err
            if pos_err <= pos_tol_m and rot_err <= rot_tol_rad:
                return IKResult(True, q, pos_err, rot_err, iteration)

            error = np.concatenate([ep, orientation_weight * er])
            jacobian = weights @ self.jacobian(q, tcp_transform)
            jj_t = jacobian @ jacobian.T + lam2 * np.eye(6)
            dq = jacobian.T @ np.linalg.solve(jj_t, error)
            # Keep the posture bias from perturbing the Cartesian task for
            # non-redundant chains such as the six-axis UR5.
            jacobian_pinv = jacobian.T @ np.linalg.solve(jj_t, np.eye(6))
            dq += nullspace_weight * (np.eye(len(q)) - jacobian_pinv @ jacobian) @ (home - q)
            step_norm = float(np.linalg.norm(dq, ord=np.inf))
            if step_norm > max_joint_step_rad:
                dq *= max_joint_step_rad / step_norm
            q = self.clamp(q + dq)

        return IKResult(False, best_q, best_pos, best_rot, max_iters)

    def sample_workspace(self, count: int) -> np.ndarray:
        limits = self.joint_limits
        positions = []
        for _ in range(count):
            q = np.array([random.uniform(lower, upper) for lower, upper in limits], dtype=float)
            pose, _ = self.fk(q)
            positions.append(pose[:3, 3])
        return np.array(positions, dtype=float)


def load_urdf_chain(
    urdf_path: str | Path,
    base_link: str | None = None,
    end_link: str | None = None,
) -> URDFKinematicChain:
    root = ET.parse(urdf_path).getroot()
    joints: list[URDFJoint] = []
    child_to_joint: dict[str, URDFJoint] = {}
    parent_links: set[str] = set()
    child_links: set[str] = set()

    for joint_node in root.findall("joint"):
        origin_node = joint_node.find("origin")
        axis_node = joint_node.find("axis")
        limit_node = joint_node.find("limit")
        joint_type = joint_node.attrib.get("type", "fixed")
        lower, upper = _default_limits(joint_type)
        if limit_node is not None:
            lower = float(limit_node.attrib.get("lower", lower))
            upper = float(limit_node.attrib.get("upper", upper))

        joint = URDFJoint(
            name=joint_node.attrib["name"],
            joint_type=joint_type,
            parent=joint_node.find("parent").attrib["link"],
            child=joint_node.find("child").attrib["link"],
            origin_xyz=_parse_vector(origin_node.attrib.get("xyz") if origin_node is not None else None, [0, 0, 0]),
            origin_rpy=_parse_vector(origin_node.attrib.get("rpy") if origin_node is not None else None, [0, 0, 0]),
            axis=_parse_vector(axis_node.attrib.get("xyz") if axis_node is not None else None, [0, 0, 1]),
            lower=lower,
            upper=upper,
        )
        joints.append(joint)
        child_to_joint[joint.child] = joint
        parent_links.add(joint.parent)
        child_links.add(joint.child)

    if not joints:
        raise ValueError(f"No joints found in {urdf_path}")

    resolved_base = base_link or next(iter(parent_links - child_links), joints[0].parent)
    resolved_end = end_link or next(iter(child_links - parent_links), joints[-1].child)
    chain: list[URDFJoint] = []
    link = resolved_end
    while link != resolved_base:
        if link not in child_to_joint:
            raise ValueError(f"Could not trace link '{link}' back to base '{resolved_base}'")
        joint = child_to_joint[link]
        chain.append(joint)
        link = joint.parent
    chain.reverse()
    return URDFKinematicChain(chain, resolved_base, resolved_end)


def _parse_vector(value: str | None, default: Iterable[float]) -> np.ndarray:
    if value is None:
        return np.array(list(default), dtype=float)
    return np.array([float(part) for part in value.split()], dtype=float)


def _default_limits(joint_type: str) -> tuple[float, float]:
    if joint_type == "continuous":
        return -math.pi, math.pi
    if joint_type == "prismatic":
        return -1.0, 1.0
    return -math.pi, math.pi


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = rotation
    if translation is not None:
        matrix[:3, 3] = translation
    return matrix


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = _unit(axis)
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def _rotvec_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    error_matrix = target @ current.T
    vector = np.array(
        [
            error_matrix[2, 1] - error_matrix[1, 2],
            error_matrix[0, 2] - error_matrix[2, 0],
            error_matrix[1, 0] - error_matrix[0, 1],
        ],
        dtype=float,
    )
    cos_angle = np.clip((np.trace(error_matrix) - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(float(cos_angle))
    if angle < 1e-9:
        return 0.5 * vector
    return angle / (2.0 * math.sin(angle)) * vector
