#!/usr/bin/env python3
"""General URDF IK analysis CLI for the robotic printing platform."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from robotic_printing_platform.robots.franka_panda import DEFAULT_ROBOT_CONFIG_DIR
from robotic_printing_platform.robots.urdf_kinematics import load_urdf_chain


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = rotation
    if translation is not None:
        matrix[:3, 3] = translation
    return matrix


def vector_text(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze FK, workspace, and IK for a serial URDF chain.")
    parser.add_argument("--robot-config-dir", default=str(DEFAULT_ROBOT_CONFIG_DIR), help="Folder containing robot_config.json and robot.urdf.")
    parser.add_argument("--urdf", default=None, help="Override URDF file path.")
    parser.add_argument("--base-link", default=None, help="Override base link name.")
    parser.add_argument("--end-link", default=None, help="Override end-effector/flange link name.")
    parser.add_argument("--home-q", nargs="*", type=float, help="Home/seed active joint values.")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Target position in meters.")
    parser.add_argument("--target-rpy", nargs=3, type=float, metavar=("R", "P", "Y"), help="Target RPY in radians.")
    parser.add_argument("--tool-xyz", nargs=3, type=float, default=[0.0, 0.0, 0.115], help="Tool TCP offset.")
    parser.add_argument("--tool-rpy", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Tool TCP orientation.")
    parser.add_argument("--samples", type=int, default=500, help="Random workspace samples.")
    parser.add_argument("--max-iters", type=int, default=180, help="Maximum IK iterations.")
    return parser


def load_robot_defaults(config_dir: Path) -> dict:
    config_path = config_dir / "robot_config.json"
    if not config_path.exists():
        return {}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if "urdf_path" in data:
        urdf_path = Path(data["urdf_path"])
        data["urdf_path"] = str(urdf_path if urdf_path.is_absolute() else config_dir / urdf_path)
    return data


def main() -> None:
    args = build_parser().parse_args()
    robot_config_dir = Path(args.robot_config_dir).resolve()
    robot_defaults = load_robot_defaults(robot_config_dir)
    urdf_path = args.urdf or robot_defaults.get("urdf_path", str(robot_config_dir / "robot.urdf"))
    base_link = args.base_link or robot_defaults.get("base_link")
    end_link = args.end_link or robot_defaults.get("end_link")

    chain = load_urdf_chain(Path(urdf_path), base_link, end_link)
    home_default = robot_defaults.get("home_q_rad", [0.0] * len(chain.active_joints))
    home_q = np.array(args.home_q if args.home_q else home_default, dtype=float)
    if home_q.shape != (len(chain.active_joints),):
        raise ValueError(f"--home-q must contain {len(chain.active_joints)} values")

    tool_t = transform(rpy_matrix(*args.tool_rpy), np.array(args.tool_xyz, dtype=float))
    home_pose, _ = chain.fk(home_q, tool_t)

    print(f"Robot config: {robot_config_dir}")
    print(f"URDF: {urdf_path}")
    print(f"Chain: {chain.base_link} -> {chain.end_link}")
    print(f"Active joints ({len(chain.active_joints)}): {', '.join(chain.joint_names)}")
    print(f"Home q: {vector_text(home_q)}")
    print(f"Home TCP xyz: {vector_text(home_pose[:3, 3])}")
    print("Joint limits:")
    for name, (lower, upper) in zip(chain.joint_names, chain.joint_limits):
        print(f"  {name}: [{lower:.6f}, {upper:.6f}]")

    if args.samples > 0:
        points = chain.sample_workspace(args.samples)
        radii = np.linalg.norm(points, axis=1)
        print("\nWorkspace sample")
        print(f"  samples: {args.samples}")
        print(f"  min xyz: {vector_text(points.min(axis=0))}")
        print(f"  max xyz: {vector_text(points.max(axis=0))}")
        print(f"  mean xyz: {vector_text(points.mean(axis=0))}")
        print(f"  max radius: {radii.max():.6f} m")

    if args.target is None:
        return

    target_t = transform(translation=np.array(args.target, dtype=float))
    if args.target_rpy is not None:
        target_t[:3, :3] = rpy_matrix(*args.target_rpy)
    else:
        target_t[:3, :3] = home_pose[:3, :3]

    result = chain.solve_ik(
        target_t,
        home_q,
        max_iters=args.max_iters,
        q_home=home_q,
        tcp_transform=tool_t,
    )
    final_pose, _ = chain.fk(result.q, tool_t)
    print("\nIK result")
    print(f"  success: {result.success}")
    print(f"  iterations: {result.iterations}")
    print(f"  pos error: {result.pos_error_m:.8f} m")
    print(f"  rot error: {result.rot_error_rad:.8f} rad")
    print(f"  q: {vector_text(result.q)}")
    print(f"  final TCP xyz: {vector_text(final_pose[:3, 3])}")


if __name__ == "__main__":
    main()
