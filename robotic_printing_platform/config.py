"""Load planner configuration for bed placement, nozzle TCP, and IK settings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robotic_printing_platform.extrusion import MaterialProfile
from robotic_printing_platform.robots.franka_panda import IKConfig, PANDA_HOME
from robotic_printing_platform.robots.franka_panda_parameters import STANDARD_PANDA_JOINT_LIMITS_RAD


DEFAULT_CONFIG_PATH = Path("planner_config.json")


@dataclass(frozen=True)
class RobotConfig:
    model: str
    parameter_source: str
    home_q_rad: np.ndarray
    joint_limits_rad: np.ndarray
    max_reach_m: float


@dataclass(frozen=True)
class BedConfig:
    center_xyz_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    min_clearance_m: float


@dataclass(frozen=True)
class NozzleTCPConfig:
    flange_to_nozzle_xyz_m: tuple[float, float, float]
    flange_to_nozzle_rpy_rad: tuple[float, float, float]


@dataclass(frozen=True)
class PathPreparationConfig:
    max_seg_len_mm: float
    simplify_deg: float


@dataclass(frozen=True)
class MaterialConfig:
    profile: MaterialProfile


@dataclass(frozen=True)
class PlannerConfig:
    robot: RobotConfig
    bed: BedConfig
    nozzle_tcp: NozzleTCPConfig
    material: MaterialConfig
    path_preparation: PathPreparationConfig
    ik: dict[str, Any]

    def make_ik_config(
        self,
        *,
        ik_stride: int | None = None,
        max_waypoints: int | None = None,
    ) -> IKConfig:
        ik = dict(self.ik)
        if ik_stride is not None:
            ik["ik_stride"] = ik_stride
        if max_waypoints is not None:
            ik["max_waypoints"] = max_waypoints

        return IKConfig(
            pos_tol_m=float(ik.get("pos_tol_m", 0.008)),
            rot_tol_rad=float(ik.get("rot_tol_rad", 0.08)),
            max_iters=int(ik.get("max_iters", 180)),
            damping=float(ik.get("damping", 0.035)),
            orientation_weight=float(ik.get("orientation_weight", 0.35)),
            nullspace_weight=float(ik.get("nullspace_weight", 0.015)),
            max_joint_step_rad=float(ik.get("max_joint_step_rad", 0.10)),
            yaw_samples=int(ik.get("yaw_samples", 13)),
            joint_limits=self.robot.joint_limits_rad.copy(),
            q_home=self.robot.home_q_rad.copy(),
            bed_z_m=self.bed.center_xyz_m[2],
            min_clearance_m=self.bed.min_clearance_m,
            max_reach_m=self.robot.max_reach_m,
            ik_stride=int(ik.get("ik_stride", 1)),
            max_waypoints=ik.get("max_waypoints"),
            tool_tcp_xyz_m=self.nozzle_tcp.flange_to_nozzle_xyz_m,
            tool_tcp_rpy_rad=self.nozzle_tcp.flange_to_nozzle_rpy_rad,
        )


def _as_float_tuple(value: Any, n: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or len(value) != n:
        raise ValueError(f"{name} must be a list of {n} numbers")
    return tuple(float(v) for v in value)


def _as_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def load_planner_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PlannerConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    robot_data = data.get("robot", {})
    bed_data = data.get("bed", {})
    nozzle_data = data.get("nozzle_tcp", {})
    material_data = data.get("material", {})
    path_data = data.get("path_preparation", {})

    robot = RobotConfig(
        model=str(robot_data.get("model", "franka_panda")),
        parameter_source=str(robot_data.get("parameter_source", "franka_panda_parameters.py")),
        home_q_rad=_as_array(robot_data.get("home_q_rad", PANDA_HOME.tolist()), (7,), "robot.home_q_rad"),
        joint_limits_rad=_as_array(
            robot_data.get("joint_limits_rad", STANDARD_PANDA_JOINT_LIMITS_RAD.tolist()),
            (7, 2),
            "robot.joint_limits_rad",
        ),
        max_reach_m=float(robot_data.get("max_reach_m", 0.855)),
    )
    bed = BedConfig(
        center_xyz_m=_as_float_tuple(bed_data.get("center_xyz_m", [0.45, 0.0, 0.10]), 3, "bed.center_xyz_m"),
        normal=_as_float_tuple(bed_data.get("normal", [0.0, 0.0, 1.0]), 3, "bed.normal"),
        min_clearance_m=float(bed_data.get("min_clearance_m", 0.006)),
    )
    nozzle_tcp = NozzleTCPConfig(
        flange_to_nozzle_xyz_m=_as_float_tuple(
            nozzle_data.get("flange_to_nozzle_xyz_m", [0.0, 0.0, 0.115]),
            3,
            "nozzle_tcp.flange_to_nozzle_xyz_m",
        ),
        flange_to_nozzle_rpy_rad=_as_float_tuple(
            nozzle_data.get("flange_to_nozzle_rpy_rad", [0.0, 0.0, 0.0]),
            3,
            "nozzle_tcp.flange_to_nozzle_rpy_rad",
        ),
    )
    material = MaterialConfig(
        profile=MaterialProfile(
            name=str(material_data.get("name", "PLA")),
            filament_diameter_mm=float(material_data.get("filament_diameter_mm", 1.75)),
            flow_multiplier=float(material_data.get("flow_multiplier", 1.0)),
            density_g_cm3=(
                None
                if material_data.get("density_g_cm3", 1.24) is None
                else float(material_data.get("density_g_cm3", 1.24))
            ),
        )
    )
    path_preparation = PathPreparationConfig(
        max_seg_len_mm=float(path_data.get("max_seg_len_mm", 3.0)),
        simplify_deg=float(path_data.get("simplify_deg", 0.8)),
    )
    return PlannerConfig(
        robot=robot,
        bed=bed,
        nozzle_tcp=nozzle_tcp,
        material=material,
        path_preparation=path_preparation,
        ik=dict(data.get("ik", {})),
    )
