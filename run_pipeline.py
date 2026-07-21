"""End-to-end runner for Cura G-code to robot/Isaac trajectory export."""

from __future__ import annotations

import argparse
from pathlib import Path

from robotic_printing_platform.config import DEFAULT_CONFIG_PATH, load_planner_config
from robotic_printing_platform.exporters.isaac import export_isaac_bundle
from robotic_printing_platform.gcode import parse_gcode
from robotic_printing_platform.path_planning import LayeredPathPlanner
from robotic_printing_platform.robots.generic import URDFRobotPlanner
from robotic_printing_platform.trajectory import retime_trajectory
from robotic_printing_platform.validation import validate_trajectory
from robotic_printing_platform.validation import sweep_position_tolerances
from visualize_pipeline import write_all_plots


DEFAULT_GCODE = "strong_universal_wall_hook_vcd.gcode"
ROBOT_CONFIG_DIRS = {
    "panda": "robotic_printing_platform/robots/robot_configs/franka_panda",
    "ur5": "robotic_printing_platform/robots/robot_configs/ur5",
}


def run(
    path: str | Path = DEFAULT_GCODE,
    lo: int = 0,
    hi: int = 1,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    max_seg_len_mm: float | None = None,
    simplify_deg: float | None = None,
    bed_x_m: float | None = None,
    bed_y_m: float | None = None,
    bed_z_m: float | None = None,
    output_dir: str | Path = "outputs",
    skip_ik: bool = False,
    ik_stride: int | None = None,
    max_ik_waypoints: int | None = None,
    ik_selection_mode: str | None = None,
    position_tolerance_sweep_mm: list[float] | None = None,
    all_layers: bool = False,
    robot: str = "panda",
):
    path = Path(path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_planner_config(config_path)
    if robot not in {"panda", "ur5", "both", "config"}:
        raise ValueError("robot must be one of: panda, ur5, both, config")

    bed_x = cfg.bed.center_xyz_m[0] if bed_x_m is None else bed_x_m
    bed_y = cfg.bed.center_xyz_m[1] if bed_y_m is None else bed_y_m
    bed_z = cfg.bed.center_xyz_m[2] if bed_z_m is None else bed_z_m
    seg_len = cfg.path_preparation.max_seg_len_mm if max_seg_len_mm is None else max_seg_len_mm
    simplify = cfg.path_preparation.simplify_deg if simplify_deg is None else simplify_deg

    res = parse_gcode(path)
    print("=== Stage 1: parse ===")
    print(res.summary())
    print()

    if all_layers:
        lo = 0
        hi = res.layer_count
    else:
        hi = min(hi, res.layer_count)
    path_planner = LayeredPathPlanner(
        max_seg_len_mm=seg_len,
        simplify_deg=simplify,
        bed_center_xy_m=(bed_x, bed_y),
        bed_z_m=bed_z,
        material_profile=cfg.material.profile,
    )
    pp = path_planner.build(res, layers=(lo, hi))
    print(f"=== Stage 2: path prep (layers {lo}..{hi - 1}) ===")
    print(pp.summary())
    print()

    selected_robots = ("panda", "ur5") if robot == "both" else (robot,)
    trajectories = {}
    bundles = {}
    plots = {}

    for robot_key in selected_robots:
        robot_cfg = (
            cfg
            if robot_key == "config"
            else load_planner_config(config_path, robot_config_dir=ROBOT_CONFIG_DIRS[robot_key])
        )
        output_name = "panda" if robot_key == "panda" else robot_cfg.robot.model
        robot_out = out / output_name
        traj = None

        print(f"=== {robot_cfg.robot.model}: output {robot_out} ===")
        if position_tolerance_sweep_mm:
            sweep_cfg = robot_cfg.make_ik_config(
                ik_stride=ik_stride,
                max_waypoints=max_ik_waypoints,
            )
            if ik_selection_mode is not None:
                sweep_cfg.ik_selection_mode = ik_selection_mode
            tolerance_sweep = sweep_position_tolerances(
                pp,
                sweep_cfg,
                position_tolerance_sweep_mm,
            )
            tolerance_sweep_path = tolerance_sweep.write_json(robot_out / "ik_tolerance_sweep.json")
            print(tolerance_sweep.summary())
            print(f"tolerance sweep: {tolerance_sweep_path}")
            print()

        if not skip_ik:
            print("=== Stage 3: robot IK / yaw optimization ===")
            robot_planner = URDFRobotPlanner(
                robot_cfg.make_ik_config(ik_stride=ik_stride, max_waypoints=max_ik_waypoints)
            )
            if ik_selection_mode is not None:
                robot_planner.config.ik_selection_mode = ik_selection_mode
            traj = robot_planner.solve(pp)
            traj = retime_trajectory(
                traj,
                robot_cfg.robot.joint_velocity_limits_rad_s,
                robot_cfg.robot.joint_acceleration_limits_rad_s2,
            )
            trajectories[robot_cfg.robot.model] = traj
            print(traj.report.summary())
            if traj.report.warnings:
                print("first warnings:")
                for warning in traj.report.warnings[:5]:
                    print(f"  - {warning}")
            print()

            validation = validate_trajectory(traj)
            validation_path = validation.write_json(robot_out / "trajectory_validation_report.json")
            print("=== Trajectory validation ===")
            print(validation.summary())
            print(f"validation report: {validation_path}")
            print()

            print("=== Stage 4: export ===")
            bundle = export_isaac_bundle(traj, robot_out)
            bundle["validation_report"] = validation_path
            bundles[robot_cfg.robot.model] = bundle
            for name, file_path in bundle.items():
                print(f"{name}: {file_path}")
            print()

        print("=== Visualization ===")
        robot_plots = write_all_plots(res, pp, traj, robot_out)
        plots[robot_cfg.robot.model] = robot_plots
        for name, file_path in robot_plots.items():
            print(f"{name}: {file_path}")

    return res, pp, trajectories, bundles, plots


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gcode", nargs="?", default=DEFAULT_GCODE, help="Cura/Marlin G-code file")
    parser.add_argument("--lo", type=int, default=0, help="first layer to process, inclusive")
    parser.add_argument("--hi", type=int, default=1, help="last layer to process, exclusive")
    parser.add_argument("--all-layers", action="store_true", help="process every parsed layer")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="planner JSON config file")
    parser.add_argument("--max-seg-len-mm", type=float, default=None, help="override densified waypoint spacing")
    parser.add_argument("--simplify-deg", type=float, default=None, help="override collinear simplification angle")
    parser.add_argument("--bed-x-m", type=float, default=None, help="override bed center X in robot base frame")
    parser.add_argument("--bed-y-m", type=float, default=None, help="override bed center Y in robot base frame")
    parser.add_argument("--bed-z-m", type=float, default=None, help="override bed height in robot base frame")
    parser.add_argument("--output-dir", default="outputs", help="directory for plots and exports")
    parser.add_argument("--skip-ik", action="store_true", help="only parse, prepare, and visualize waypoints")
    parser.add_argument("--ik-stride", type=int, default=None, help="override solve every Nth waypoint")
    parser.add_argument("--max-ik-waypoints", type=int, default=None, help="cap IK waypoints for smoke tests")
    parser.add_argument(
        "--ik-selection-mode",
        choices=["greedy", "global_dp"],
        default=None,
        help="override the configured local or global yaw/IK selection mode",
    )
    parser.add_argument(
        "--position-tolerance-sweep-mm",
        type=float,
        nargs="+",
        default=None,
        metavar="MM",
        help="run an IK convergence sweep, e.g. 8 5 3 2 1",
    )
    parser.add_argument(
        "--robot",
        choices=["panda", "ur5", "both", "config"],
        default="panda",
        help="robot package(s) to run; both writes separate panda/ and ur5/ outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        args.gcode,
        args.lo,
        args.hi,
        config_path=args.config,
        max_seg_len_mm=args.max_seg_len_mm,
        simplify_deg=args.simplify_deg,
        bed_x_m=args.bed_x_m,
        bed_y_m=args.bed_y_m,
        bed_z_m=args.bed_z_m,
        output_dir=args.output_dir,
        skip_ik=args.skip_ik,
        ik_stride=args.ik_stride,
        max_ik_waypoints=args.max_ik_waypoints,
        ik_selection_mode=args.ik_selection_mode,
        position_tolerance_sweep_mm=args.position_tolerance_sweep_mm,
        all_layers=args.all_layers,
        robot=args.robot,
    )
