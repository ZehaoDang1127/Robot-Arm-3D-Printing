"""End-to-end runner for Cura G-code to robot/Isaac trajectory export."""

from __future__ import annotations

import argparse
from pathlib import Path

from robotic_printing_platform.config import DEFAULT_CONFIG_PATH, load_planner_config
from robotic_printing_platform.exporters.isaac import export_isaac_bundle
from robotic_printing_platform.gcode import parse_gcode
from robotic_printing_platform.path_planning import LayeredPathPlanner
from robotic_printing_platform.robots import URDFRobotPlanner
from visualize_pipeline import write_all_plots


DEFAULT_GCODE = "strong_universal_wall_hook_vcd.gcode"


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
    all_layers: bool = False,
):
    path = Path(path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_planner_config(config_path)

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

    traj = None
    bundle = {}
    if not skip_ik:
        print("=== Stage 3: robot IK / yaw optimization ===")
        robot_planner = URDFRobotPlanner(
            cfg.make_ik_config(ik_stride=ik_stride, max_waypoints=max_ik_waypoints)
        )
        traj = robot_planner.solve(pp)
        print(traj.report.summary())
        if traj.report.warnings:
            print("first warnings:")
            for warning in traj.report.warnings[:5]:
                print(f"  - {warning}")
        print()

        print("=== Stage 4: export ===")
        bundle = export_isaac_bundle(traj, out)
        for name, file_path in bundle.items():
            print(f"{name}: {file_path}")
        print()

    print("=== Visualization ===")
    plots = write_all_plots(res, pp, traj, out)
    for name, file_path in plots.items():
        print(f"{name}: {file_path}")

    return res, pp, traj, bundle, plots


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
        all_layers=args.all_layers,
    )
