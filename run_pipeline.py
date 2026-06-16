"""End-to-end runner for Cura G-code to Franka/Isaac trajectory export."""

from __future__ import annotations

import argparse
from pathlib import Path

from export_isaac import export_isaac_bundle
from gcode_parser import parse_gcode
from stage2_pathprep import build_waypoints
from stage3_franka_ik import IKConfig, solve_path_ik
from visualize_pipeline import write_all_plots


DEFAULT_GCODE = "strong_universal_wall_hook_vcd.gcode"


def run(
    path: str | Path = DEFAULT_GCODE,
    lo: int = 0,
    hi: int = 1,
    *,
    max_seg_len_mm: float = 3.0,
    simplify_deg: float = 0.8,
    bed_x_m: float = 0.45,
    bed_y_m: float = 0.0,
    bed_z_m: float = 0.10,
    output_dir: str | Path = "outputs",
    skip_ik: bool = False,
    ik_stride: int = 1,
    max_ik_waypoints: int | None = None,
):
    path = Path(path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    res = parse_gcode(path)
    print("=== Stage 1: parse ===")
    print(res.summary())
    print()

    hi = min(hi, res.layer_count)
    pp = build_waypoints(
        res,
        layers=(lo, hi),
        max_seg_len_mm=max_seg_len_mm,
        simplify_deg=simplify_deg,
        bed_center_xy_m=(bed_x_m, bed_y_m),
        bed_z_m=bed_z_m,
    )
    print(f"=== Stage 2: path prep (layers {lo}..{hi - 1}) ===")
    print(pp.summary())
    print()

    traj = None
    bundle = {}
    if not skip_ik:
        print("=== Stage 3: Franka IK / yaw optimization ===")
        traj = solve_path_ik(
            pp,
            IKConfig(
                bed_z_m=bed_z_m,
                ik_stride=ik_stride,
                max_waypoints=max_ik_waypoints,
            ),
        )
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
    parser.add_argument("--max-seg-len-mm", type=float, default=3.0, help="densified waypoint spacing")
    parser.add_argument("--simplify-deg", type=float, default=0.8, help="collinear simplification angle")
    parser.add_argument("--bed-x-m", type=float, default=0.45, help="bed center X in robot base frame")
    parser.add_argument("--bed-y-m", type=float, default=0.0, help="bed center Y in robot base frame")
    parser.add_argument("--bed-z-m", type=float, default=0.10, help="bed height in robot base frame")
    parser.add_argument("--output-dir", default="outputs", help="directory for plots and exports")
    parser.add_argument("--skip-ik", action="store_true", help="only parse, prepare, and visualize waypoints")
    parser.add_argument("--ik-stride", type=int, default=1, help="solve every Nth waypoint for quick previews")
    parser.add_argument("--max-ik-waypoints", type=int, default=None, help="cap IK waypoints for smoke tests")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        args.gcode,
        args.lo,
        args.hi,
        max_seg_len_mm=args.max_seg_len_mm,
        simplify_deg=args.simplify_deg,
        bed_x_m=args.bed_x_m,
        bed_y_m=args.bed_y_m,
        bed_z_m=args.bed_z_m,
        output_dir=args.output_dir,
        skip_ik=args.skip_ik,
        ik_stride=args.ik_stride,
        max_ik_waypoints=args.max_ik_waypoints,
    )
