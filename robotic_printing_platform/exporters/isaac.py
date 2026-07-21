"""
Export helpers for using the optimized trajectory in NVIDIA Isaac Sim.

The CSV/JSON files are simulator-agnostic. The generated Isaac script is a
small starting point: run it inside Isaac Sim's Python environment and point it
at the exported CSV. Asset paths vary across Isaac Sim releases, so the script
keeps them in one editable constant near the top.
"""

from __future__ import annotations

import json
from pathlib import Path

from robotic_printing_platform.robots.generic import RobotTrajectory


ISAAC_SCRIPT = r'''"""
Replay a {robot_model} joint trajectory exported by this project.

Run inside Isaac Sim, for example:
    ./python.sh replay_isaac.py

If your Isaac install stores this robot's USD elsewhere, edit ROBOT_USD below.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp

simulation_app = SimulationApp({{"headless": False}})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Gf, Sdf, UsdGeom

TRAJECTORY_CSV = Path(r"{trajectory_csv}")
TRACKING_CSV = TRAJECTORY_CSV.with_name("joint_tracking.csv")
TRACKING_JSON = TRAJECTORY_CSV.with_name("joint_tracking_summary.json")
TRACKING_SVG = TRAJECTORY_CSV.with_name("joint_tracking.svg")
JOINT_COLUMNS = {joint_columns}
ROBOT_USD = {robot_usd_path!r}
ROBOT_PRIM = "/World/{robot_prim_name}"
DEPOSITION_PRIM = "/World/PrintedMaterial"
DEPOSITION_ENABLED = True
DEPOSITION_EVERY_N_PRINT_POINTS = 1
MAX_DEPOSITION_MARKERS = 20000
BEAD_RADIUS_M = 0.0012
BEAD_COLOR = Gf.Vec3f(1.0, 0.28, 0.03)
SETTLING_TIME_S = 2.0
TRACKING_PLOT_SAMPLE_STRIDE = 10


def load_rows(path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({{
                "q": [float(row[name]) for name in JOINT_COLUMNS],
                "p": [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                "time_from_start_s": float(row.get("time_from_start_s") or 0.0),
                "is_print": row.get("is_print", "0") == "1",
                "de": float(row.get("de") or 0.0),
                "volume_mm3": float(row.get("extrusion_volume_mm3") or 0.0),
            }})
    return sorted(rows, key=lambda row: row["time_from_start_s"])


def ensure_deposition_root(stage):
    if stage.GetPrimAtPath(DEPOSITION_PRIM):
        return
    UsdGeom.Xform.Define(stage, Sdf.Path(DEPOSITION_PRIM))


def spawn_deposition_marker(stage, marker_index, position, volume_mm3):
    """Create a visual bead marker at a deposited waypoint.

    This is intentionally visual-only. A later version can replace these
    spheres with cylinders/curves between waypoints or physics-enabled material.
    """
    prim_path = Sdf.Path(f"{{DEPOSITION_PRIM}}/bead_{{marker_index:06d}}")
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    radius = BEAD_RADIUS_M
    if volume_mm3 > 0.0:
        # Keep visual size stable, but let higher-flow segments read slightly thicker.
        radius *= max(0.6, min(1.8, (volume_mm3 / 0.28) ** (1.0 / 3.0)))
    sphere.CreateRadiusAttr(radius)
    sphere.CreateDisplayColorAttr([BEAD_COLOR])
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2]))


def interpolate_joint_target(rows, time_s, cursor):
    """Return q_desired(time_s) and the latest completed trajectory row."""
    if len(rows) == 1 or time_s <= rows[0]["time_from_start_s"]:
        return rows[0]["q"], 0
    while cursor + 1 < len(rows) and time_s >= rows[cursor + 1]["time_from_start_s"]:
        cursor += 1
    if cursor >= len(rows) - 1:
        return rows[-1]["q"], len(rows) - 1
    first = rows[cursor]
    second = rows[cursor + 1]
    dt = second["time_from_start_s"] - first["time_from_start_s"]
    alpha = 1.0 if dt <= 1e-9 else max(0.0, min(1.0, (time_s - first["time_from_start_s"]) / dt))
    return [a + alpha * (b - a) for a, b in zip(first["q"], second["q"])], cursor


def write_tracking_outputs(samples, sample_count, sum_squared_error, maximum_error):
    rms = math.sqrt(sum_squared_error / max(1, sample_count))
    TRACKING_JSON.write_text(json.dumps({{
        "samples": sample_count,
        "plot_samples": len(samples),
        "maximum_tracking_error_rad": maximum_error,
        "rms_tracking_error_rad": rms,
    }}, indent=2))
    if samples:
        write_tracking_svg(samples)
    print(f"tracking log: {{TRACKING_CSV}}")
    print(f"maximum tracking error: {{maximum_error:.6g}} rad")
    print(f"RMS tracking error: {{rms:.6g}} rad")


def write_tracking_svg(samples):
    """Write a dependency-free desired-vs-actual joint tracking plot."""
    width, left, right, panel_height = 1000, 80, 20, 125
    height = 40 + panel_height * len(JOINT_COLUMNS)
    t_end = max(samples[-1]["time_s"], 1e-6)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{{width}}" height="{{height}}" viewBox="0 0 {{width}} {{height}}">', '<rect width="100%" height="100%" fill="white"/>']
    for joint_index, name in enumerate(JOINT_COLUMNS):
        top = 25 + joint_index * panel_height
        values = [sample["desired"][joint_index] for sample in samples] + [sample["actual"][joint_index] for sample in samples]
        lower, upper = min(values), max(values)
        if upper - lower < 1e-6:
            lower -= 0.5
            upper += 0.5
        def points(key):
            return " ".join(
                f"{{left + (width - left - right) * sample['time_s'] / t_end:.2f}},{{top + 95 - 90 * (sample[key][joint_index] - lower) / (upper - lower):.2f}}"
                for sample in samples
            )
        parts.extend([
            f'<text x="5" y="{{top + 14}}" font-size="12">{{name}}</text>',
            f'<line x1="{{left}}" y1="{{top + 95}}" x2="{{width - right}}" y2="{{top + 95}}" stroke="#999"/>',
            f'<polyline points="{{points("desired")}}" fill="none" stroke="#1565c0" stroke-width="1.5"/>',
            f'<polyline points="{{points("actual")}}" fill="none" stroke="#ef6c00" stroke-width="1.5"/>',
        ])
    parts.append('<text x="80" y="18" font-size="12" fill="#1565c0">desired</text><text x="150" y="18" font-size="12" fill="#ef6c00">actual</text></svg>')
    TRACKING_SVG.write_text("\n".join(parts))


world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(ROBOT_USD, ROBOT_PRIM)
robot = Articulation(ROBOT_PRIM)
world.scene.add(robot)
world.reset()
ensure_deposition_root(world.stage)

trajectory = load_rows(TRAJECTORY_CSV)
controller = robot.get_articulation_controller()
joint_indices = []
for name in JOINT_COLUMNS:
    index = robot.get_dof_index(name)
    if index is None or index < 0:
        raise RuntimeError(f"robot does not expose required joint '{{name}}'")
    joint_indices.append(index)
joint_indices = np.asarray(joint_indices, dtype=int)
cursor = 0
last_row_index = -1
print_point_counter = 0
marker_count = 0
tracking_samples = []
tracking_sample_count = 0
sum_squared_error = 0.0
maximum_tracking_error = 0.0
physics_step_count = 0
fallback_time_s = 0.0
tracking_file = TRACKING_CSV.open("w", newline="")
tracking_writer = csv.writer(tracking_file)
tracking_writer.writerow([
    "simulation_time_s",
    *[f"desired_{{name}}" for name in JOINT_COLUMNS],
    *[f"actual_{{name}}" for name in JOINT_COLUMNS],
    *[f"error_{{name}}" for name in JOINT_COLUMNS],
])

try:
    while simulation_app.is_running():
        if not trajectory:
            world.step(render=True)
            continue
        try:
            simulation_time_s = float(world.current_time)
        except AttributeError:
            simulation_time_s = fallback_time_s
        q_desired, row_index = interpolate_joint_target(trajectory, simulation_time_s, cursor)
        cursor = row_index
        # Send targets to Isaac's articulation controller; do not teleport the arm.
        controller.apply_action(ArticulationAction(
            joint_positions=np.asarray(q_desired, dtype=float),
            joint_indices=joint_indices,
        ))
        world.step(render=True)
        fallback_time_s += world.get_physics_dt()
        actual_positions = robot.get_joint_positions(joint_indices=joint_indices)
        q_actual = actual_positions.tolist() if hasattr(actual_positions, "tolist") else list(actual_positions)
        error = [actual - desired for actual, desired in zip(q_actual, q_desired)]
        tracking_writer.writerow([simulation_time_s, *q_desired, *q_actual, *error])
        tracking_file.flush()
        tracking_sample_count += len(error)
        sum_squared_error += sum(value * value for value in error)
        maximum_tracking_error = max(maximum_tracking_error, max((abs(value) for value in error), default=0.0))
        if physics_step_count % TRACKING_PLOT_SAMPLE_STRIDE == 0:
            tracking_samples.append({{
                "time_s": simulation_time_s,
                "desired": q_desired,
                "actual": q_actual,
                "error": error,
            }})
        physics_step_count += 1
        if DEPOSITION_ENABLED and row_index > last_row_index:
            for deposition_row in trajectory[last_row_index + 1:row_index + 1]:
                if not deposition_row["is_print"] or deposition_row["de"] <= 0.0:
                    continue
                print_point_counter += 1
                if print_point_counter % DEPOSITION_EVERY_N_PRINT_POINTS != 0:
                    continue
                if marker_count >= MAX_DEPOSITION_MARKERS:
                    continue
                marker_count += 1
                spawn_deposition_marker(
                    world.stage,
                    marker_count,
                    deposition_row["p"],
                    deposition_row["volume_mm3"],
            )
            last_row_index = row_index
        if simulation_time_s >= trajectory[-1]["time_from_start_s"] + SETTLING_TIME_S:
            break
finally:
    tracking_file.close()
    write_tracking_outputs(
        tracking_samples,
        tracking_sample_count,
        sum_squared_error,
        maximum_tracking_error,
    )
    simulation_app.close()
'''


def export_isaac_bundle(
    traj: RobotTrajectory,
    output_dir: str | Path = "outputs",
    basename: str = "robot_print",
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{basename}_trajectory.csv"
    json_path = out / f"{basename}_trajectory.json"
    script_path = out / "replay_isaac.py"

    traj.export_csv(csv_path)
    traj.export_json(json_path)
    script_path.write_text(
        ISAAC_SCRIPT.format(
            trajectory_csv=str(csv_path.resolve()),
            joint_columns=json.dumps(traj.config.joint_names),
            robot_model=traj.config.robot_model,
            robot_usd_path=traj.config.isaac_usd_path,
            robot_prim_name="".join(c if c.isalnum() else "_" for c in traj.config.robot_model.title()),
        )
    )
    return {"csv": csv_path, "json": json_path, "isaac_script": script_path}
