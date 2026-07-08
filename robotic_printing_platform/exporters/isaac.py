"""
Export helpers for using the optimized trajectory in NVIDIA Isaac Sim.

The CSV/JSON files are simulator-agnostic. The generated Isaac script is a
small starting point: run it inside Isaac Sim's Python environment and point it
at the exported CSV. Asset paths vary across Isaac Sim releases, so the script
keeps them in one editable constant near the top.
"""

from __future__ import annotations

from pathlib import Path

from robotic_printing_platform.robots.franka_panda import RobotTrajectory


ISAAC_SCRIPT = r'''"""
Replay a Franka Panda joint trajectory exported by this project.

Run inside Isaac Sim, for example:
    ./python.sh replay_isaac.py

If your Isaac install stores the Franka USD elsewhere, edit FRANKA_USD below.
"""

import csv
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({{"headless": False}})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Gf, Sdf, UsdGeom

TRAJECTORY_CSV = Path(r"{trajectory_csv}")
FRANKA_USD = "/Isaac/Robots/Franka/franka.usd"
ROBOT_PRIM = "/World/Franka"
DEPOSITION_PRIM = "/World/PrintedMaterial"
DEPOSITION_ENABLED = True
DEPOSITION_EVERY_N_PRINT_POINTS = 1
MAX_DEPOSITION_MARKERS = 20000
BEAD_RADIUS_M = 0.0012
BEAD_COLOR = Gf.Vec3f(1.0, 0.28, 0.03)


def load_rows(path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({{
                "q": [float(row[f"q{{i}}"]) for i in range(1, 8)],
                "p": [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                "is_print": row.get("is_print", "0") == "1",
                "de": float(row.get("de") or 0.0),
                "volume_mm3": float(row.get("extrusion_volume_mm3") or 0.0),
            }})
    return rows


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


world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(FRANKA_USD, ROBOT_PRIM)
robot = Articulation(ROBOT_PRIM)
world.scene.add(robot)
world.reset()
ensure_deposition_root(world.stage)

trajectory = load_rows(TRAJECTORY_CSV)
frame_hold = 4
cursor = 0
last_row_index = -1
print_point_counter = 0
marker_count = 0

while simulation_app.is_running():
    world.step(render=True)
    if not trajectory:
        continue
    row_index = min(cursor // frame_hold, len(trajectory) - 1)
    row = trajectory[row_index]
    robot.set_joint_positions(row["q"])
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
    cursor += 1

simulation_app.close()
'''


def export_isaac_bundle(
    traj: RobotTrajectory,
    output_dir: str | Path = "outputs",
    basename: str = "franka_print",
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{basename}_trajectory.csv"
    json_path = out / f"{basename}_trajectory.json"
    script_path = out / "replay_isaac.py"

    traj.export_csv(csv_path)
    traj.export_json(json_path)
    script_path.write_text(ISAAC_SCRIPT.format(trajectory_csv=str(csv_path.resolve())))
    return {"csv": csv_path, "json": json_path, "isaac_script": script_path}
