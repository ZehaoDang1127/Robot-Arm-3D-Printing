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

TRAJECTORY_CSV = Path(r"{trajectory_csv}")
FRANKA_USD = "/Isaac/Robots/Franka/franka.usd"
ROBOT_PRIM = "/World/Franka"


def load_rows(path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append([float(row[f"q{{i}}"]) for i in range(1, 8)])
    return rows


world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(FRANKA_USD, ROBOT_PRIM)
robot = Articulation(ROBOT_PRIM)
world.scene.add(robot)
world.reset()

trajectory = load_rows(TRAJECTORY_CSV)
frame_hold = 4
cursor = 0

while simulation_app.is_running():
    world.step(render=True)
    if not trajectory:
        continue
    q = trajectory[min(cursor // frame_hold, len(trajectory) - 1)]
    robot.set_joint_positions(q)
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
