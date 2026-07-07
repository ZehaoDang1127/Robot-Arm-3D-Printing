"""
Replay a Franka Panda joint trajectory exported by this project.

Run inside Isaac Sim, for example:
    ./python.sh replay_isaac.py

If your Isaac install stores the Franka USD elsewhere, edit FRANKA_USD below.
"""

import csv
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage

TRAJECTORY_CSV = Path(r"D:\HAIM_Lab\robotic-printing-platform\verify_sim\franka_print_trajectory.csv")
FRANKA_USD = "/Isaac/Robots/Franka/franka.usd"
ROBOT_PRIM = "/World/Franka"


def load_rows(path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append([float(row[f"q{i}"]) for i in range(1, 8)])
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
