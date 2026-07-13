# Robotic Printing Platform

This project converts Cura/Marlin G-code into robot-ready printing waypoints,
then solves a robot-specific trajectory for simulation or export. The code is
now organized so the parser, material/extrusion model, path planner, and robot
solver can evolve independently.

## Package Layout

- `robotic_printing_platform/gcode/` parses G-code motion, including explicit
  `E` extrusion words, absolute extruder state, per-move `de`, feedrate, layers,
  travels, and retractions.
- `robotic_printing_platform/extrusion/` converts raw `E` deltas into
  material-specific volume and optional mass using a swappable
  `MaterialProfile`.
- `robotic_printing_platform/path_planning/` contains the default
  `LayeredPathPlanner` plus the `PathPlanningAlgorithm` interface. Replace this
  module when you want a different ordering, smoothing, non-planar planning, or
  nozzle-normal strategy.
- `robotic_printing_platform/robots/` contains the `RobotPlanner` interface,
  URDF kinematics helpers, robot configuration folders, and the default
  URDF-backed planner. The default robot package is
  `robotic_printing_platform/robots/robot_configs/franka_panda/`.
- `robotic_printing_platform/exporters/` writes simulator/runtime artifacts.

Top-level scripts stay small: `run_pipeline.py` runs the workflow and
`visualize_pipeline.py` writes SVG diagnostics. `analyze_urdf_ik.py` runs direct
URDF FK/workspace/IK checks. Core implementation lives in the
`robotic_printing_platform/` package.

## Install

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

The runtime dependency is `numpy`. Isaac Sim is installed separately through
NVIDIA Omniverse or NVIDIA's Isaac Sim container.

## Quick Run

Run parsing, path preparation, and visualization:

```bash
python run_pipeline.py strong_universal_wall_hook_vcd.gcode --lo 0 --hi 1 --skip-ik --output-dir outputs
```

Run a small IK/export smoke test:

```bash
python run_pipeline.py strong_universal_wall_hook_vcd.gcode --lo 0 --hi 1 --max-seg-len-mm 20 --simplify-deg 2 --max-ik-waypoints 30 --output-dir outputs
```

Use a different configuration:

```bash
python run_pipeline.py strong_universal_wall_hook_vcd.gcode --config planner_config.json --lo 0 --hi 1
```

Run a direct robot-folder IK analysis:

```bash
python analyze_urdf_ik.py --robot-config-dir robotic_printing_platform/robots/robot_configs/franka_panda --samples 500 --target 0.45 0.0 0.25
```

Process every parsed layer:

```bash
python run_pipeline.py strong_universal_wall_hook_vcd.gcode --all-layers --skip-ik --output-dir verify_all_layers
```

For a coarse all-layer IK preview, keep the full path but sample the IK solve:

```bash
python run_pipeline.py strong_universal_wall_hook_vcd.gcode --all-layers --max-seg-len-mm 20 --simplify-deg 2 --ik-stride 5000 --output-dir verify_all_layers_preview
```

## Changing Material

Edit the `material` section in `planner_config.json`:

```json
{
  "material": {
    "name": "PLA",
    "filament_diameter_mm": 1.75,
    "flow_multiplier": 1.0,
    "density_g_cm3": 1.24
  }
}
```

The parser preserves raw G-code extrusion as `Move.e`, `Move.de`, and
`Move.has_e`. The planner maps `de` into each waypoint's
`extrusion_volume_mm3`, `extrusion_mass_g`, and `material`.

## Changing Path Planning

Implement `robotic_printing_platform.path_planning.base.PathPlanningAlgorithm`
and return a `PathPrep`-compatible object. The default implementation is
`LayeredPathPlanner`, which groups print/travel runs, simplifies collinear print
vertices, densifies long segments, places the path on the bed, and assigns a
planar downward nozzle axis.

## Changing Robot

Implement `robotic_printing_platform.robots.base.RobotPlanner`. The current
planner loads its arm geometry from the folder named by
`planner_config.json -> robot.config_dir`, wraps the NumPy URDF IK solver, and
exports joint trajectories. To swap from Franka Panda to UR5 or another serial
arm, copy `robotic_printing_platform/robots/robot_configs/franka_panda/`,
replace `robot.urdf`, edit `robot_config.json`, then point `robot.config_dir`
at the new folder.

Important limitation: the included Franka IK is a lightweight NumPy
implementation for planning and simulation export. For real printing, calibrate
the nozzle TCP, bed transform, and robot model in your production stack before
sending anything to hardware.

## Outputs

The pipeline writes files under `--output-dir`.

- `gcode_path.svg`: top view of parsed print and travel moves.
- `robot_waypoints.svg`: top view after placement in robot base coordinates.
- `robot_waypoints_xz.svg`: side view in robot base XZ coordinates.
- `joint_trajectory.svg`: per-step joint-motion plot after IK.
- `robot_print_trajectory.csv`: joint trajectory plus position, feed,
  extrusion, material, layer, segment, and IK residual columns.
- `robot_print_trajectory.json`: same data in structured form.
- `replay_isaac.py`: starter Isaac Sim replay script with visual deposition
  markers for print moves.

## Isaac Visual Deposition

The generated `replay_isaac.py` now creates visual bead markers while the robot
replays print waypoints. It reads `is_print`, `de`, and
`extrusion_volume_mm3` from the exported CSV, then spawns orange spheres at
deposition points.

This is visual-only deposition. It does not simulate thermal behavior, flow,
cooling, bead contact, or the mechanics of a growing part. The deposition
settings live near the top of the generated Isaac script:

- `DEPOSITION_ENABLED`
- `DEPOSITION_EVERY_N_PRINT_POINTS`
- `MAX_DEPOSITION_MARKERS`
- `BEAD_RADIUS_M`
