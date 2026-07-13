# Repository Structure

```text
robotic-printing-platform/
├── README.md
├── REPOSITORY_STRUCTURE.md
├── requirements.txt
├── planner_config.json
├── run_pipeline.py
├── visualize_pipeline.py
├── analyze_urdf_ik.py
├── strong_universal_wall_hook_vcd.gcode
├── strong_universal_wall_hook_vcd.stl
├── robotic_printing_platform/
│   ├── __init__.py
│   ├── config.py
│   ├── gcode/
│   │   ├── __init__.py
│   │   └── parser.py
│   ├── extrusion/
│   │   ├── __init__.py
│   │   └── materials.py
│   ├── path_planning/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── layered.py
│   ├── robots/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── franka_panda.py
│   │   ├── urdf_kinematics.py
│   │   └── robot_configs/
│   │       └── franka_panda/
│   │           ├── README.md
│   │           ├── robot_config.json
│   │           └── robot.urdf
│   └── exporters/
│       ├── __init__.py
│       └── isaac.py
└── verify_sim/
    ├── franka_print_trajectory.csv
    ├── franka_print_trajectory.json
    ├── replay_isaac.py
    ├── gcode_path.svg
    ├── robot_waypoints.svg
    ├── robot_waypoints_xz.svg
    └── joint_trajectory.svg
```

## Top-Level Files

- `README.md` documents setup, usage, modular extension points, and Isaac visual deposition.
- `REPOSITORY_STRUCTURE.md` describes the repository layout.
- `requirements.txt` lists Python package dependencies.
- `planner_config.json` stores robot, bed, nozzle, path planning, IK, and material settings.
- `run_pipeline.py` is the main CLI entry point for parsing, planning, IK, export, and visualization.
- `visualize_pipeline.py` writes SVG diagnostics for G-code, robot waypoints, XZ waypoint side view, and joint motion.
- `analyze_urdf_ik.py` runs direct robot-folder FK, workspace, and IK analysis.
- `robotic_printing_platform/robots/robot_configs/` stores swappable robot packages. Replace the folder contents or point `robot.config_dir` at another folder to change robots.
- `strong_universal_wall_hook_vcd.gcode` is the sample sliced print path.
- `strong_universal_wall_hook_vcd.stl` is the sample source model.

## Python Package

### `robotic_printing_platform/gcode/`

Parses Cura/Marlin-style G-code into motion primitives.

- `parser.py` handles `X/Y/Z`, feedrate `F`, extrusion `E`, layer comments, absolute/relative positioning, retractions, and travel moves.

### `robotic_printing_platform/extrusion/`

Keeps extrusion and material behavior modular.

- `materials.py` defines `MaterialProfile` and converts G-code `E` deltas into volume and optional mass.

### `robotic_printing_platform/path_planning/`

Converts parsed G-code into robot-frame waypoints.

- `base.py` defines the `PathPlanningAlgorithm` interface.
- `layered.py` implements the default layer-by-layer planner, waypoint densification, bed placement, nozzle pose assignment, and extrusion metadata transfer.

### `robotic_printing_platform/robots/`

Converts robot-frame waypoints into robot-specific motion.

- `base.py` defines the `RobotPlanner` interface.
- `franka_panda.py` implements the default URDF-backed forward kinematics, IK, yaw sampling, trajectory export data, and `FrankaPandaPlanner`.
- `urdf_kinematics.py` provides general serial-chain URDF loading, FK, Jacobians, workspace sampling, and damped least-squares IK.

### `robotic_printing_platform/exporters/`

Writes simulator/runtime outputs.

- `isaac.py` exports CSV/JSON trajectories and generates `replay_isaac.py` for Isaac Sim, including visual deposition markers.

## Generated Verification Outputs

`verify_sim/` contains a small tracked simulation/export sample:

- `robot_print_trajectory.csv` and `.json` contain the generated robot trajectory and extrusion fields.
- `replay_isaac.py` replays Franka motion in Isaac Sim and creates visual deposited material markers.
- `gcode_path.svg` shows the parsed G-code path.
- `robot_waypoints.svg` shows robot waypoints in base-frame XY.
- `robot_waypoints_xz.svg` shows robot waypoints in base-frame XZ.
- `joint_trajectory.svg` shows per-step joint motion.

## Ignored Generated Outputs

The following all-layer output folders are intentionally ignored because they are large and reproducible:

```text
verify_all_layers/
verify_all_layers_preview/
outputs*/
```
