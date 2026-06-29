# Cura G-code to Franka Panda Printing Path

This repository now contains a Python pipeline for turning Cura/Marlin G-code
into a Franka Panda joint trajectory that can be inspected visually and replayed
in Isaac Sim.

## Files

- `gcode_parser.py` can parse Cura G-code into motion commands.
- `stage2_pathprep.py` can clean, simplify, densify, and transform paths into a
  Franka base-frame waypoint list.
- `stage3_franka_ik.py` solves Panda IK with yaw-redundancy optimization.
- `franka_panda_parameters.py` stores extracted Franka modified-DH and dynamic
  parameters from the referenced MATLAB model.
- `planner_config.json` stores robot home/limits, bed transform, nozzle TCP,
  path-prep defaults, and IK tolerances.
- `export_isaac.py` writes CSV/JSON trajectories and an Isaac Sim replay script.
- `visualize_pipeline.py` writes SVG path and joint-motion visualizations.
- `run_pipeline.py` runs the stages from one command.

Important limitation: the included IK is a lightweight NumPy implementation for
planning and simulation export. For real printing, calibrate the nozzle TCP,
bed transform, and Panda model in Isaac Sim or Pinocchio, then tighten the IK
tolerances before sending anything to hardware.

## Install

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

The only pip dependency is `numpy`. Isaac Sim is installed separately through
NVIDIA Omniverse or NVIDIA's Isaac Sim container.

## Quick Run

Run only parsing, path preparation, and visualization:

```bash
python run_pipeline.py strong_universal_wall_hook_vcd.gcode --lo 0 --hi 1 --skip-ik --output-dir outputs
```

Run a small IK/export smoke test:

```bash
python run_pipeline.py strong_universal_wall_hook_vcd.gcode --lo 0 --hi 1 --max-seg-len-mm 20 --simplify-deg 2 --max-ik-waypoints 30 --output-dir outputs
```

Run a coarser preview over more waypoints:

```bash
python run_pipeline.py strong_universal_wall_hook_vcd.gcode --lo 0 --hi 3 --max-seg-len-mm 10 --ik-stride 5 --output-dir outputs_preview
```

Use a different planner configuration:

```bash
python run_pipeline.py strong_universal_wall_hook_vcd.gcode --config planner_config.json --lo 0 --hi 1
```

Full-resolution IK over the whole model can be slow. Start with one layer,
large segment lengths, or `--ik-stride`, then refine once the placement and
solver tolerances look good.

## Outputs

The pipeline writes files under `--output-dir`.

- `gcode_path.svg`: top view of parsed Cura print and travel moves.
- `robot_waypoints.svg`: top view after placement into the Franka base frame.
- `joint_trajectory.svg`: per-step joint-motion plot after IK.
- `franka_print_trajectory.csv`: joint trajectory plus position, extrusion,
  layer, segment, and IK residual columns.
- `franka_print_trajectory.json`: same data in structured form.
- `replay_isaac.py`: starter Isaac Sim script for replaying the joint path.

Open the `.svg` files in a browser.

## Isaac Sim Replay

After generating outputs, open Isaac Sim's Python environment and run:

```bash
./python.sh outputs/replay_isaac.py
```

If Isaac cannot find the Franka asset, edit `FRANKA_USD` near the top of
`outputs/replay_isaac.py`. Isaac Sim asset paths vary by release.

## Pipeline Details

1. `gcode_parser.py`
   - Handles `G0/G1`, `G90/G91`, `M82/M83`, `G92`, units, Cura layer comments,
     feed rates, retractions, and travel moves.
   - Keeps coordinates in printer millimeters.

2. `stage2_pathprep.py`
   - Groups print and travel runs.
   - Simplifies near-collinear print vertices.
   - Densifies long segments.
   - Recenters the part on a virtual bed.
   - Converts printer millimeters to robot-base meters.

3. `stage3_franka_ik.py`
   - Uses the extracted modified-DH Panda model from
     `franka_panda_parameters.py` and damped least-squares IK.
   - Preserves print order for printability.
   - Optimizes robot motion by trying yaw candidates and choosing the solution
     closest to the previous joint state.
   - Checks joint limits, reach, and simple bed clearance.
   - Records residual position/orientation errors for every point.

`compare_franka_models.py` prints the difference between the previous inline
Stage-3 kinematic constants and the extracted modified-DH parameter model.

4. `export_isaac.py`
   - Exports simulator-friendly CSV/JSON.
   - Generates a simple Isaac Sim replay script.

## Coordinate Setup

The default bed and nozzle placement live in `planner_config.json`.

The default bed placement is:

- bed center: `(0.45, 0.0)` meters in the Franka base frame
- bed height: `0.10` meters

Change this with:

```bash
python run_pipeline.py strong_universal_wall_hook_vcd.gcode --bed-x-m 0.40 --bed-y-m 0.00 --bed-z-m 0.20
```

The `PathPrep.summary()` output reports reach from the robot base. Keep the
path well inside the Panda workspace before running IK.

## Planner Config

Edit `planner_config.json` when you want persistent calibrated values.

- `robot.home_q_rad`: nominal Panda seed posture.
- `robot.joint_limits_rad`: joint limits used by IK.
- `robot.max_reach_m`: reach warning threshold.
- `bed.center_xyz_m`: print-bed center in Franka base coordinates.
- `bed.min_clearance_m`: simple bed-clearance warning margin.
- `nozzle_tcp.flange_to_nozzle_xyz_m`: nozzle TCP translation from final Panda
  link frame.
- `nozzle_tcp.flange_to_nozzle_rpy_rad`: nozzle TCP roll, pitch, yaw.
- `path_preparation.max_seg_len_mm`: waypoint spacing after densification.
- `path_preparation.simplify_deg`: collinear simplification tolerance.
- `ik`: numerical IK tolerances and solver settings.

Command-line flags like `--bed-z-m`, `--max-seg-len-mm`, and `--ik-stride`
override the config for that run only.

## Reading IK Results

The IK summary can say `incomplete` while still generating trajectory points.
That means best-effort poses were exported, but some waypoints exceeded the
configured residual tolerance. Check these CSV columns:

- `pos_error_m`
- `rot_error_rad`

For physical printing, those errors must be much smaller than your acceptable
bead width and layer-height tolerance.
