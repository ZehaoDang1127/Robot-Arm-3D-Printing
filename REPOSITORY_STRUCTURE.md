# Repository structure

```text
robotic-printing-platform/
|-- README.md
|-- REPOSITORY_STRUCTURE.md
|-- requirements.txt
|-- planner_config.json
|-- material_profiles/
|   |-- alginate_chitosan_pic_al1ch1_research.json
|   `-- pla.json
|-- outputs/
|   |-- README.md
|   |-- volumetric_patch.gcode
|   |-- panda/
|   |   `-- replay_isaac.py
|   |-- ur5/
|   |   `-- replay_isaac.py
|   `-- ur5e/
|       |-- replay_isaac.py
|       |-- resolved_material_profile.json
|       |-- robot_print_trajectory.csv
|       |-- robot_print_trajectory.json
|       `-- trajectory_validation_report.json
|-- run_pipeline.py
|-- visualize_pipeline.py
|-- analyze_urdf_ik.py
|-- robotic_printing_platform/
|   |-- config.py
|   |-- gcode/
|   |-- extrusion/
|   |   |-- deposition.py
|   |   `-- materials.py
|   |-- path_planning/
|   |-- trajectory/
|   |-- validation/
|   |-- exporters/
|   `-- robots/
|       `-- robot_configs/
|           |-- franka_panda/
|           |-- ur5/
|           `-- ur5e/
`-- test_*.py
```

## Universal material workflow

The pipeline, robot planner, Isaac exporter, and replay structure are not tied
to a named material. A material is selected through `--material PROFILE_ID` or
`planner_config.json`, loaded from `material_profiles/`, and copied into the
generated bundle as `resolved_material_profile.json`.

`outputs/` is both the normal pipeline destination and the stable, ready-to-run
example location. Bundles are classified only by robot (`panda`, `ur5`, or
`ur5e`), while the selected material is recorded as metadata inside each robot
bundle. Regenerating a robot with another compatible profile does not change
the directory layout or launch command. Input G-code must use the extrusion
convention declared by the selected profile.

## Main components

- `run_pipeline.py` parses G-code, prepares paths, solves robot IK, validates
  trajectories, and exports simulator artifacts.
- `material_profiles/` contains independently selectable material/process
  profiles. Adding a material does not require changing the pipeline code.
- `robotic_printing_platform/extrusion/materials.py` validates profiles and
  converts G-code extrusion values to volume and mass without hard-coding a
  selected material.
- `robotic_printing_platform/extrusion/deposition.py` integrates piecewise
  volumetric flow over measured TCP samples and dispatches bead segments or
  stationary droplets through a simulator-independent sink interface; it also
  provides the constant-parameter bead spreading/shrinkage model.
- `robotic_printing_platform/exporters/isaac.py` generates an Isaac Sim replay
  bundle that loads the selected profile and imports the central deposition
  module from the cloned repository at runtime, resolves the TCP from the USD
  hierarchy, and selects a curve or PBD particle sink. The curve sink updates
  unsettled widths and droplet radii on every replay step.
- `outputs/ur5e/resolved_material_profile.json` identifies the exact profile
  used for the committed replay snapshot.

## Generated outputs

Normal pipeline output is written under the directory supplied by
`--output-dir` and then separated by robot model. The canonical files in
`outputs/panda/`, `outputs/ur5/`, and `outputs/ur5e/` are intentionally tracked
as immediately launchable examples. Other directories under `outputs/`, and
directories with other `outputs*` names, remain ignored because they are
reproducible and may be large.
