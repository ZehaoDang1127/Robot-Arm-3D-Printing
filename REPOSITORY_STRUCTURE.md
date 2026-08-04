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
|-- isaac_demo/
|   |-- README.md
|   |-- volumetric_patch.gcode
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

`isaac_demo/` is the stable, ready-to-run example location. Its committed
snapshot uses one profile, but regenerating it with another compatible profile
does not change the directory layout or launch command. Input G-code must use
the extrusion convention declared by the selected profile.

## Main components

- `run_pipeline.py` parses G-code, prepares paths, solves robot IK, validates
  trajectories, and exports simulator artifacts.
- `material_profiles/` contains independently selectable material/process
  profiles. Adding a material does not require changing the pipeline code.
- `robotic_printing_platform/extrusion/materials.py` validates profiles and
  converts G-code extrusion values to volume and mass without hard-coding a
  selected material.
- `robotic_printing_platform/exporters/isaac.py` generates a standalone Isaac
  Sim replay with the selected profile embedded.
- `isaac_demo/ur5e/resolved_material_profile.json` identifies the exact profile
  used for the committed replay snapshot.

## Generated outputs

Normal pipeline output is written under the directory supplied by
`--output-dir` and then separated by robot model. Directories matching
`outputs*` are ignored because their contents are reproducible and may be
large. The smaller `isaac_demo/` bundle is intentionally tracked so the lab
desktop has an immediately launchable example.
