# Pipeline outputs and Isaac Sim replay demo

This directory is both the normal pipeline output location and a
material-independent, ready-to-run example of the export workflow. The material
is selected by `--material PROFILE_ID`; it is not selected by the directory
name. Every generated robot bundle records the exact selection in
`<robot>/resolved_material_profile.json`; its universal replay script loads
those values at runtime.

The current UR5e bundle uses `alginate_chitosan_pic_al1ch1_research` with
volumetric extrusion and contains the complete first layer of the wall-hook
model. Its G-code `E` values were converted from 1.75 mm filament length to
cubic millimetres before planning. The included `volumetric_patch.gcode` remains
a compact 25 mm x 15 mm input for quick regeneration tests. When changing to a
profile with a different extrusion convention, supply compatible G-code.

The canonical `panda/`, `ur5/`, and `ur5e/` snapshots are tracked in the
repository. Each replay loads its robot assets from the repository or the
configured Isaac asset path. Other generated run directories remain ignored.
Replay scripts also import the central deposition implementation from the
clone. Scripts below this repository discover it automatically; set
`RPP_PROJECT_ROOT` to the clone root when launching an output stored elsewhere.

## Launch the current bundle on the lab desktop

Run these commands in PowerShell:

```powershell
cd "C:\Users\haim_\Desktop\Robot-Arm-3D-Printing"
git pull
$env:RPP_DEPOSITION_MODE = "visual"
$env:RPP_VISUAL_BEAD_GEOMETRY = "mesh"
$env:RPP_MESH_RING_SEGMENTS = "12"
cd "C:\isaac-sim"
.\python.bat "C:\Users\haim_\Desktop\Robot-Arm-3D-Printing\outputs\ur5e\replay_isaac.py"
```

The console prints the material profile ID, selected mesh backend, and
spreading/shrinkage settings. Check those lines to confirm which material and
visual geometry are running.

## Regenerate a compact bundle with any compatible material profile

Change the value after `--material` without changing the output directory or
the Isaac launch command:

```powershell
cd "C:\isaac-sim"
.\python.bat "C:\Users\haim_\Desktop\Robot-Arm-3D-Printing\run_pipeline.py" `
  "C:\Users\haim_\Desktop\Robot-Arm-3D-Printing\outputs\volumetric_patch.gcode" `
  --material "alginate_chitosan_pic_al1ch1_research" `
  --robot ur5e `
  --isaac-usd "C:\Users\haim_\Desktop\Robot-Arm-3D-Printing\UR5e_extruder.usd" `
  --lo 0 --hi 1 `
  --max-seg-len-mm 1 `
  --simplify-deg 0 `
  --ik-selection-mode greedy `
  --output-dir "C:\Users\haim_\Desktop\Robot-Arm-3D-Printing\outputs"
```

Available profiles are JSON files under `material_profiles/`. The same command
can use a different G-code path, material profile, robot, and output directory;
`run_pipeline.py` and the generated replay remain universal.

The safe launch keeps the optional isosurface renderer off while retaining the
full PhysX PBD particle simulation. Isosurface affects rendering only and can
be explicitly enabled with `RPP_PARTICLE_ISOSURFACE=1` on a compatible GPU and
driver.

## Current UR5e hook validation

- 13,099/13,099 inverse-kinematics waypoints solved
- 0 velocity-limit violations
- 0 acceleration-limit violations
- 0 collision warnings
- 2,526.91265 mm^3 total deposited volume
- 602.50 s estimated replay duration

PhysX PBD is a particle approximation of material deposition. It does not model
molecular behavior, thermal phase change, or chemical curing.
