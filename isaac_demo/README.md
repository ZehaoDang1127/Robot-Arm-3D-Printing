# Universal Isaac Sim replay demo

This directory is a material-independent, ready-to-run example of the normal
export workflow. The material is selected by `--material PROFILE_ID`; it is not
selected by the directory name. Every generated bundle records the exact
selection in `ur5e/resolved_material_profile.json` and embeds the same values in
`ur5e/replay_isaac.py`.

The committed snapshot currently uses
`alginate_chitosan_pic_al1ch1_research` with volumetric extrusion. The included
`volumetric_patch.gcode` contains a 25 mm x 15 mm single-layer path whose `E`
values represent deposited volume. When changing to a profile with a different
extrusion convention, such as PLA filament length, supply compatible G-code.

Keep this directory in the repository layout because the replay loads the
robot and extruder USD assets from the repository root.

## Launch the current bundle on the lab desktop

Run these commands in PowerShell:

```powershell
cd "C:\Users\haim_\Desktop\Robot-Arm-3D-Printing"
git pull
$env:RPP_DEPOSITION_MODE = "particles"
$env:RPP_PARTICLE_ISOSURFACE = "0"
cd "C:\isaac-sim"
.\python.bat "C:\Users\haim_\Desktop\Robot-Arm-3D-Printing\isaac_demo\ur5e\replay_isaac.py"
```

The console prints the material profile ID, density, extrusion mode, and
particle settings. Check those lines to confirm which material is running.

## Regenerate with any compatible material profile

Change the value after `--material` without changing the output directory or
the Isaac launch command:

```powershell
cd "C:\isaac-sim"
.\python.bat "C:\Users\haim_\Desktop\Robot-Arm-3D-Printing\run_pipeline.py" `
  "C:\Users\haim_\Desktop\Robot-Arm-3D-Printing\isaac_demo\volumetric_patch.gcode" `
  --material "alginate_chitosan_pic_al1ch1_research" `
  --robot ur5e `
  --isaac-usd "C:\Users\haim_\Desktop\Robot-Arm-3D-Printing\UR5e_extruder.usd" `
  --lo 0 --hi 1 `
  --max-seg-len-mm 1 `
  --simplify-deg 0 `
  --ik-selection-mode greedy `
  --output-dir "C:\Users\haim_\Desktop\Robot-Arm-3D-Printing\isaac_demo"
```

Available profiles are JSON files under `material_profiles/`. The same command
can use a different G-code path, material profile, robot, and output directory;
`run_pipeline.py` and the generated replay remain universal.

The safe launch keeps the optional isosurface renderer off while retaining the
full PhysX PBD particle simulation. Isosurface affects rendering only and can
be explicitly enabled with `RPP_PARTICLE_ISOSURFACE=1` on a compatible GPU and
driver.

## Current snapshot validation

- 446/446 inverse-kinematics waypoints solved
- 0 velocity-limit violations
- 0 acceleration-limit violations
- 0 collision warnings
- 64 mm^3 total deposited volume
- 40.35 s estimated replay duration

PhysX PBD is a particle approximation of material deposition. It does not model
molecular behavior, thermal phase change, or chemical curing.
