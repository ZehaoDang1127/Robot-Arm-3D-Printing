# Hydrogel Isaac Sim demo

This bundle replays a validated 25 mm x 15 mm single-layer patch using the
`alginate_chitosan_pic_al1ch1_research` material profile. Extrusion is encoded
as deposited volume: the 16 print lines deposit 64 mm³ in total.

Keep this directory in the repository layout because the replay script loads
the robot and extruder USD assets from the repository root.

## Launch on the lab desktop

From the repository root in PowerShell, replace the Isaac Sim path with the
installation path on that computer:

```powershell
$env:RPP_DEPOSITION_MODE = "particles"
$env:RPP_PARTICLE_ISOSURFACE = "1"
& "C:\PATH\TO\ISAAC_SIM\python.bat" ".\hydrogel_demo\ur5e\replay_isaac.py"
```

The replay console prints the selected material profile, extrusion mode,
hydrogel density, and particle-system settings at startup. The complete
resolved profile is also saved in `ur5e/resolved_material_profile.json`.

## Validation summary

- 446/446 inverse-kinematics waypoints solved
- 0 velocity-limit violations
- 0 acceleration-limit violations
- 0 collision warnings
- 64 mm³ total deposited volume
- 40.35 s estimated replay duration

This is a particle/isosurface approximation of hydrogel deposition in PhysX;
it is not a molecular or chemical curing simulation.
