"""
Export helpers for using the optimized trajectory in NVIDIA Isaac Sim.

The CSV/JSON files are simulator-agnostic. The generated Isaac script is a
small starting point: run it inside Isaac Sim's Python environment and point it
at the exported CSV. Asset paths vary across Isaac Sim releases, so the script
keeps them in one editable constant near the top.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from robotic_printing_platform.robots.generic import RobotTrajectory


ISAAC_SCRIPT = r'''"""
Replay a {robot_model} joint trajectory exported by this project.

Run inside Isaac Sim, for example:
    ./python.sh replay_isaac.py

Set RPP_ROBOT_USD to override the robot asset path at launch time.
"""

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp

simulation_app = SimulationApp({{"headless": False}})

try:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:  # Isaac Sim 4.x compatibility
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation as SingleArticulation
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.types import ArticulationAction
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

SCRIPT_DIR = Path(__file__).resolve().parent
TRAJECTORY_CSV = Path(
    os.environ.get("RPP_TRAJECTORY_CSV", SCRIPT_DIR / {trajectory_filename!r})
).resolve()
TRACKING_CSV = TRAJECTORY_CSV.with_name("joint_tracking.csv")
TRACKING_JSON = TRAJECTORY_CSV.with_name("joint_tracking_summary.json")
TRACKING_SVG = TRAJECTORY_CSV.with_name("joint_tracking.svg")
JOINT_COLUMNS = {joint_columns}
ROBOT_USD_RELATIVE = {robot_usd_relative!r}
ROBOT_USD_DEFAULT = (
    str((SCRIPT_DIR / ROBOT_USD_RELATIVE).resolve())
    if ROBOT_USD_RELATIVE
    else {robot_usd_fallback!r}
)
ROBOT_USD = os.environ.get("RPP_ROBOT_USD", ROBOT_USD_DEFAULT)
ROBOT_PRIM = "/World/{robot_prim_name}"
DEPOSITION_PRIM = "/World/PrintedMaterial"
DEPOSITION_ENABLED = True
DEPOSITION_EVERY_N_PRINT_POINTS = 1
MAX_DEPOSITION_MARKERS = 20000
BEAD_RADIUS_M = 0.0012
BEAD_COLOR = Gf.Vec3f(1.0, 0.28, 0.03)
SETTLING_TIME_S = 2.0
INITIALIZATION_TIMEOUT_S = 10.0
INITIALIZATION_TOLERANCE_RAD = 0.02
INITIALIZATION_STABLE_STEPS = 10
DEPOSITION_MAX_JOINT_ERROR_RAD = 0.05
MAX_ACCEPTABLE_TRACKING_ERROR_RAD = 0.05
MAX_ACCEPTABLE_RMS_TRACKING_ERROR_RAD = 0.02
TRACKING_PLOT_SAMPLE_STRIDE = 10


def enable_robot_physics_variants(stage, reference_prim_path):
    """Select PhysX on referenced robot assets that were saved as visual-only."""
    reference_prim = stage.GetPrimAtPath(reference_prim_path)
    if not reference_prim.IsValid():
        raise RuntimeError(f"robot reference prim does not exist: {{reference_prim_path}}")
    subtree = list(Usd.PrimRange(reference_prim, Usd.TraverseInstanceProxies()))
    physics_variant_prims = [
        prim
        for prim in subtree
        if "Physics" in prim.GetVariantSets().GetNames()
    ]
    for prim in physics_variant_prims:
        variant_set = prim.GetVariantSets().GetVariantSet("Physics")
        variant_names = variant_set.GetVariantNames()
        if "PhysX" not in variant_names:
            continue
        previous = variant_set.GetVariantSelection()
        if previous == "PhysX":
            continue
        if prim.IsInstanceProxy():
            raise RuntimeError(
                f"cannot select the PhysX variant on instance proxy {{prim.GetPath()}}"
            )
        if not variant_set.SetVariantSelection("PhysX"):
            raise RuntimeError(
                f"failed to select the PhysX variant on {{prim.GetPath()}}; "
                f"available variants: {{variant_names}}"
            )
        print(
            f"enabled robot physics variant: {{prim.GetPath()}} "
            f"Physics={{previous!r}} -> 'PhysX'"
        )


def repair_extruder_mount(stage, reference_prim_path):
    """Repair the known mount payload's mesh metadata and collider in memory."""
    mount_path = f"{{reference_prim_path}}/ur5_mount_extruder"
    mesh_path = f"{{mount_path}}/MeshBody1"
    mount_prim = stage.GetPrimAtPath(mount_path)
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mount_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh):
        print(f"extruder mount repair skipped; mesh not found: {{mesh_path}}")
        return

    mesh = UsdGeom.Mesh(mesh_prim)
    normals = mesh.GetNormalsAttr().Get() or []
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get() or []
    expected_face_varying_count = sum(int(count) for count in face_vertex_counts)
    if (
        len(normals) > 0
        and len(normals) == expected_face_varying_count
        and mesh.GetNormalsInterpolation() != UsdGeom.Tokens.faceVarying
    ):
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        print(f"repaired mount normals interpolation: {{mesh_path}} -> faceVarying")

    collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
    collision_api.CreateCollisionEnabledAttr(True)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision_api.CreateApproximationAttr().Set("convexHull")
    print(f"enabled mount convex-hull collision: {{mesh_path}}")

    mass_override = os.environ.get("RPP_MOUNT_MASS_KG")
    if mass_override:
        try:
            mass_kg = float(mass_override)
        except ValueError as error:
            raise RuntimeError("RPP_MOUNT_MASS_KG must be a positive number") from error
        if not math.isfinite(mass_kg) or mass_kg <= 0.0:
            raise RuntimeError("RPP_MOUNT_MASS_KG must be a positive number")
        mass_api = UsdPhysics.MassAPI.Apply(mount_prim)
        mass_api.CreateMassAttr().Set(mass_kg)
        print(f"set measured mount/extruder mass: {{mass_kg:.6g}} kg")
    else:
        print("mount mass: automatic from collider (set RPP_MOUNT_MASS_KG to override)")


def find_or_create_articulation_root(stage, reference_prim_path):
    """Find an articulation root, or mark the assembly root when omitted.

    Custom robot USDs commonly wrap the actual articulation below a stage or
    assembly Xform. Some Robot Assembler exports retain all PhysicsJoint prims
    but omit PhysicsArticulationRootAPI. For those assets, marking the assembly
    Xform is valid for both fixed-base and floating articulations.
    """
    reference_prim = stage.GetPrimAtPath(reference_prim_path)
    if not reference_prim.IsValid():
        raise RuntimeError(f"robot reference prim does not exist: {{reference_prim_path}}")
    # Instanceable robot references expose their descendants as instance
    # proxies, which ordinary Usd.PrimRange traversal intentionally skips.
    subtree = list(Usd.PrimRange(reference_prim, Usd.TraverseInstanceProxies()))
    roots = [
        str(prim.GetPath())
        for prim in subtree
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(roots) > 1:
        raise RuntimeError(
            f"expected exactly one articulation root below {{reference_prim_path}}, "
            f"found {{len(roots)}}: {{roots}}"
        )
    if roots:
        return roots[0]

    joints = [
        str(prim.GetPath())
        for prim in subtree
        if prim.IsA(UsdPhysics.Joint)
    ]
    movable_joints = [
        str(prim.GetPath())
        for prim in subtree
        if prim.IsA(UsdPhysics.RevoluteJoint)
        or prim.IsA(UsdPhysics.PrismaticJoint)
    ]
    if len(movable_joints) < len(JOINT_COLUMNS):
        raise RuntimeError(
            f"robot asset below {{reference_prim_path}} exposes only "
            f"{{len(movable_joints)}} movable physics joints (expected at least "
            f"{{len(JOINT_COLUMNS)}}). Refusing to mark a mount or rigid body as "
            f"the robot articulation. Joints found: {{joints}}; asset: {{ROBOT_USD}}"
        )
    articulation_api = UsdPhysics.ArticulationRootAPI.Apply(reference_prim)
    if not articulation_api:
        raise RuntimeError(
            f"could not apply PhysicsArticulationRootAPI to {{reference_prim_path}}"
        )
    print(
        f"asset had no articulation root marker; applied one to "
        f"{{reference_prim_path}} (found {{len(joints)}} physics joints)"
    )
    return reference_prim_path


def load_rows(path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({{
                "q": [float(row[name]) for name in JOINT_COLUMNS],
                "p": [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                "time_from_start_s": float(row.get("time_from_start_s") or 0.0),
                "is_print": row.get("is_print", "0") == "1",
                "de": float(row.get("de") or 0.0),
                "volume_mm3": float(row.get("extrusion_volume_mm3") or 0.0),
            }})
    return sorted(rows, key=lambda row: row["time_from_start_s"])


def ensure_deposition_root(stage):
    if stage.GetPrimAtPath(DEPOSITION_PRIM):
        return
    UsdGeom.Xform.Define(stage, Sdf.Path(DEPOSITION_PRIM))


def spawn_deposition_marker(stage, marker_index, position, volume_mm3):
    """Create a visual bead marker at a deposited waypoint.

    This is intentionally visual-only. A later version can replace these
    spheres with cylinders/curves between waypoints or physics-enabled material.
    """
    prim_path = Sdf.Path(f"{{DEPOSITION_PRIM}}/bead_{{marker_index:06d}}")
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    radius = BEAD_RADIUS_M
    if volume_mm3 > 0.0:
        # Keep visual size stable, but let higher-flow segments read slightly thicker.
        radius *= max(0.6, min(1.8, (volume_mm3 / 0.28) ** (1.0 / 3.0)))
    sphere.CreateRadiusAttr(radius)
    sphere.CreateDisplayColorAttr([BEAD_COLOR])
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2]))


def interpolate_joint_target(rows, time_s, cursor):
    """Return q_desired(time_s) and the latest completed trajectory row."""
    if len(rows) == 1 or time_s <= rows[0]["time_from_start_s"]:
        return rows[0]["q"], 0
    while cursor + 1 < len(rows) and time_s >= rows[cursor + 1]["time_from_start_s"]:
        cursor += 1
    if cursor >= len(rows) - 1:
        return rows[-1]["q"], len(rows) - 1
    first = rows[cursor]
    second = rows[cursor + 1]
    dt = second["time_from_start_s"] - first["time_from_start_s"]
    alpha = 1.0 if dt <= 1e-9 else max(0.0, min(1.0, (time_s - first["time_from_start_s"]) / dt))
    return [a + alpha * (b - a) for a, b in zip(first["q"], second["q"])], cursor


def initialize_at_first_target(world, robot, controller, joint_indices, initial_q):
    """Place the arm at q[0], then settle before starting the replay clock."""
    initial_q = np.asarray(initial_q, dtype=float)
    robot.set_joint_positions(initial_q, joint_indices=joint_indices)
    robot.set_joint_velocities(np.zeros_like(initial_q), joint_indices=joint_indices)
    stable_steps = 0
    elapsed_s = 0.0
    maximum_error = float("inf")
    while simulation_app.is_running() and elapsed_s < INITIALIZATION_TIMEOUT_S:
        controller.apply_action(ArticulationAction(
            joint_positions=initial_q,
            joint_indices=joint_indices,
        ))
        world.step(render=True)
        elapsed_s += world.get_physics_dt()
        actual = robot.get_joint_positions(joint_indices=joint_indices)
        maximum_error = float(np.max(np.abs(np.asarray(actual) - initial_q)))
        stable_steps = stable_steps + 1 if maximum_error <= INITIALIZATION_TOLERANCE_RAD else 0
        if stable_steps >= INITIALIZATION_STABLE_STEPS:
            print(
                f"initialization settled in {{elapsed_s:.3f}} s; "
                f"maximum joint error {{maximum_error:.6g}} rad"
            )
            return elapsed_s, maximum_error
    raise RuntimeError(
        f"robot did not settle at the first trajectory pose within "
        f"{{INITIALIZATION_TIMEOUT_S:.3f}} s; maximum joint error "
        f"{{maximum_error:.6g}} rad"
    )


def write_tracking_outputs(
    samples,
    sample_count,
    sum_squared_error,
    maximum_error,
    initialization_duration_s,
    initialization_error_rad,
    deposited_markers,
    skipped_deposition_points,
):
    rms = math.sqrt(sum_squared_error / max(1, sample_count))
    tracking_passed = (
        maximum_error <= MAX_ACCEPTABLE_TRACKING_ERROR_RAD
        and rms <= MAX_ACCEPTABLE_RMS_TRACKING_ERROR_RAD
    )
    TRACKING_JSON.write_text(json.dumps({{
        "initialization_duration_s": initialization_duration_s,
        "initialization_error_rad": initialization_error_rad,
        "samples": sample_count,
        "plot_samples": len(samples),
        "maximum_tracking_error_rad": maximum_error,
        "rms_tracking_error_rad": rms,
        "maximum_acceptable_tracking_error_rad": MAX_ACCEPTABLE_TRACKING_ERROR_RAD,
        "maximum_acceptable_rms_tracking_error_rad": MAX_ACCEPTABLE_RMS_TRACKING_ERROR_RAD,
        "tracking_passed": tracking_passed,
        "deposited_markers": deposited_markers,
        "skipped_deposition_points_due_to_tracking": skipped_deposition_points,
    }}, indent=2))
    if samples:
        write_tracking_svg(samples)
    print(f"tracking log: {{TRACKING_CSV}}")
    print(f"maximum tracking error: {{maximum_error:.6g}} rad")
    print(f"RMS tracking error: {{rms:.6g}} rad")
    print(f"tracking validation: {{'PASS' if tracking_passed else 'FAIL'}}")


def write_tracking_svg(samples):
    """Write a dependency-free desired-vs-actual joint tracking plot."""
    width, left, right, panel_height = 1000, 80, 20, 125
    height = 40 + panel_height * len(JOINT_COLUMNS)
    t_end = max(samples[-1]["time_s"], 1e-6)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{{width}}" height="{{height}}" viewBox="0 0 {{width}} {{height}}">', '<rect width="100%" height="100%" fill="white"/>']
    for joint_index, name in enumerate(JOINT_COLUMNS):
        top = 25 + joint_index * panel_height
        values = [sample["desired"][joint_index] for sample in samples] + [sample["actual"][joint_index] for sample in samples]
        lower, upper = min(values), max(values)
        if upper - lower < 1e-6:
            lower -= 0.5
            upper += 0.5
        def points(key):
            return " ".join(
                f"{{left + (width - left - right) * sample['time_s'] / t_end:.2f}},{{top + 95 - 90 * (sample[key][joint_index] - lower) / (upper - lower):.2f}}"
                for sample in samples
            )
        parts.extend([
            f'<text x="5" y="{{top + 14}}" font-size="12">{{name}}</text>',
            f'<line x1="{{left}}" y1="{{top + 95}}" x2="{{width - right}}" y2="{{top + 95}}" stroke="#999"/>',
            f'<polyline points="{{points("desired")}}" fill="none" stroke="#1565c0" stroke-width="1.5"/>',
            f'<polyline points="{{points("actual")}}" fill="none" stroke="#ef6c00" stroke-width="1.5"/>',
        ])
    parts.append('<text x="80" y="18" font-size="12" fill="#1565c0">desired</text><text x="150" y="18" font-size="12" fill="#ef6c00">actual</text></svg>')
    TRACKING_SVG.write_text("\n".join(parts))


world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(ROBOT_USD, ROBOT_PRIM)
enable_robot_physics_variants(world.stage, ROBOT_PRIM)
repair_extruder_mount(world.stage, ROBOT_PRIM)
ARTICULATION_PRIM = find_or_create_articulation_root(world.stage, ROBOT_PRIM)
print(f"robot asset: {{ROBOT_USD}}")
print(f"articulation root: {{ARTICULATION_PRIM}}")
robot = SingleArticulation(prim_path=ARTICULATION_PRIM, name="replay_robot")
world.scene.add(robot)
world.reset()
ensure_deposition_root(world.stage)

trajectory = load_rows(TRAJECTORY_CSV)
if not trajectory:
    raise RuntimeError(f"trajectory contains no rows: {{TRAJECTORY_CSV}}")
controller = robot.get_articulation_controller()
available_dof_names = list(robot.dof_names)
print(f"available articulation DOFs: {{available_dof_names}}")
joint_indices = []
for name in JOINT_COLUMNS:
    try:
        index = robot.get_dof_index(name)
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            f"robot articulation {{ARTICULATION_PRIM}} does not expose required "
            f"joint '{{name}}'; available DOFs: {{available_dof_names}}"
        ) from error
    if index is None or index < 0:
        raise RuntimeError(
            f"robot articulation {{ARTICULATION_PRIM}} returned invalid index "
            f"{{index}} for joint '{{name}}'; available DOFs: {{available_dof_names}}"
        )
    joint_indices.append(index)
joint_indices = np.asarray(joint_indices, dtype=int)
initialization_duration_s, initialization_error_rad = initialize_at_first_target(
    world,
    robot,
    controller,
    joint_indices,
    trajectory[0]["q"],
)
cursor = 0
last_row_index = -1
print_point_counter = 0
marker_count = 0
skipped_deposition_points = 0
tracking_samples = []
tracking_sample_count = 0
sum_squared_error = 0.0
maximum_tracking_error = 0.0
physics_step_count = 0
replay_time_s = 0.0
tracking_file = TRACKING_CSV.open("w", newline="")
tracking_writer = csv.writer(tracking_file)
tracking_writer.writerow([
    "trajectory_time_s",
    *[f"desired_{{name}}" for name in JOINT_COLUMNS],
    *[f"actual_{{name}}" for name in JOINT_COLUMNS],
    *[f"error_{{name}}" for name in JOINT_COLUMNS],
])

try:
    while simulation_app.is_running():
        if not trajectory:
            world.step(render=True)
            continue
        command_time_s = replay_time_s
        q_desired, row_index = interpolate_joint_target(trajectory, command_time_s, cursor)
        cursor = row_index
        # Send targets to Isaac's controller; do not teleport during replay.
        controller.apply_action(ArticulationAction(
            joint_positions=np.asarray(q_desired, dtype=float),
            joint_indices=joint_indices,
        ))
        world.step(render=True)
        replay_time_s += world.get_physics_dt()
        actual_positions = robot.get_joint_positions(joint_indices=joint_indices)
        q_actual = actual_positions.tolist() if hasattr(actual_positions, "tolist") else list(actual_positions)
        error = [actual - desired for actual, desired in zip(q_actual, q_desired)]
        tracking_writer.writerow([command_time_s, *q_desired, *q_actual, *error])
        tracking_file.flush()
        tracking_sample_count += len(error)
        sum_squared_error += sum(value * value for value in error)
        maximum_tracking_error = max(maximum_tracking_error, max((abs(value) for value in error), default=0.0))
        if physics_step_count % TRACKING_PLOT_SAMPLE_STRIDE == 0:
            tracking_samples.append({{
                "time_s": command_time_s,
                "desired": q_desired,
                "actual": q_actual,
                "error": error,
            }})
        physics_step_count += 1
        if DEPOSITION_ENABLED and row_index > last_row_index:
            for deposition_row in trajectory[last_row_index + 1:row_index + 1]:
                if not deposition_row["is_print"] or deposition_row["de"] <= 0.0:
                    continue
                print_point_counter += 1
                if max((abs(value) for value in error), default=0.0) > DEPOSITION_MAX_JOINT_ERROR_RAD:
                    skipped_deposition_points += 1
                    continue
                if print_point_counter % DEPOSITION_EVERY_N_PRINT_POINTS != 0:
                    continue
                if marker_count >= MAX_DEPOSITION_MARKERS:
                    continue
                marker_count += 1
                spawn_deposition_marker(
                    world.stage,
                    marker_count,
                    deposition_row["p"],
                    deposition_row["volume_mm3"],
            )
            last_row_index = row_index
        if command_time_s >= trajectory[-1]["time_from_start_s"] + SETTLING_TIME_S:
            break
finally:
    tracking_file.close()
    write_tracking_outputs(
        tracking_samples,
        tracking_sample_count,
        sum_squared_error,
        maximum_tracking_error,
        initialization_duration_s,
        initialization_error_rad,
        marker_count,
        skipped_deposition_points,
    )
    simulation_app.close()
'''


def export_isaac_bundle(
    traj: RobotTrajectory,
    output_dir: str | Path = "outputs",
    basename: str = "robot_print",
    robot_usd_path: str | Path | None = None,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    effective_robot_usd_path = (
        traj.config.isaac_usd_path
        if robot_usd_path is None
        else str(Path(robot_usd_path).resolve())
    )
    csv_path = out / f"{basename}_trajectory.csv"
    json_path = out / f"{basename}_trajectory.json"
    script_path = out / "replay_isaac.py"
    robot_usd_relative = (
        Path(os.path.relpath(effective_robot_usd_path, start=out.resolve())).as_posix()
        if robot_usd_path is not None
        else ""
    )
    robot_usd_fallback = effective_robot_usd_path if robot_usd_path is None else ""

    traj.export_csv(csv_path)
    traj.export_json(json_path)
    script_path.write_text(
        ISAAC_SCRIPT.format(
            trajectory_filename=csv_path.name,
            joint_columns=json.dumps(traj.config.joint_names),
            robot_model=traj.config.robot_model,
            robot_usd_path=effective_robot_usd_path,
            robot_usd_relative=robot_usd_relative,
            robot_usd_fallback=robot_usd_fallback,
            robot_prim_name="".join(c if c.isalnum() else "_" for c in traj.config.robot_model.title()),
        )
    )
    return {"csv": csv_path, "json": json_path, "isaac_script": script_path}
