"""
Replay a ur5 joint trajectory exported by this project.

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

simulation_app = SimulationApp({"headless": False})

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
from omni.physx.scripts import particleUtils, physicsUtils
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt, PhysxSchema

SCRIPT_DIR = Path(__file__).resolve().parent
TRAJECTORY_CSV = Path(
    os.environ.get("RPP_TRAJECTORY_CSV", SCRIPT_DIR / 'robot_print_trajectory.csv')
).resolve()
TRACKING_CSV = TRAJECTORY_CSV.with_name("joint_tracking.csv")
TRACKING_JSON = TRAJECTORY_CSV.with_name("joint_tracking_summary.json")
TRACKING_SVG = TRAJECTORY_CSV.with_name("joint_tracking.svg")
JOINT_COLUMNS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ROBOT_USD_RELATIVE = '../../../UR5e_extruder.usd'
ROBOT_USD_DEFAULT = (
    str((SCRIPT_DIR / ROBOT_USD_RELATIVE).resolve())
    if ROBOT_USD_RELATIVE
    else ''
)
ROBOT_USD = os.environ.get("RPP_ROBOT_USD", ROBOT_USD_DEFAULT)
ROBOT_PRIM = "/World/Ur5"
DEPOSITION_PRIM = "/World/PrintedMaterial"
PARTICLE_SYSTEM_PRIM = f"{DEPOSITION_PRIM}/ParticleSystem"
PARTICLE_SET_PRIM = f"{DEPOSITION_PRIM}/Particles"
PARTICLE_MATERIAL_PRIM = f"{DEPOSITION_PRIM}/Material"
PRINT_BED_PRIM = "/World/PrintBed"
DEPOSITION_ENABLED = True
DEPOSITION_MODE = os.environ.get("RPP_DEPOSITION_MODE", "particles").strip().lower()
DEPOSITION_EVERY_N_PRINT_POINTS = 1
MAX_DEPOSITION_MARKERS = 20000
MAX_DEPOSITION_PARTICLES = int(os.environ.get("RPP_MAX_DEPOSITION_PARTICLES", "250000"))
BEAD_RADIUS_M = 0.0012
BEAD_COLOR = Gf.Vec3f(1.0, 0.28, 0.03)
BED_CENTER_M = (0.45, 0.0, 0.1)
BED_HALF_EXTENTS_XY_M = (0.15, 0.15)
BED_THICKNESS_M = 0.02
MATERIAL_DENSITY_KG_M3 = 1240.0
PARTICLE_CONTACT_OFFSET_M = 0.0005
PARTICLE_FLUID_REST_OFFSET_M = 0.99 * 0.6 * PARTICLE_CONTACT_OFFSET_M
PARTICLE_SOLID_REST_OFFSET_M = 0.99 * PARTICLE_CONTACT_OFFSET_M
PARTICLE_REST_OFFSET_M = PARTICLE_FLUID_REST_OFFSET_M
PARTICLE_COLLISION_CONTACT_OFFSET_M = 1.05 * PARTICLE_REST_OFFSET_M
PARTICLE_VOLUME_MM3 = 8.0 * PARTICLE_FLUID_REST_OFFSET_M ** 3 * 1.0e9
PARTICLE_VISCOSITY = 1000.0
PARTICLE_COHESION = 5.0
PARTICLE_ADHESION = 10.0
PARTICLE_SURFACE_TENSION = 0.02
PARTICLE_FRICTION = 1000.0
PARTICLE_DAMPING = 0.99
PARTICLE_ISOSURFACE_ENABLED = os.environ.get(
    "RPP_PARTICLE_ISOSURFACE", "1"
).strip().lower() in {"1", "true", "yes", "on"}
SETTLING_TIME_S = 2.0
INITIALIZATION_TIMEOUT_S = 30.0
INITIALIZATION_TOLERANCE_RAD = 0.05
INITIALIZATION_STABLE_STEPS = 10
MOUNT_MAX_REASONABLE_DIMENSION_M = 0.30
MOUNT_TARGET_MAX_DIMENSION_M = 0.148
DEFAULT_MOUNT_MASS_KG = 1.0
DEPOSITION_MAX_JOINT_ERROR_RAD = 0.05
MAX_ACCEPTABLE_TRACKING_ERROR_RAD = 0.05
MAX_ACCEPTABLE_RMS_TRACKING_ERROR_RAD = 0.02
TRACKING_PLOT_SAMPLE_STRIDE = 10

if DEPOSITION_MODE not in {"visual", "particles"}:
    raise RuntimeError("RPP_DEPOSITION_MODE must be 'visual' or 'particles'")
if MAX_DEPOSITION_PARTICLES <= 0:
    raise RuntimeError("RPP_MAX_DEPOSITION_PARTICLES must be a positive integer")


def enable_robot_physics_variants(stage, reference_prim_path):
    """Select PhysX on referenced robot assets that were saved as visual-only."""
    reference_prim = stage.GetPrimAtPath(reference_prim_path)
    if not reference_prim.IsValid():
        raise RuntimeError(f"robot reference prim does not exist: {reference_prim_path}")
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
                f"cannot select the PhysX variant on instance proxy {prim.GetPath()}"
            )
        if not variant_set.SetVariantSelection("PhysX"):
            raise RuntimeError(
                f"failed to select the PhysX variant on {prim.GetPath()}; "
                f"available variants: {variant_names}"
            )
        print(
            f"enabled robot physics variant: {prim.GetPath()} "
            f"Physics={previous!r} -> 'PhysX'"
        )


def repair_extruder_mount(stage, reference_prim_path):
    """Repair the known mount payload's scale, mesh metadata, and collider."""
    mount_path = f"{reference_prim_path}/ur5_mount_extruder"
    mesh_path = f"{mount_path}/MeshBody1"
    mount_prim = stage.GetPrimAtPath(mount_path)
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mount_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh):
        print(f"extruder mount repair skipped; mesh not found: {mesh_path}")
        return

    mesh = UsdGeom.Mesh(mesh_prim)
    scale_attr = mount_prim.GetAttribute("xformOp:scale")
    authored_scale = scale_attr.Get() if scale_attr.IsValid() else None
    mount_scale_correction = 1.0
    if authored_scale is not None:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        world_size = bbox_cache.ComputeWorldBound(mesh_prim).ComputeAlignedRange().GetSize()
        current_dimension_m = max(abs(float(world_size[axis])) for axis in range(3))
        scale_override = os.environ.get("RPP_MOUNT_SCALE")
        if scale_override:
            try:
                uniform_scale = float(scale_override)
            except ValueError as error:
                raise RuntimeError("RPP_MOUNT_SCALE must be a positive number") from error
            if not math.isfinite(uniform_scale) or uniform_scale <= 0.0:
                raise RuntimeError("RPP_MOUNT_SCALE must be a positive number")
            corrected_scale = Gf.Vec3f(uniform_scale, uniform_scale, uniform_scale)
        elif current_dimension_m > MOUNT_MAX_REASONABLE_DIMENSION_M:
            correction = MOUNT_TARGET_MAX_DIMENSION_M / current_dimension_m
            corrected_scale = Gf.Vec3f(*[
                float(authored_scale[axis]) * correction
                for axis in range(3)
            ])
        else:
            corrected_scale = None
        if corrected_scale is not None:
            if abs(float(authored_scale[0])) > 1e-12:
                mount_scale_correction = (
                    float(corrected_scale[0]) / float(authored_scale[0])
                )
            scale_attr.Set(corrected_scale)
            corrected_bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            )
            corrected_size = corrected_bbox_cache.ComputeWorldBound(
                mesh_prim
            ).ComputeAlignedRange().GetSize()
            corrected_dimension_m = max(
                abs(float(corrected_size[axis])) for axis in range(3)
            )
            print(
                f"corrected mount scale: {tuple(authored_scale)} -> "
                f"{tuple(corrected_scale)}; maximum dimension "
                f"{current_dimension_m:.3f} m -> {corrected_dimension_m:.3f} m"
            )

    fixed_joint_prim = stage.GetPrimAtPath(f"{mount_path}/FixedJoint")
    if fixed_joint_prim.IsValid() and not math.isclose(mount_scale_correction, 1.0):
        local_pos_attr = fixed_joint_prim.GetAttribute("physics:localPos0")
        local_pos = local_pos_attr.Get() if local_pos_attr.IsValid() else None
        if local_pos is not None:
            corrected_local_pos = Gf.Vec3f(*[
                float(local_pos[axis]) * mount_scale_correction
                for axis in range(3)
            ])
            local_pos_attr.Set(corrected_local_pos)
            print(
                f"corrected mount fixed-joint anchor: {tuple(local_pos)} -> "
                f"{tuple(corrected_local_pos)}"
            )

    normals = mesh.GetNormalsAttr().Get() or []
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get() or []
    expected_face_varying_count = sum(int(count) for count in face_vertex_counts)
    if (
        len(normals) > 0
        and len(normals) == expected_face_varying_count
        and mesh.GetNormalsInterpolation() != UsdGeom.Tokens.faceVarying
    ):
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        print(f"repaired mount normals interpolation: {mesh_path} -> faceVarying")

    collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
    collision_enabled = os.environ.get("RPP_ENABLE_MOUNT_COLLISION", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    collision_api.CreateCollisionEnabledAttr(collision_enabled)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision_api.CreateApproximationAttr().Set("convexHull")
    print(
        f"mount convex-hull collision: "
        f"{'enabled' if collision_enabled else 'disabled for stable replay'}; {mesh_path}"
    )

    mass_override = os.environ.get("RPP_MOUNT_MASS_KG")
    if mass_override:
        try:
            mass_kg = float(mass_override)
        except ValueError as error:
            raise RuntimeError("RPP_MOUNT_MASS_KG must be a positive number") from error
        if not math.isfinite(mass_kg) or mass_kg <= 0.0:
            raise RuntimeError("RPP_MOUNT_MASS_KG must be a positive number")
        print(f"set measured mount/extruder mass: {mass_kg:.6g} kg")
    else:
        mass_kg = DEFAULT_MOUNT_MASS_KG
        print(
            f"mount mass: {mass_kg:.6g} kg simulation fallback "
            f"(set RPP_MOUNT_MASS_KG to measured payload mass)"
        )
    mass_api = UsdPhysics.MassAPI.Apply(mount_prim)
    mass_api.CreateMassAttr().Set(mass_kg)


def find_or_create_articulation_root(stage, reference_prim_path):
    """Find an articulation root, or mark the assembly root when omitted.

    Custom robot USDs commonly wrap the actual articulation below a stage or
    assembly Xform. Some Robot Assembler exports retain all PhysicsJoint prims
    but omit PhysicsArticulationRootAPI. For those assets, marking the assembly
    Xform is valid for both fixed-base and floating articulations.
    """
    reference_prim = stage.GetPrimAtPath(reference_prim_path)
    if not reference_prim.IsValid():
        raise RuntimeError(f"robot reference prim does not exist: {reference_prim_path}")
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
            f"expected exactly one articulation root below {reference_prim_path}, "
            f"found {len(roots)}: {roots}"
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
            f"robot asset below {reference_prim_path} exposes only "
            f"{len(movable_joints)} movable physics joints (expected at least "
            f"{len(JOINT_COLUMNS)}). Refusing to mark a mount or rigid body as "
            f"the robot articulation. Joints found: {joints}; asset: {ROBOT_USD}"
        )
    articulation_api = UsdPhysics.ArticulationRootAPI.Apply(reference_prim)
    if not articulation_api:
        raise RuntimeError(
            f"could not apply PhysicsArticulationRootAPI to {reference_prim_path}"
        )
    print(
        f"asset had no articulation root marker; applied one to "
        f"{reference_prim_path} (found {len(joints)} physics joints)"
    )
    return reference_prim_path


def load_rows(path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "q": [float(row[name]) for name in JOINT_COLUMNS],
                "p": [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                "time_from_start_s": float(row.get("time_from_start_s") or 0.0),
                "is_print": row.get("is_print", "0") == "1",
                "de": float(row.get("de") or 0.0),
                "volume_mm3": float(row.get("extrusion_volume_mm3") or 0.0),
            })
    return sorted(rows, key=lambda row: row["time_from_start_s"])


def ensure_deposition_root(stage):
    if stage.GetPrimAtPath(DEPOSITION_PRIM):
        return
    UsdGeom.Xform.Define(stage, Sdf.Path(DEPOSITION_PRIM))


def configure_gpu_particle_scene(stage):
    """Enable the GPU PhysX features required by PBD particles."""
    physics_scenes = [
        UsdPhysics.Scene(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Scene)
    ]
    if not physics_scenes:
        physics_scenes = [UsdPhysics.Scene.Define(stage, "/World/physicsScene")]
    if len(physics_scenes) != 1:
        raise RuntimeError(
            f"expected one physics scene for particle simulation, found "
            f"{[str(scene.GetPath()) for scene in physics_scenes]}"
        )
    physics_scene = physics_scenes[0]
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
    physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
    physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
    physx_scene.CreateEnableCCDAttr().Set(True)
    return physics_scene.GetPath()


def create_print_bed(stage):
    """Create the configured print bed as a static collision body."""
    bed = UsdGeom.Cube.Define(stage, PRINT_BED_PRIM)
    bed.CreateSizeAttr(1.0)
    bed.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.20, 0.22)])
    bed.AddTranslateOp().Set(Gf.Vec3d(
        float(BED_CENTER_M[0]),
        float(BED_CENTER_M[1]),
        float(BED_CENTER_M[2]) - 0.5 * BED_THICKNESS_M,
    ))
    bed.AddScaleOp().Set(Gf.Vec3d(
        2.0 * float(BED_HALF_EXTENTS_XY_M[0]),
        2.0 * float(BED_HALF_EXTENTS_XY_M[1]),
        BED_THICKNESS_M,
    ))
    UsdPhysics.CollisionAPI.Apply(bed.GetPrim())
    collision = PhysxSchema.PhysxCollisionAPI.Apply(bed.GetPrim())
    collision.CreateContactOffsetAttr().Set(0.5 * PARTICLE_REST_OFFSET_M)
    collision.CreateRestOffsetAttr().Set(0.0)

    bed_material_path = Sdf.Path(f"{PRINT_BED_PRIM}Material")
    UsdShade.Material.Define(stage, bed_material_path)
    bed_material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(bed_material_path))
    bed_material.CreateStaticFrictionAttr().Set(0.8)
    bed_material.CreateDynamicFrictionAttr().Set(0.7)
    bed_material.CreateRestitutionAttr().Set(0.0)
    physicsUtils.add_physics_material_to_prim(stage, bed.GetPrim(), bed_material_path)
    return bed


def create_preview_material(stage, path):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(BEAD_COLOR)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.42)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def extend_array_attribute(attribute, elements):
    values = attribute.Get()
    combined = list(values) if values is not None else []
    combined.extend(elements)
    attribute.Set(combined)


class PhysxExtrusionEmitter:
    """Append volume-conserving particles to one shared PhysX fluid set."""

    def __init__(self, stage, simulation_owner):
        ensure_deposition_root(stage)
        self.stage = stage
        self.volume_remainder_mm3 = 0.0
        self.particle_count = 0
        self.skipped_particles = 0

        system_path = Sdf.Path(PARTICLE_SYSTEM_PRIM)
        self.system = particleUtils.add_physx_particle_system(
            stage,
            system_path,
            simulation_owner=simulation_owner,
            contact_offset=PARTICLE_COLLISION_CONTACT_OFFSET_M,
            rest_offset=PARTICLE_REST_OFFSET_M,
            particle_contact_offset=PARTICLE_CONTACT_OFFSET_M,
            solid_rest_offset=PARTICLE_SOLID_REST_OFFSET_M,
            fluid_rest_offset=PARTICLE_FLUID_REST_OFFSET_M,
            enable_ccd=True,
            solver_position_iterations=8,
            max_neighborhood=96,
            max_velocity=2.0,
            global_self_collision_enabled=True,
            non_particle_collision_enabled=True,
        )
        particleUtils.add_physx_particle_smoothing(
            stage, system_path, enabled=True, strength=1.0
        )
        if PARTICLE_ISOSURFACE_ENABLED:
            particleUtils.add_physx_particle_isosurface(
                stage,
                system_path,
                enabled=True,
                grid_spacing=PARTICLE_FLUID_REST_OFFSET_M,
                surface_distance=1.6 * PARTICLE_FLUID_REST_OFFSET_M,
                num_mesh_smoothing_passes=2,
                num_mesh_normal_smoothing_passes=2,
            )

        material_path = Sdf.Path(PARTICLE_MATERIAL_PRIM)
        self.material = create_preview_material(stage, material_path)
        particleUtils.add_pbd_particle_material(
            stage,
            material_path,
            density=MATERIAL_DENSITY_KG_M3,
            viscosity=PARTICLE_VISCOSITY,
            cohesion=PARTICLE_COHESION,
            adhesion=PARTICLE_ADHESION,
            surface_tension=PARTICLE_SURFACE_TENSION,
            friction=PARTICLE_FRICTION,
            damping=PARTICLE_DAMPING,
        )
        physicsUtils.add_physics_material_to_prim(
            stage, self.system.GetPrim(), material_path
        )
        UsdShade.MaterialBindingAPI(self.system.GetPrim()).Bind(self.material)

        self.points = particleUtils.add_physx_particleset_points(
            stage,
            Sdf.Path(PARTICLE_SET_PRIM),
            [],
            [],
            [],
            system_path,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=0.0,
            density=0.0,
        )
        self.points.CreateDisplayColorAttr([BEAD_COLOR])
        self.points.GetPrim().CreateAttribute(
            "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
        ).Set(MAX_DEPOSITION_PARTICLES)
        UsdShade.MaterialBindingAPI(self.points.GetPrim()).Bind(self.material)

    def emit_segment(self, start, end, volume_mm3, duration_s):
        available_volume = self.volume_remainder_mm3 + max(0.0, volume_mm3)
        requested_count = int(available_volume / PARTICLE_VOLUME_MM3)
        self.volume_remainder_mm3 = available_volume - requested_count * PARTICLE_VOLUME_MM3
        remaining_capacity = MAX_DEPOSITION_PARTICLES - self.particle_count
        emit_count = min(requested_count, max(0, remaining_capacity))
        self.skipped_particles += requested_count - emit_count
        if emit_count <= 0:
            return 0

        start_array = np.asarray(start, dtype=float)
        end_array = np.asarray(end, dtype=float)
        delta = end_array - start_array
        minimum_z = float(BED_CENTER_M[2]) + PARTICLE_REST_OFFSET_M
        segment_velocity = delta / max(float(duration_s), 1.0e-6)
        positions = []
        velocities = []
        for index in range(emit_count):
            alpha = (index + 0.5) / emit_count
            position = start_array + alpha * delta
            position[2] = max(float(position[2]), minimum_z)
            positions.append(Gf.Vec3f(*[float(value) for value in position]))
            velocities.append(Gf.Vec3f(*[float(value) for value in segment_velocity]))

        particle_set = PhysxSchema.PhysxParticleSetAPI(self.points.GetPrim())
        simulation_points = particle_set.GetSimulationPointsAttr()
        if not simulation_points.HasAuthoredValue():
            simulation_points.Set(Vt.Vec3fArray([]))
        extend_array_attribute(simulation_points, positions)
        extend_array_attribute(self.points.GetPointsAttr(), positions)
        extend_array_attribute(self.points.GetVelocitiesAttr(), velocities)
        extend_array_attribute(
            self.points.GetWidthsAttr(),
            [2.0 * PARTICLE_FLUID_REST_OFFSET_M] * emit_count,
        )
        self.particle_count += emit_count
        return emit_count


def spawn_deposition_segment(stage, marker_index, start, end, volume_mm3):
    """Draw the complete extruded move, including space between waypoints."""
    prim_path = Sdf.Path(f"{DEPOSITION_PRIM}/bead_{marker_index:06d}")
    radius = BEAD_RADIUS_M
    if volume_mm3 > 0.0:
        # Keep visual size stable, but let higher-flow segments read slightly thicker.
        radius *= max(0.6, min(1.8, (volume_mm3 / 0.28) ** (1.0 / 3.0)))
    curves = UsdGeom.BasisCurves.Define(stage, prim_path)
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curves.CreateCurveVertexCountsAttr([2])
    curves.CreatePointsAttr([
        Gf.Vec3f(float(start[0]), float(start[1]), float(start[2])),
        Gf.Vec3f(float(end[0]), float(end[1]), float(end[2])),
    ])
    curves.CreateWidthsAttr([2.0 * radius])
    curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curves.CreateDisplayColorAttr([BEAD_COLOR])


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
                f"initialization settled in {elapsed_s:.3f} s; "
                f"maximum joint error {maximum_error:.6g} rad"
            )
            return elapsed_s, maximum_error
    raise RuntimeError(
        f"robot did not settle at the first trajectory pose within "
        f"{INITIALIZATION_TIMEOUT_S:.3f} s; maximum joint error "
        f"{maximum_error:.6g} rad"
    )


def write_tracking_outputs(
    samples,
    sample_count,
    sum_squared_error,
    maximum_error,
    initialization_duration_s,
    initialization_error_rad,
    deposited_markers,
    deposited_particles,
    unrepresented_volume_mm3,
    particle_limit_skips,
    skipped_deposition_points,
):
    rms = math.sqrt(sum_squared_error / max(1, sample_count))
    tracking_passed = (
        maximum_error <= MAX_ACCEPTABLE_TRACKING_ERROR_RAD
        and rms <= MAX_ACCEPTABLE_RMS_TRACKING_ERROR_RAD
    )
    TRACKING_JSON.write_text(json.dumps({
        "initialization_duration_s": initialization_duration_s,
        "initialization_error_rad": initialization_error_rad,
        "samples": sample_count,
        "plot_samples": len(samples),
        "maximum_tracking_error_rad": maximum_error,
        "rms_tracking_error_rad": rms,
        "maximum_acceptable_tracking_error_rad": MAX_ACCEPTABLE_TRACKING_ERROR_RAD,
        "maximum_acceptable_rms_tracking_error_rad": MAX_ACCEPTABLE_RMS_TRACKING_ERROR_RAD,
        "tracking_passed": tracking_passed,
        "deposition_mode": DEPOSITION_MODE,
        "deposited_markers": deposited_markers,
        "deposited_particles": deposited_particles,
        "particle_volume_mm3": PARTICLE_VOLUME_MM3 if DEPOSITION_MODE == "particles" else None,
        "unrepresented_volume_mm3": unrepresented_volume_mm3,
        "particles_skipped_due_to_limit": particle_limit_skips,
        "skipped_deposition_points_due_to_tracking": skipped_deposition_points,
    }, indent=2))
    if samples:
        write_tracking_svg(samples)
    print(f"tracking log: {TRACKING_CSV}")
    print(f"maximum tracking error: {maximum_error:.6g} rad")
    print(f"RMS tracking error: {rms:.6g} rad")
    print(f"tracking validation: {'PASS' if tracking_passed else 'FAIL'}")


def write_tracking_svg(samples):
    """Write a dependency-free desired-vs-actual joint tracking plot."""
    width, left, right, panel_height = 1000, 80, 20, 125
    height = 40 + panel_height * len(JOINT_COLUMNS)
    t_end = max(samples[-1]["time_s"], 1e-6)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for joint_index, name in enumerate(JOINT_COLUMNS):
        top = 25 + joint_index * panel_height
        values = [sample["desired"][joint_index] for sample in samples] + [sample["actual"][joint_index] for sample in samples]
        lower, upper = min(values), max(values)
        if upper - lower < 1e-6:
            lower -= 0.5
            upper += 0.5
        def points(key):
            return " ".join(
                f"{left + (width - left - right) * sample['time_s'] / t_end:.2f},{top + 95 - 90 * (sample[key][joint_index] - lower) / (upper - lower):.2f}"
                for sample in samples
            )
        parts.extend([
            f'<text x="5" y="{top + 14}" font-size="12">{name}</text>',
            f'<line x1="{left}" y1="{top + 95}" x2="{width - right}" y2="{top + 95}" stroke="#999"/>',
            f'<polyline points="{points("desired")}" fill="none" stroke="#1565c0" stroke-width="1.5"/>',
            f'<polyline points="{points("actual")}" fill="none" stroke="#ef6c00" stroke-width="1.5"/>',
        ])
    parts.append('<text x="80" y="18" font-size="12" fill="#1565c0">desired</text><text x="150" y="18" font-size="12" fill="#ef6c00">actual</text></svg>')
    TRACKING_SVG.write_text("\n".join(parts))


trajectory = load_rows(TRAJECTORY_CSV)
if not trajectory:
    raise RuntimeError(f"trajectory contains no rows: {TRAJECTORY_CSV}")

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(ROBOT_USD, ROBOT_PRIM)
enable_robot_physics_variants(world.stage, ROBOT_PRIM)
repair_extruder_mount(world.stage, ROBOT_PRIM)
ARTICULATION_PRIM = find_or_create_articulation_root(world.stage, ROBOT_PRIM)
ensure_deposition_root(world.stage)
create_print_bed(world.stage)
particle_emitter = None
if DEPOSITION_ENABLED and DEPOSITION_MODE == "particles":
    physics_scene_path = configure_gpu_particle_scene(world.stage)
    particle_emitter = PhysxExtrusionEmitter(world.stage, physics_scene_path)
    estimated_particles = math.ceil(
        sum(max(0.0, row["volume_mm3"]) for row in trajectory) / PARTICLE_VOLUME_MM3
    )
    print(
        f"PhysX PBD extrusion: estimated {estimated_particles} particles; "
        f"capacity {MAX_DEPOSITION_PARTICLES}; density "
        f"{MATERIAL_DENSITY_KG_M3:.6g} kg/m^3"
    )
print(f"robot asset: {ROBOT_USD}")
print(f"articulation root: {ARTICULATION_PRIM}")
robot = SingleArticulation(prim_path=ARTICULATION_PRIM, name="replay_robot")
world.scene.add(robot)
world.reset()
controller = robot.get_articulation_controller()
available_dof_names = list(robot.dof_names)
print(f"available articulation DOFs: {available_dof_names}")
joint_indices = []
for name in JOINT_COLUMNS:
    try:
        index = robot.get_dof_index(name)
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            f"robot articulation {ARTICULATION_PRIM} does not expose required "
            f"joint '{name}'; available DOFs: {available_dof_names}"
        ) from error
    if index is None or index < 0:
        raise RuntimeError(
            f"robot articulation {ARTICULATION_PRIM} returned invalid index "
            f"{index} for joint '{name}'; available DOFs: {available_dof_names}"
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
    *[f"desired_{name}" for name in JOINT_COLUMNS],
    *[f"actual_{name}" for name in JOINT_COLUMNS],
    *[f"error_{name}" for name in JOINT_COLUMNS],
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
            tracking_samples.append({
                "time_s": command_time_s,
                "desired": q_desired,
                "actual": q_actual,
                "error": error,
            })
        physics_step_count += 1
        if DEPOSITION_ENABLED and row_index > last_row_index:
            for deposition_index in range(last_row_index + 1, row_index + 1):
                deposition_row = trajectory[deposition_index]
                if not deposition_row["is_print"] or deposition_row["de"] <= 0.0:
                    continue
                if max((abs(value) for value in error), default=0.0) > DEPOSITION_MAX_JOINT_ERROR_RAD:
                    skipped_deposition_points += 1
                    continue
                if (deposition_index + 1) % DEPOSITION_EVERY_N_PRINT_POINTS != 0:
                    continue
                if DEPOSITION_MODE == "visual" and marker_count >= MAX_DEPOSITION_MARKERS:
                    continue
                marker_count += 1
                segment_start = (
                    trajectory[deposition_index - 1]["p"]
                    if deposition_index > 0
                    else deposition_row["p"]
                )
                if DEPOSITION_MODE == "particles":
                    segment_start_time = (
                        trajectory[deposition_index - 1]["time_from_start_s"]
                        if deposition_index > 0
                        else deposition_row["time_from_start_s"]
                    )
                    particle_emitter.emit_segment(
                        segment_start,
                        deposition_row["p"],
                        deposition_row["volume_mm3"],
                        deposition_row["time_from_start_s"] - segment_start_time,
                    )
                else:
                    spawn_deposition_segment(
                        world.stage,
                        marker_count,
                        segment_start,
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
        particle_emitter.particle_count if particle_emitter is not None else 0,
        (
            particle_emitter.volume_remainder_mm3
            + particle_emitter.skipped_particles * PARTICLE_VOLUME_MM3
            if particle_emitter is not None
            else 0.0
        ),
        particle_emitter.skipped_particles if particle_emitter is not None else 0,
        skipped_deposition_points,
    )
    simulation_app.close()
