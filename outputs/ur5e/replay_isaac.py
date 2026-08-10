"""
Replay a ur5e joint trajectory exported by this project.

Run inside Isaac Sim, for example:
    ./python.sh replay_isaac.py

Set RPP_ROBOT_USD to override the robot asset path at launch time.
Set RPP_PROJECT_ROOT if this script is not below the cloned repository root.
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_project_root(script_dir):
    """Find the cloned repository that owns the central deposition module."""
    def is_project_root(candidate):
        return (
            (candidate / "robotic_printing_platform" / "__init__.py").is_file()
            and (
                candidate
                / "robotic_printing_platform"
                / "extrusion"
                / "deposition.py"
            ).is_file()
        )

    override = os.environ.get("RPP_PROJECT_ROOT", "").strip()
    if override:
        candidate = Path(override).expanduser().resolve()
        if is_project_root(candidate):
            return candidate
        raise RuntimeError(
            f"RPP_PROJECT_ROOT is not a robotic-printing-platform repository: "
            f"{candidate}"
        )

    candidates = (script_dir, *script_dir.parents)
    for candidate in candidates:
        if is_project_root(candidate):
            return candidate
    raise RuntimeError(
        f"could not locate the robotic-printing-platform repository above "
        f"{script_dir}; searched {', '.join(str(path) for path in candidates)}. "
        f"Set RPP_PROJECT_ROOT to the cloned repository root."
    )


PROJECT_ROOT = resolve_project_root(SCRIPT_DIR)
project_root_text = str(PROJECT_ROOT)
while project_root_text in sys.path:
    sys.path.remove(project_root_text)
sys.path.insert(0, project_root_text)

try:
    from robotic_printing_platform.extrusion import deposition as deposition_module
except ImportError as error:
    raise RuntimeError(
        f"failed to import the central deposition module from {PROJECT_ROOT}: "
        f"{error}"
    ) from error
expected_deposition_path = (
    PROJECT_ROOT / "robotic_printing_platform" / "extrusion" / "deposition.py"
).resolve()
actual_deposition_path = Path(deposition_module.__file__).resolve()
if actual_deposition_path != expected_deposition_path:
    raise RuntimeError(
        f"resolved project root {PROJECT_ROOT}, but imported deposition module "
        f"from {actual_deposition_path}"
    )
BeadEvolutionModel = deposition_module.BeadEvolutionModel
DepositionManager = deposition_module.DepositionManager
FlowSchedule = deposition_module.FlowSchedule
TcpPose = deposition_module.TcpPose
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

TRAJECTORY_CSV = Path(
    os.environ.get("RPP_TRAJECTORY_CSV", SCRIPT_DIR / 'robot_print_trajectory.csv')
).resolve()
MATERIAL_PROFILE_PATH = SCRIPT_DIR / "resolved_material_profile.json"
TRACKING_CSV = TRAJECTORY_CSV.with_name("joint_tracking.csv")
TRACKING_JSON = TRAJECTORY_CSV.with_name("joint_tracking_summary.json")
TRACKING_SVG = TRAJECTORY_CSV.with_name("joint_tracking.svg")
JOINT_COLUMNS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ROBOT_USD_RELATIVE = '../../UR5e_extruder.usd'
ROBOT_USD_DEFAULT = (
    str((SCRIPT_DIR / ROBOT_USD_RELATIVE).resolve())
    if ROBOT_USD_RELATIVE
    else ''
)
ROBOT_USD = os.environ.get("RPP_ROBOT_USD", ROBOT_USD_DEFAULT)
ROBOT_PRIM = "/World/Ur5E"
TCP_ANCHOR_CANDIDATES = ({'link_name': 'tool0', 'translation_m': (-0.028434579200000032, 0.007742159999999998, 0.146409252), 'rotation_xyzw': (0.0, 0.0, 0.0, 1.0)}, {'link_name': 'flange', 'translation_m': (0.146409252, -0.028434579200000032, 0.007742160000000012), 'rotation_xyzw': (0.5, 0.5, 0.5, 0.5)}, {'link_name': 'wrist_3_link', 'translation_m': (-0.028434579200000032, 0.007742159999999998, 0.146409252), 'rotation_xyzw': (0.0, -3.061616997868383e-17, 3.0616169978683836e-17, 1.0)})
TCP_PRIM_OVERRIDE = os.environ.get("RPP_TCP_PRIM", "").strip()
DEPOSITION_PRIM = "/World/PrintedMaterial"
PARTICLE_SYSTEM_PRIM = f"{DEPOSITION_PRIM}/ParticleSystem"
PARTICLE_SET_PRIM = f"{DEPOSITION_PRIM}/Particles"
PARTICLE_MATERIAL_PRIM = f"{DEPOSITION_PRIM}/Material"
PRINT_BED_PRIM = "/World/PrintBed"
DEPOSITION_MODE = os.environ.get("RPP_DEPOSITION_MODE", "particles").strip().lower()
MAX_DEPOSITION_MARKERS = int(os.environ.get("RPP_MAX_DEPOSITION_SEGMENTS", "100000"))
VISUAL_SEGMENTS_PER_CHUNK = 256
MAX_DEPOSITION_PARTICLES = int(os.environ.get("RPP_MAX_DEPOSITION_PARTICLES", "250000"))
MAX_TCP_STEP_M = float(os.environ.get("RPP_MAX_TCP_STEP_M", "0.02"))
BEAD_COLOR = Gf.Vec3f(1.0, 0.28, 0.03)
BED_CENTER_M = (0.45, 0.0, 0.1)
BED_HALF_EXTENTS_XY_M = (0.15, 0.15)
BED_THICKNESS_M = 0.02


def load_material_profile(path):
    """Load the material metadata paired with the exported trajectory."""
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"material profile is missing: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read material profile {path}: {error}") from error

    required = {
        "profile_id",
        "name",
        "extrusion_mode",
        "physx_particle_contact_offset_m",
        "physx_viscosity",
        "physx_cohesion",
        "physx_adhesion",
        "physx_surface_tension",
        "physx_friction",
        "physx_damping",
    }
    missing = sorted(required.difference(profile))
    if missing:
        raise RuntimeError(
            f"material profile {path} is missing fields: {', '.join(missing)}"
        )
    return profile


MATERIAL_PROFILE = load_material_profile(MATERIAL_PROFILE_PATH)
MATERIAL_PROFILE_ID = str(MATERIAL_PROFILE["profile_id"])
MATERIAL_NAME = str(MATERIAL_PROFILE["name"])
MATERIAL_EXTRUSION_MODE = str(MATERIAL_PROFILE["extrusion_mode"])
MATERIAL_DENSITY_KG_M3 = 1000.0 * float(
    MATERIAL_PROFILE.get("density_g_cm3") or 1.0
)
PARTICLE_CONTACT_OFFSET_M = float(
    MATERIAL_PROFILE["physx_particle_contact_offset_m"]
)
PARTICLE_FLUID_REST_OFFSET_M = 0.99 * 0.6 * PARTICLE_CONTACT_OFFSET_M
PARTICLE_SOLID_REST_OFFSET_M = 0.99 * PARTICLE_CONTACT_OFFSET_M
PARTICLE_REST_OFFSET_M = PARTICLE_FLUID_REST_OFFSET_M
PARTICLE_COLLISION_CONTACT_OFFSET_M = 1.05 * PARTICLE_REST_OFFSET_M
PARTICLE_VOLUME_MM3 = 8.0 * PARTICLE_FLUID_REST_OFFSET_M ** 3 * 1.0e9
PARTICLE_VISCOSITY = float(MATERIAL_PROFILE["physx_viscosity"])
PARTICLE_COHESION = float(MATERIAL_PROFILE["physx_cohesion"])
PARTICLE_ADHESION = float(MATERIAL_PROFILE["physx_adhesion"])
PARTICLE_SURFACE_TENSION = float(MATERIAL_PROFILE["physx_surface_tension"])
PARTICLE_FRICTION = float(MATERIAL_PROFILE["physx_friction"])
PARTICLE_DAMPING = float(MATERIAL_PROFILE["physx_damping"])
MATERIAL_SPREADING_RATIO = float(MATERIAL_PROFILE.get("spreading_ratio", 1.0))
MATERIAL_SPREADING_TIME_S = float(MATERIAL_PROFILE.get("spreading_time_s", 1.0))
MATERIAL_SHRINKAGE_FRACTION = float(
    MATERIAL_PROFILE.get("shrinkage_fraction", 0.0)
)
MATERIAL_SHRINKAGE_TIME_S = float(MATERIAL_PROFILE.get("shrinkage_time_s", 1.0))
PARTICLE_ISOSURFACE_ENABLED = os.environ.get(
    "RPP_PARTICLE_ISOSURFACE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
SETTLING_TIME_S = float(os.environ.get("RPP_POST_DEPOSITION_TIME_S", "2.0"))
INITIALIZATION_TIMEOUT_S = 30.0
INITIALIZATION_TOLERANCE_RAD = 0.05
INITIALIZATION_STABLE_STEPS = 10
MOUNT_MAX_REASONABLE_DIMENSION_M = 0.30
MOUNT_TARGET_MAX_DIMENSION_M = 0.148
DEFAULT_MOUNT_MASS_KG = 1.0
MAX_ACCEPTABLE_TRACKING_ERROR_RAD = 0.05
MAX_ACCEPTABLE_RMS_TRACKING_ERROR_RAD = 0.02
TRACKING_PLOT_SAMPLE_STRIDE = 10

if DEPOSITION_MODE not in {"visual", "particles"}:
    raise RuntimeError("RPP_DEPOSITION_MODE must be 'visual' or 'particles'")
if MAX_DEPOSITION_PARTICLES <= 0:
    raise RuntimeError("RPP_MAX_DEPOSITION_PARTICLES must be a positive integer")
if MAX_DEPOSITION_MARKERS <= 0:
    raise RuntimeError("RPP_MAX_DEPOSITION_SEGMENTS must be a positive integer")
if not math.isfinite(MAX_TCP_STEP_M) or MAX_TCP_STEP_M <= 0.0:
    raise RuntimeError("RPP_MAX_TCP_STEP_M must be a positive finite number")
if not math.isfinite(SETTLING_TIME_S) or SETTLING_TIME_S < 0.0:
    raise RuntimeError("RPP_POST_DEPOSITION_TIME_S must be a non-negative finite number")


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
                "extrusion_volume_mm3": float(
                    row.get("extrusion_volume_mm3") or 0.0
                ),
            })
    return sorted(rows, key=lambda row: row["time_from_start_s"])


def multiply_quaternions_xyzw(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    result = np.asarray([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ], dtype=float)
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-12:
        raise RuntimeError("computed TCP orientation has zero quaternion norm")
    return tuple(float(value) for value in result / norm)


def resolve_tcp_anchor(stage, robot_prim_path):
    """Resolve the best available simulated link and its local TCP transform."""
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"robot reference prim does not exist: {robot_prim_path}")

    subtree = list(Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()))
    if TCP_PRIM_OVERRIDE:
        prim = stage.GetPrimAtPath(TCP_PRIM_OVERRIDE)
        if not prim.IsValid():
            raise RuntimeError(f"RPP_TCP_PRIM does not exist: {TCP_PRIM_OVERRIDE}")
        robot_path = str(robot_prim.GetPath())
        override_path = str(prim.GetPath())
        if override_path != robot_path and not override_path.startswith(robot_path + "/"):
            raise RuntimeError(
                f"RPP_TCP_PRIM must be below {robot_path}, got {override_path}"
            )
        if not UsdGeom.Xformable(prim):
            raise RuntimeError(f"RPP_TCP_PRIM is not transformable: {TCP_PRIM_OVERRIDE}")
        matching = [
            item for item in TCP_ANCHOR_CANDIDATES
            if item["link_name"] == prim.GetName()
        ]
        if not matching:
            expected = [item["link_name"] for item in TCP_ANCHOR_CANDIDATES]
            raise RuntimeError(
                f"RPP_TCP_PRIM must select one of the configured TCP anchor "
                f"links {expected}, got prim name '{prim.GetName()}'"
            )
        return prim, matching[0]

    searched_names = []
    for anchor in TCP_ANCHOR_CANDIDATES:
        link_name = anchor["link_name"]
        searched_names.append(link_name)
        matches = [
            prim for prim in subtree
            if prim.GetName() == link_name and UsdGeom.Xformable(prim)
        ]
        if len(matches) == 1:
            return matches[0], anchor
        if len(matches) > 1:
            paths = [str(prim.GetPath()) for prim in matches]
            raise RuntimeError(
                f"TCP anchor '{link_name}' is ambiguous below {robot_prim_path}: "
                f"{paths}; set RPP_TCP_PRIM to the intended link"
            )
    raise RuntimeError(
        f"could not find a TCP anchor below {robot_prim_path}; searched "
        f"{searched_names}. Set RPP_TCP_PRIM for a differently named asset."
    )


class RobotTcpPoseReader:
    """Read the realized nozzle TCP pose from the USD articulation hierarchy."""

    def __init__(self, stage, robot_prim_path):
        self.stage = stage
        self.anchor_prim, self.anchor = resolve_tcp_anchor(stage, robot_prim_path)
        print(
            f"TCP anchor: {self.anchor_prim.GetPath()} -> configured nozzle frame"
        )

    def read_pose(self):
        # Build a fresh transform query after every physics step; retaining an
        # XformCache across steps would return stale articulation transforms.
        transform = UsdGeom.Xformable(
            self.anchor_prim
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        offset = self.anchor["translation_m"]
        position = transform.Transform(Gf.Vec3d(*offset))
        anchor_quaternion = transform.ExtractRotationQuat()
        imaginary = anchor_quaternion.GetImaginary()
        orientation = multiply_quaternions_xyzw(
            (
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
                float(anchor_quaternion.GetReal()),
            ),
            self.anchor["rotation_xyzw"],
        )
        return TcpPose(
            position_m=tuple(float(value) for value in position),
            orientation_xyzw=orientation,
        )


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

    def emit_segment(self, start_pose, end_pose, volume_mm3, duration_s):
        current_volume_mm3 = max(0.0, float(volume_mm3))
        prior_remainder_mm3 = self.volume_remainder_mm3
        available_volume = prior_remainder_mm3 + current_volume_mm3
        requested_count = int(available_volume / PARTICLE_VOLUME_MM3)
        self.volume_remainder_mm3 = available_volume - requested_count * PARTICLE_VOLUME_MM3
        remaining_capacity = MAX_DEPOSITION_PARTICLES - self.particle_count
        emit_count = min(requested_count, max(0, remaining_capacity))
        self.skipped_particles += requested_count - emit_count
        if emit_count <= 0:
            return 0

        start_array = np.asarray(start_pose.position_m, dtype=float)
        end_array = np.asarray(end_pose.position_m, dtype=float)
        delta = end_array - start_array
        minimum_z = float(BED_CENTER_M[2]) + PARTICLE_REST_OFFSET_M
        segment_velocity = delta / max(float(duration_s), 1.0e-6)
        positions = []
        velocities = []
        for index in range(emit_count):
            # Account for fractional volume carried from the preceding step so
            # materializing particles do not all shift to the next TCP chord.
            threshold_volume = (index + 1) * PARTICLE_VOLUME_MM3 - prior_remainder_mm3
            alpha = max(
                0.0,
                min(1.0, threshold_volume / max(current_volume_mm3, 1.0e-12)),
            )
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

    def emit_point(self, pose, volume_mm3, duration_s):
        return self.emit_segment(pose, pose, volume_mm3, duration_s)

    def reset(self, clear_geometry=False):
        if not clear_geometry:
            return
        particle_set = PhysxSchema.PhysxParticleSetAPI(self.points.GetPrim())
        particle_set.GetSimulationPointsAttr().Set(Vt.Vec3fArray([]))
        self.points.GetPointsAttr().Set(Vt.Vec3fArray([]))
        self.points.GetVelocitiesAttr().Set(Vt.Vec3fArray([]))
        self.points.GetWidthsAttr().Set([])
        self.volume_remainder_mm3 = 0.0
        self.particle_count = 0
        self.skipped_particles = 0

    def flush(self):
        return None


def visual_bead_radius_m(start_pose, end_pose, volume_mm3):
    """Return the circular radius whose swept cylinder has the target volume."""
    start = np.asarray(start_pose.position_m, dtype=float)
    end = np.asarray(end_pose.position_m, dtype=float)
    length_m = float(np.linalg.norm(end - start))
    volume_m3 = max(0.0, float(volume_mm3)) * 1.0e-9
    if length_m <= 1.0e-12:
        radius_m = (3.0 * volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    else:
        radius_m = math.sqrt(volume_m3 / (math.pi * length_m))
    return radius_m


def spawn_deposition_segment(stage, material, marker_index, start_pose, end_pose, volume_mm3):
    """Draw one volume-scaled curve or stationary droplet at the actual TCP."""
    prim_path = Sdf.Path(f"{DEPOSITION_PRIM}/bead_{marker_index:06d}")
    radius = visual_bead_radius_m(start_pose, end_pose, volume_mm3)
    start = start_pose.position_m
    end = end_pose.position_m
    if np.linalg.norm(np.asarray(end, dtype=float) - np.asarray(start, dtype=float)) <= 1.0e-12:
        sphere = UsdGeom.Sphere.Define(stage, prim_path)
        sphere.CreateRadiusAttr(radius)
        sphere.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in end]))
        sphere.CreateDisplayColorAttr([BEAD_COLOR])
        UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(material)
        return prim_path, sphere, radius
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
    UsdShade.MaterialBindingAPI(curves.GetPrim()).Bind(material)
    return prim_path, None, radius


class _VisualCurveChunk:
    """Retained USD curve arrays needed to evolve every historical segment."""

    def __init__(self, curve):
        self.curve = curve
        self.counts = []
        self.points = []
        self.initial_widths = []
        self.widths = []
        self.birth_times_s = []
        self.active_indices = []


class _VisualDroplet:
    """One stationary sphere and its immutable deposited radius."""

    def __init__(self, sphere, initial_radius_m, birth_time_s):
        self.sphere = sphere
        self.initial_radius_m = initial_radius_m
        self.birth_time_s = birth_time_s
        self.radius_m = initial_radius_m


class VisualBeadSink:
    """USD sink with explicit post-deposition spreading and shrinkage."""

    def __init__(self, stage, evolution_model):
        ensure_deposition_root(stage)
        self.stage = stage
        self.material = create_preview_material(stage, Sdf.Path(PARTICLE_MATERIAL_PRIM))
        self.evolution_model = evolution_model
        self.marker_count = 0
        self.skipped_volume_mm3 = 0.0
        self.geometry_update_count = 0
        self._created_paths = []
        self._chunk_index = 0
        self._curve_chunks = []
        self._current_chunk = None
        self._active_droplets = []
        self._time_s = 0.0
        self._evolution_enabled = (
            evolution_model.spreading_ratio != 1.0
            or evolution_model.shrinkage_fraction != 0.0
        )
        active_time_constants = []
        if evolution_model.spreading_ratio != 1.0:
            active_time_constants.append(evolution_model.spreading_time_s)
        if evolution_model.shrinkage_fraction != 0.0:
            active_time_constants.append(evolution_model.shrinkage_time_s)
        self._settling_age_s = (
            8.0 * max(active_time_constants) if active_time_constants else 0.0
        )
        self._final_visual_radius_scale = math.sqrt(
            evolution_model.spreading_ratio
            * (1.0 - evolution_model.shrinkage_fraction)
        )

    def _begin_curve_chunk(self):
        self._chunk_index += 1
        path = Sdf.Path(
            f"{DEPOSITION_PRIM}/bead_chunk_{self._chunk_index:06d}"
        )
        curves = UsdGeom.BasisCurves.Define(self.stage, path)
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curves.CreateCurveVertexCountsAttr([])
        curves.CreatePointsAttr([])
        curves.CreateWidthsAttr([])
        curves.CreateDisplayColorAttr([BEAD_COLOR])
        curves.SetWidthsInterpolation(UsdGeom.Tokens.uniform)
        UsdShade.MaterialBindingAPI(curves.GetPrim()).Bind(self.material)
        self._created_paths.append(path)
        chunk = _VisualCurveChunk(curves)
        self._curve_chunks.append(chunk)
        self._current_chunk = chunk

    def _emit_curve(self, start_pose, end_pose, volume_mm3):
        if (
            self._current_chunk is None
            or len(self._current_chunk.counts) >= VISUAL_SEGMENTS_PER_CHUNK
        ):
            self._begin_curve_chunk()
        chunk = self._current_chunk
        radius = visual_bead_radius_m(start_pose, end_pose, volume_mm3)
        width = 2.0 * radius
        width_index = len(chunk.widths)
        chunk.counts.append(2)
        chunk.points.extend([
            Gf.Vec3f(*[float(value) for value in start_pose.position_m]),
            Gf.Vec3f(*[float(value) for value in end_pose.position_m]),
        ])
        chunk.initial_widths.append(width)
        chunk.widths.append(width)
        chunk.birth_times_s.append(self._time_s)
        if self._evolution_enabled:
            chunk.active_indices.append(width_index)
        chunk.curve.GetCurveVertexCountsAttr().Set(chunk.counts)
        chunk.curve.GetPointsAttr().Set(chunk.points)
        chunk.curve.GetWidthsAttr().Set(chunk.widths)

    def _emit(self, start_pose, end_pose, volume_mm3):
        if self.marker_count >= MAX_DEPOSITION_MARKERS:
            self.skipped_volume_mm3 += max(0.0, float(volume_mm3))
            return
        self.marker_count += 1
        distance_m = float(np.linalg.norm(
            np.asarray(end_pose.position_m) - np.asarray(start_pose.position_m)
        ))
        if distance_m <= 1.0e-12:
            path, sphere, radius = spawn_deposition_segment(
                self.stage,
                self.material,
                self.marker_count,
                start_pose,
                end_pose,
                volume_mm3,
            )
            self._created_paths.append(path)
            if self._evolution_enabled:
                self._active_droplets.append(
                    _VisualDroplet(sphere, radius, self._time_s)
                )
        else:
            self._emit_curve(start_pose, end_pose, volume_mm3)

    def emit_segment(self, start_pose, end_pose, volume_mm3, duration_s):
        self._emit(start_pose, end_pose, volume_mm3)

    def emit_point(self, pose, volume_mm3, duration_s):
        self._emit(pose, pose, volume_mm3)

    def advance_to(self, simulation_time_s):
        """Age existing beads to an absolute replay time and update USD geometry.

        BasisCurves and spheres are round primitives, so ``visual_radius_scale``
        is a lightweight preview proxy for the model's anisotropic width/height
        scales.  Records snap to the exact asymptote after eight active time
        constants and are then retired from per-step scanning.
        """
        next_time_s = float(simulation_time_s)
        if not math.isfinite(next_time_s):
            raise ValueError("simulation_time_s must be finite")
        if next_time_s < self._time_s:
            raise ValueError("visual bead time must be non-decreasing")
        self._time_s = next_time_s
        if not self._evolution_enabled:
            return

        updated_geometry = 0
        for chunk in self._curve_chunks:
            if not chunk.active_indices:
                continue
            remaining_indices = []
            dirty = False
            for index in chunk.active_indices:
                age_s = max(0.0, next_time_s - chunk.birth_times_s[index])
                if age_s >= self._settling_age_s:
                    scale = self._final_visual_radius_scale
                else:
                    scale = self.evolution_model.state_at(age_s).visual_radius_scale
                    remaining_indices.append(index)
                width = chunk.initial_widths[index] * scale
                if not math.isclose(
                    width,
                    chunk.widths[index],
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                ):
                    chunk.widths[index] = width
                    dirty = True
                    updated_geometry += 1
            if dirty:
                chunk.curve.GetWidthsAttr().Set(chunk.widths)
            chunk.active_indices = remaining_indices

        active_droplets = []
        for droplet in self._active_droplets:
            age_s = max(0.0, next_time_s - droplet.birth_time_s)
            if age_s >= self._settling_age_s:
                scale = self._final_visual_radius_scale
            else:
                scale = self.evolution_model.state_at(age_s).visual_radius_scale
                active_droplets.append(droplet)
            radius_m = droplet.initial_radius_m * scale
            if not math.isclose(
                radius_m,
                droplet.radius_m,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                droplet.sphere.GetRadiusAttr().Set(radius_m)
                droplet.radius_m = radius_m
                updated_geometry += 1
        self._active_droplets = active_droplets
        self.geometry_update_count += updated_geometry

    def reset(self, clear_geometry=False):
        if not clear_geometry:
            return
        for path in self._created_paths:
            self.stage.RemovePrim(path)
        self.marker_count = 0
        self.skipped_volume_mm3 = 0.0
        self.geometry_update_count = 0
        self._created_paths = []
        self._chunk_index = 0
        self._curve_chunks = []
        self._current_chunk = None
        self._active_droplets = []
        self._time_s = 0.0

    def flush(self):
        return None


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
    deposition_statistics,
    deposited_markers,
    deposited_particles,
    visual_geometry_updates,
    sink_unrepresented_volume_mm3,
    particle_limit_skips,
):
    rms = math.sqrt(sum_squared_error / max(1, sample_count))
    tracking_passed = (
        maximum_error <= MAX_ACCEPTABLE_TRACKING_ERROR_RAD
        and rms <= MAX_ACCEPTABLE_RMS_TRACKING_ERROR_RAD
    )
    unrepresented_volume_mm3 = (
        deposition_statistics.unrepresented_volume_mm3
        + sink_unrepresented_volume_mm3
    )
    represented_volume_mm3 = max(
        0.0,
        deposition_statistics.emitted_volume_mm3 - sink_unrepresented_volume_mm3,
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
        "visual_geometry_updates": visual_geometry_updates,
        "particle_volume_mm3": PARTICLE_VOLUME_MM3 if DEPOSITION_MODE == "particles" else None,
        "tcp_pose_samples": deposition_statistics.observed_samples,
        "deposition_interval_updates": deposition_statistics.interval_updates,
        "deposition_segments": deposition_statistics.emitted_segments,
        "deposition_points": deposition_statistics.emitted_points,
        "deposition_discontinuities": deposition_statistics.discontinuities,
        "commanded_deposition_volume_mm3": deposition_statistics.commanded_volume_mm3,
        "emitted_deposition_volume_mm3": deposition_statistics.emitted_volume_mm3,
        "represented_deposition_volume_mm3": represented_volume_mm3,
        "unrepresented_volume_mm3": unrepresented_volume_mm3,
        "particles_skipped_due_to_limit": particle_limit_skips,
        "skipped_deposition_points_due_to_tracking": 0,
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
flow_schedule = FlowSchedule.from_trajectory_points(trajectory)

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(ROBOT_USD, ROBOT_PRIM)
enable_robot_physics_variants(world.stage, ROBOT_PRIM)
repair_extruder_mount(world.stage, ROBOT_PRIM)
ARTICULATION_PRIM = find_or_create_articulation_root(world.stage, ROBOT_PRIM)
ensure_deposition_root(world.stage)
create_print_bed(world.stage)
particle_emitter = None
visual_bead_sink = None
if DEPOSITION_MODE == "particles":
    physics_scene_path = configure_gpu_particle_scene(world.stage)
    particle_emitter = PhysxExtrusionEmitter(world.stage, physics_scene_path)
    estimated_particles = math.ceil(
        sum(interval.volume_mm3 for interval in flow_schedule.intervals)
        / PARTICLE_VOLUME_MM3
    )
    print(
        f"PhysX PBD extrusion: estimated {estimated_particles} particles; "
        f"capacity {MAX_DEPOSITION_PARTICLES}; density "
        f"{MATERIAL_DENSITY_KG_M3:.6g} kg/m^3; material "
        f"{MATERIAL_PROFILE_ID}"
    )
    if PARTICLE_ISOSURFACE_ENABLED:
        print("particle isosurface: enabled (optional render-only GPU path)")
    else:
        print(
            "particle isosurface: disabled "
            "(safe default; PBD physics remains enabled)"
        )
else:
    bead_evolution_model = BeadEvolutionModel(
        spreading_ratio=MATERIAL_SPREADING_RATIO,
        spreading_time_s=MATERIAL_SPREADING_TIME_S,
        shrinkage_fraction=MATERIAL_SHRINKAGE_FRACTION,
        shrinkage_time_s=MATERIAL_SHRINKAGE_TIME_S,
    )
    visual_bead_sink = VisualBeadSink(world.stage, bead_evolution_model)
    print(
        f"visual bead evolution: spreading ratio "
        f"{MATERIAL_SPREADING_RATIO:.6g} over "
        f"{MATERIAL_SPREADING_TIME_S:.6g} s; shrinkage "
        f"{MATERIAL_SHRINKAGE_FRACTION:.6g} over "
        f"{MATERIAL_SHRINKAGE_TIME_S:.6g} s"
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
tcp_pose_reader = RobotTcpPoseReader(world.stage, ROBOT_PRIM)
deposition_sink = particle_emitter or visual_bead_sink
deposition_manager = DepositionManager(
    deposition_sink,
    max_tcp_step_m=MAX_TCP_STEP_M,
)
deposition_manager.reset(
    pose=tcp_pose_reader.read_pose(),
    simulation_time_s=0.0,
)
cursor = 0
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
        if visual_bead_sink is not None:
            # Age material that already existed during this completed step;
            # beads emitted below are born at the new replay time with age zero.
            visual_bead_sink.advance_to(replay_time_s)
        # Sample the realized nozzle pose after every completed physics step;
        # the manager integrates scheduled material flow over the same interval.
        deposition_manager.update_from_schedule(
            tcp_pose_reader.read_pose(),
            replay_time_s,
            flow_schedule,
        )
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
        if command_time_s >= trajectory[-1]["time_from_start_s"] + SETTLING_TIME_S:
            break
finally:
    tracking_file.close()
    deposition_manager.close()
    deposition_statistics = deposition_manager.statistics
    sink_unrepresented_volume_mm3 = (
        particle_emitter.volume_remainder_mm3
        + particle_emitter.skipped_particles * PARTICLE_VOLUME_MM3
        if particle_emitter is not None
        else visual_bead_sink.skipped_volume_mm3
    )
    write_tracking_outputs(
        tracking_samples,
        tracking_sample_count,
        sum_squared_error,
        maximum_tracking_error,
        initialization_duration_s,
        initialization_error_rad,
        deposition_statistics,
        visual_bead_sink.marker_count if visual_bead_sink is not None else 0,
        particle_emitter.particle_count if particle_emitter is not None else 0,
        visual_bead_sink.geometry_update_count if visual_bead_sink is not None else 0,
        sink_unrepresented_volume_mm3,
        particle_emitter.skipped_particles if particle_emitter is not None else 0,
    )
    simulation_app.close()
