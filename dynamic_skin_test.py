import csv
import math
import os
import random

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt


# Arm settings
_ARM_LENGTH = 0.27
_SHOULDER_WIDTH = 0.135
_END_WIDTH = 0.090
_SHOULDER_HEIGHT = 0.115
_END_HEIGHT = 0.075

_BODY_X_STEPS = 80
_BODY_RING_STEPS = 56


# Interface settings
_INTERFACE_RING_STEPS = 30
_INTERFACE_SEGMENTS = 72
_OUTER_RING_STEPS = 6

_INTERFACE_RADIUS_Y = _END_WIDTH / 2.0
_INTERFACE_RADIUS_Z = _END_HEIGHT / 2.0


# Wound settings
_WOUND_RADIUS_Y = 0.024
_WOUND_RADIUS_Z = 0.015
_WOUND_DEPTH = 0.007


# Breathing settings
_BREATHING_AMOUNT = 0.003
_BREATHING_PERIOD = 5.0


# Local deformation settings
_LOCAL_CENTER_Y = 0.010
_LOCAL_CENTER_Z = 0.000

_LOCAL_AMOUNT = 0.003
_LOCAL_SIZE_Y = 0.016
_LOCAL_SIZE_Z = 0.013
_LOCAL_PERIOD = 3.0
_LOCAL_PHASE = math.pi / 3.0


# Body shift settings
_SHIFT_MIN_TIME = 5.0
_SHIFT_MAX_TIME = 20.0

_SHIFT_MIN_DISTANCE = 0.002
_SHIFT_MAX_DISTANCE = 0.005
_SHIFT_MOVE_TIME = 1.5


# Printing settings
_PRINT_SPEED = 0.018
_PRINT_RATE = 0.012

_BEAD_SIZE_Y = 0.004
_BEAD_SIZE_Z = 0.004

_PRINT_START_Y = -0.026
_PRINT_END_Y = 0.026
_PRINT_START_Z = -0.017
_PRINT_END_Z = 0.017

_PRINT_LANES = 9
_PRINT_HEAD_GAP = 0.010


# Paths
_ROOT_PATH = "/World/UpperArmResidual"
_BODY_PATH = _ROOT_PATH + "/ArmBody"
_INTERFACE_PATH = _ROOT_PATH + "/Interface"
_PRINT_HEAD_PATH = _ROOT_PATH + "/PrintHead"

_SKIN_MATERIAL_PATH = "/World/Materials/SkinMaterial"
_WOUND_MATERIAL_PATH = "/World/Materials/WoundMaterial"
_PRINT_HEAD_MATERIAL_PATH = "/World/Materials/PrintHeadMaterial"
_GROUND_MATERIAL_PATH = "/World/Materials/GroundMaterial"
_BACKGROUND_MATERIAL_PATH = "/World/Materials/BackgroundMaterial"

_ENVIRONMENT_PATH = "/World/DemoEnvironment"
_GROUND_PATH = _ENVIRONMENT_PATH + "/Ground"
_BACKGROUND_PATH = _ENVIRONMENT_PATH + "/Background"
_DOME_LIGHT_PATH = _ENVIRONMENT_PATH + "/DomeLight"
_KEY_LIGHT_PATH = _ENVIRONMENT_PATH + "/KeyLight"

_OUTPUT_DIR = r"C:\yifeiDev\dynamic_skin_project\output"


# Animation data
_update_sub = None
_time = 0.0

_interface_start_points = None
_interface_y = None
_interface_z = None
_interface_weight = None
_deposit_height = None


# Body shift data
_shift_wait_time = 0.0
_shift_move_time = 0.0
_is_shifting = False

_shift_start = Gf.Vec3d(0.0, 0.0, 0.0)
_shift_target = Gf.Vec3d(0.0, 0.0, 0.0)
_shift_current = Gf.Vec3d(0.0, 0.0, 0.0)


# Printing data
_print_path = []
_print_part = 0
_print_part_distance = 0.0

_print_y = 0.0
_print_z = 0.0

_printing = False
_data_saved = False


def stop_animation():
    global _update_sub

    _update_sub = None
    print("Animation stopped.")


def create_material(stage, path, color, roughness):
    material = UsdShade.Material.Define(
        stage,
        Sdf.Path(path),
    )

    shader = UsdShade.Shader.Define(
        stage,
        Sdf.Path(path + "/Shader"),
    )

    shader.CreateIdAttr("UsdPreviewSurface")

    shader.CreateInput(
        "diffuseColor",
        Sdf.ValueTypeNames.Color3f,
    ).Set(color)

    shader.CreateInput(
        "roughness",
        Sdf.ValueTypeNames.Float,
    ).Set(roughness)

    shader.CreateInput(
        "metallic",
        Sdf.ValueTypeNames.Float,
    ).Set(0.0)

    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )

    return material


def clear_old_scene(stage):
    # Delete old objects before making a new scene
    for path in [
        _ROOT_PATH,
        _ENVIRONMENT_PATH,
        _SKIN_MATERIAL_PATH,
        _WOUND_MATERIAL_PATH,
        _PRINT_HEAD_MATERIAL_PATH,
        _GROUND_MATERIAL_PATH,
        _BACKGROUND_MATERIAL_PATH,
    ]:
        prim = stage.GetPrimAtPath(path)

        if prim.IsValid():
            stage.RemovePrim(path)


def create_root(stage):
    root = UsdGeom.Xform.Define(
        stage,
        Sdf.Path(_ROOT_PATH),
    )

    xform = UsdGeom.Xformable(
        root.GetPrim()
    )

    xform.ClearXformOpOrder()

    xform.AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, 0.0)
    )


def create_arm_body(stage, skin_material):
    points = []
    face_indices = []
    face_counts = []

    start_x = -_ARM_LENGTH
    end_x = 0.0

    # Create body rings
    for col in range(_BODY_X_STEPS):
        ratio = col / (_BODY_X_STEPS - 1)
        x = start_x + _ARM_LENGTH * ratio

        # Shoulder side is larger
        shape = ratio ** 0.78

        width = (
            _SHOULDER_WIDTH
            + (_END_WIDTH - _SHOULDER_WIDTH) * shape
        )

        height = (
            _SHOULDER_HEIGHT
            + (_END_HEIGHT - _SHOULDER_HEIGHT) * shape
        )

        half_width = width / 2.0
        half_height = height / 2.0

        # Small curve for more human shape
        center_z = 0.008 * math.sin(
            math.pi * ratio
        )

        # Small shoulder round part
        shoulder_bump = 0.008 * math.exp(
            -(ratio * ratio)
            / (2.0 * 0.20 * 0.20)
        )

        for row in range(_BODY_RING_STEPS):
            angle = (
                2.0
                * math.pi
                * row
                / _BODY_RING_STEPS
            )

            sin_a = math.sin(angle)
            cos_a = math.cos(angle)

            y = half_width * sin_a
            z = center_z + half_height * cos_a

            upper = max(0.0, cos_a)
            lower = max(0.0, -cos_a)

            # Shoulder is a little rounder
            z += shoulder_bump * upper

            # Bottom is a little flat
            z += 0.006 * lower * lower

            # Small side difference
            z += 0.0015 * sin_a * upper

            points.append(
                Gf.Vec3f(
                    float(x),
                    float(y),
                    float(z),
                )
            )

    # Connect body rings
    for col in range(_BODY_X_STEPS - 1):
        for row in range(_BODY_RING_STEPS):
            next_row = (
                row + 1
            ) % _BODY_RING_STEPS

            p1 = col * _BODY_RING_STEPS + row
            p2 = col * _BODY_RING_STEPS + next_row
            p3 = (
                col + 1
            ) * _BODY_RING_STEPS + row
            p4 = (
                col + 1
            ) * _BODY_RING_STEPS + next_row

            face_indices.extend([
                p1, p3, p4,
                p1, p4, p2,
            ])

            face_counts.extend([3, 3])

    # Close shoulder side
    shoulder_center = len(points)

    points.append(
        Gf.Vec3f(
            float(start_x),
            0.0,
            0.0,
        )
    )

    for row in range(_BODY_RING_STEPS):
        next_row = (
            row + 1
        ) % _BODY_RING_STEPS

        face_indices.extend([
            shoulder_center,
            next_row,
            row,
        ])

        face_counts.append(3)

    mesh = UsdGeom.Mesh.Define(
        stage,
        Sdf.Path(_BODY_PATH),
    )

    point_array = Vt.Vec3fArray(points)

    mesh.GetPointsAttr().Set(point_array)

    mesh.GetFaceVertexIndicesAttr().Set(
        Vt.IntArray(face_indices)
    )

    mesh.GetFaceVertexCountsAttr().Set(
        Vt.IntArray(face_counts)
    )

    mesh.GetSubdivisionSchemeAttr().Set(
        UsdGeom.Tokens.none
    )

    mesh.GetDoubleSidedAttr().Set(True)

    mesh.GetExtentAttr().Set(
        UsdGeom.PointBased.ComputeExtent(
            point_array
        )
    )

    UsdShade.MaterialBindingAPI(
        mesh.GetPrim()
    ).Bind(skin_material)




def create_interface(stage, wound_material):
    global _interface_start_points
    global _interface_y
    global _interface_z
    global _interface_weight
    global _deposit_height

    points = []
    interface_y = []
    interface_z = []
    area_weight = []

    face_indices = []
    face_counts = []

    half_width = _END_WIDTH / 2.0
    half_height = _END_HEIGHT / 2.0

    # Center point of the wound
    points.append(
        Gf.Vec3f(
            -_WOUND_DEPTH,
            0.0,
            0.0,
        )
    )

    interface_y.append(0.0)
    interface_z.append(0.0)
    area_weight.append(1.0)

    # Create rings that match the real arm-end contour
    for ring in range(1, _INTERFACE_RING_STEPS + 1):
        radius = ring / _INTERFACE_RING_STEPS

        for seg in range(_INTERFACE_SEGMENTS):
            angle = (
                2.0
                * math.pi
                * seg
                / _INTERFACE_SEGMENTS
            )

            sin_a = math.sin(angle)
            cos_a = math.cos(angle)

            # Same end shape used by the arm body
            outer_y = half_width * sin_a
            outer_z = half_height * cos_a

            upper = max(0.0, cos_a)
            lower = max(0.0, -cos_a)

            outer_z += 0.006 * lower * lower
            outer_z += 0.0015 * sin_a * upper

            # Move from center to the exact body boundary
            y = outer_y * radius
            z = outer_z * radius

            wound_value = (
                (y / _WOUND_RADIUS_Y) ** 4
                + (z / _WOUND_RADIUS_Z) ** 4
            )

            edge_weight = max(
                0.0,
                1.0 - radius * radius,
            )

            motion_weight = (
                edge_weight
                * edge_weight
                * (3.0 - 2.0 * edge_weight)
            )

            # Boundary stays at x = 0, center goes inward
            wound_x = (
                -_WOUND_DEPTH
                * math.exp(-wound_value)
                * motion_weight
            )

            points.append(
                Gf.Vec3f(
                    float(wound_x),
                    float(y),
                    float(z),
                )
            )

            interface_y.append(float(y))
            interface_z.append(float(z))
            area_weight.append(float(motion_weight))

    # Center triangles
    for seg in range(_INTERFACE_SEGMENTS):
        next_seg = (
            seg + 1
        ) % _INTERFACE_SEGMENTS

        p1 = 0
        p2 = 1 + seg
        p3 = 1 + next_seg

        face_indices.extend([
            p1,
            p2,
            p3,
        ])

        face_counts.append(3)

    # Connect rings
    for ring in range(2, _INTERFACE_RING_STEPS + 1):
        prev_start = (
            1
            + (ring - 2) * _INTERFACE_SEGMENTS
        )

        curr_start = (
            1
            + (ring - 1) * _INTERFACE_SEGMENTS
        )

        for seg in range(_INTERFACE_SEGMENTS):
            next_seg = (
                seg + 1
            ) % _INTERFACE_SEGMENTS

            p1 = prev_start + seg
            p2 = prev_start + next_seg
            p3 = curr_start + seg
            p4 = curr_start + next_seg

            face_indices.extend([
                p1, p3, p4,
                p1, p4, p2,
            ])

            face_counts.extend([3, 3])

    mesh = UsdGeom.Mesh.Define(
        stage,
        Sdf.Path(_INTERFACE_PATH),
    )

    point_array = Vt.Vec3fArray(points)

    mesh.GetPointsAttr().Set(point_array)

    mesh.GetFaceVertexIndicesAttr().Set(
        Vt.IntArray(face_indices)
    )

    mesh.GetFaceVertexCountsAttr().Set(
        Vt.IntArray(face_counts)
    )

    mesh.GetSubdivisionSchemeAttr().Set(
        UsdGeom.Tokens.none
    )

    mesh.GetDoubleSidedAttr().Set(True)

    mesh.GetExtentAttr().Set(
        UsdGeom.PointBased.ComputeExtent(
            point_array
        )
    )

    UsdShade.MaterialBindingAPI(
        mesh.GetPrim()
    ).Bind(wound_material)

    _interface_start_points = [
        Gf.Vec3f(
            float(point[0]),
            float(point[1]),
            float(point[2]),
        )
        for point in points
    ]

    _interface_y = interface_y
    _interface_z = interface_z
    _interface_weight = area_weight

    _deposit_height = [
        0.0 for _ in points
    ]

def create_print_head(stage):
    sphere = UsdGeom.Sphere.Define(
        stage,
        Sdf.Path(_PRINT_HEAD_PATH),
    )

    sphere.GetRadiusAttr().Set(0.004)
    sphere.GetVisibilityAttr().Set("invisible")

    xform = UsdGeom.Xformable(
        sphere.GetPrim()
    )

    xform.ClearXformOpOrder()

    xform.AddTranslateOp().Set(
        Gf.Vec3d(
            _PRINT_HEAD_GAP,
            0.0,
            0.0,
        )
    )

    material = create_material(
        stage,
        _PRINT_HEAD_MATERIAL_PATH,
        Gf.Vec3f(
            1.0,
            0.8,
            0.1,
        ),
        0.35,
    )

    UsdShade.MaterialBindingAPI(
        sphere.GetPrim()
    ).Bind(material)



def create_environment(stage):
    UsdGeom.Xform.Define(
        stage,
        Sdf.Path(_ENVIRONMENT_PATH),
    )

    ground_material = create_material(
        stage,
        _GROUND_MATERIAL_PATH,
        Gf.Vec3f(0.72, 0.72, 0.72),
        0.82,
    )

    background_material = create_material(
        stage,
        _BACKGROUND_MATERIAL_PATH,
        Gf.Vec3f(0.86, 0.86, 0.86),
        0.90,
    )

    # Ground under the arm
    ground = UsdGeom.Cube.Define(
        stage,
        Sdf.Path(_GROUND_PATH),
    )

    ground.GetSizeAttr().Set(1.0)

    ground_xform = UsdGeom.Xformable(
        ground.GetPrim()
    )

    ground_xform.ClearXformOpOrder()
    ground_xform.AddTranslateOp().Set(
        Gf.Vec3d(-0.13, 0.0, -0.095)
    )
    ground_xform.AddScaleOp().Set(
        Gf.Vec3f(0.44, 0.34, 0.012)
    )

    UsdShade.MaterialBindingAPI(
        ground.GetPrim()
    ).Bind(ground_material)

    # Large wall behind the model
    background = UsdGeom.Cube.Define(
        stage,
        Sdf.Path(_BACKGROUND_PATH),
    )

    background.GetSizeAttr().Set(1.0)

    background_xform = UsdGeom.Xformable(
        background.GetPrim()
    )

    background_xform.ClearXformOpOrder()
    background_xform.AddTranslateOp().Set(
        Gf.Vec3d(-0.48, 0.0, 0.10)
    )
    background_xform.AddScaleOp().Set(
        Gf.Vec3f(0.012, 0.55, 0.42)
    )

    UsdShade.MaterialBindingAPI(
        background.GetPrim()
    ).Bind(background_material)

    # Soft light for the full scene
    dome_light = UsdLux.DomeLight.Define(
        stage,
        Sdf.Path(_DOME_LIGHT_PATH),
    )

    dome_light.GetIntensityAttr().Set(650.0)
    dome_light.GetColorAttr().Set(
        Gf.Vec3f(1.0, 0.96, 0.92)
    )

    # Main light from upper front
    key_light = UsdLux.DistantLight.Define(
        stage,
        Sdf.Path(_KEY_LIGHT_PATH),
    )

    key_light.GetIntensityAttr().Set(1200.0)
    key_light.GetAngleAttr().Set(4.0)
    key_light.GetColorAttr().Set(
        Gf.Vec3f(1.0, 0.95, 0.90)
    )

    key_xform = UsdGeom.Xformable(
        key_light.GetPrim()
    )

    key_xform.ClearXformOpOrder()
    key_xform.AddRotateXYZOp().Set(
        Gf.Vec3f(-35.0, -25.0, 20.0)
    )

    print("Light gray environment created.")


def create_scene():
    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("No stage found.")

    clear_old_scene(stage)
    create_environment(stage)
    create_root(stage)

    skin_material = create_material(
        stage,
        _SKIN_MATERIAL_PATH,
        Gf.Vec3f(
            0.74,
            0.46,
            0.38,
        ),
        0.68,
    )

    wound_material = create_material(
        stage,
        _WOUND_MATERIAL_PATH,
        Gf.Vec3f(
            0.70,
            0.30,
            0.32,
        ),
        0.76,
    )

    create_arm_body(
        stage,
        skin_material,
    )

    create_interface(
        stage,
        wound_material,
    )

    create_print_head(stage)

    print("Upper arm residual created.")
    print("Only one yellow print head should be visible.")


def get_breathing(t):
    speed = (
        2.0
        * math.pi
        / _BREATHING_PERIOD
    )

    return (
        _BREATHING_AMOUNT
        * math.sin(speed * t)
    )


def get_local_move(y, z, t):
    dy = y - _LOCAL_CENTER_Y
    dz = z - _LOCAL_CENTER_Z

    local_weight = math.exp(
        -(
            (dy * dy)
            / (
                2.0
                * _LOCAL_SIZE_Y
                * _LOCAL_SIZE_Y
            )
            +
            (dz * dz)
            / (
                2.0
                * _LOCAL_SIZE_Z
                * _LOCAL_SIZE_Z
            )
        )
    )

    speed = (
        2.0
        * math.pi
        / _LOCAL_PERIOD
    )

    wave = 0.5 + 0.5 * math.sin(
        speed * t + _LOCAL_PHASE
    )

    return (
        _LOCAL_AMOUNT
        * local_weight
        * wave
    )


def make_new_shift():
    global _shift_start
    global _shift_target
    global _shift_move_time
    global _is_shifting

    _shift_start = Gf.Vec3d(
        _shift_current[0],
        _shift_current[1],
        _shift_current[2],
    )

    angle = random.uniform(
        0.0,
        2.0 * math.pi,
    )

    distance = random.uniform(
        _SHIFT_MIN_DISTANCE,
        _SHIFT_MAX_DISTANCE,
    )

    move_y = math.cos(angle) * distance
    move_z = math.sin(angle) * distance

    move_x = random.uniform(
        -_SHIFT_MIN_DISTANCE / 2.0,
        _SHIFT_MIN_DISTANCE / 2.0,
    )

    _shift_target = Gf.Vec3d(
        move_x,
        move_y,
        move_z,
    )

    _shift_move_time = 0.0
    _is_shifting = True

    print("New body shift.")


def update_body_shift(dt):
    global _shift_wait_time
    global _shift_move_time
    global _shift_current
    global _is_shifting

    if not _is_shifting:
        _shift_wait_time -= dt

        if _shift_wait_time <= 0.0:
            make_new_shift()

        return

    _shift_move_time += dt

    amount = min(
        _shift_move_time / _SHIFT_MOVE_TIME,
        1.0,
    )

    smooth = (
        amount
        * amount
        * (3.0 - 2.0 * amount)
    )

    _shift_current = Gf.Vec3d(
        _shift_start[0]
        + (
            _shift_target[0]
            - _shift_start[0]
        ) * smooth,

        _shift_start[1]
        + (
            _shift_target[1]
            - _shift_start[1]
        ) * smooth,

        _shift_start[2]
        + (
            _shift_target[2]
            - _shift_start[2]
        ) * smooth,
    )

    if amount >= 1.0:
        _is_shifting = False

        _shift_wait_time = random.uniform(
            _SHIFT_MIN_TIME,
            _SHIFT_MAX_TIME,
        )


def set_root_position():
    stage = omni.usd.get_context().get_stage()

    prim = stage.GetPrimAtPath(
        _ROOT_PATH
    )

    if not prim.IsValid():
        return

    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()

    if len(ops) == 0:
        return

    breathing_x = get_breathing(_time)

    ops[0].Set(
        Gf.Vec3d(
            _shift_current[0] + breathing_x,
            _shift_current[1],
            _shift_current[2],
        )
    )

def make_print_path():
    global _print_path

    _print_path = []

    for lane in range(_PRINT_LANES):
        z = (
            _PRINT_START_Z
            + (
                _PRINT_END_Z
                - _PRINT_START_Z
            )
            * lane
            / (_PRINT_LANES - 1)
        )

        if lane % 2 == 0:
            start_y = _PRINT_START_Y
            end_y = _PRINT_END_Y
        else:
            start_y = _PRINT_END_Y
            end_y = _PRINT_START_Y

        if lane == 0:
            _print_path.append(
                (start_y, z)
            )

        _print_path.append(
            (end_y, z)
        )

        if lane < _PRINT_LANES - 1:
            next_z = (
                _PRINT_START_Z
                + (
                    _PRINT_END_Z
                    - _PRINT_START_Z
                )
                * (lane + 1)
                / (_PRINT_LANES - 1)
            )

            _print_path.append(
                (end_y, next_z)
            )


def reset_printing():
    global _print_part
    global _print_part_distance
    global _print_y
    global _print_z
    global _printing
    global _data_saved

    make_print_path()

    _print_part = 0
    _print_part_distance = 0.0

    _print_y = _print_path[0][0]
    _print_z = _print_path[0][1]

    _printing = True
    _data_saved = False

    print("Printing started.")


def add_material(y, z, dt):
    if _deposit_height is None:
        return

    for index in range(
        len(_interface_start_points)
    ):
        dy = _interface_y[index] - y
        dz = _interface_z[index] - z

        weight = math.exp(
            -(
                (dy * dy)
                / (
                    2.0
                    * _BEAD_SIZE_Y
                    * _BEAD_SIZE_Y
                )
                +
                (dz * dz)
                / (
                    2.0
                    * _BEAD_SIZE_Z
                    * _BEAD_SIZE_Z
                )
            )
        )

        add_height = (
            _PRINT_RATE
            * weight
            * dt
        )

        start_x = float(
            _interface_start_points[index][0]
        )

        max_height = -start_x

        new_height = (
            _deposit_height[index]
            + add_height
        )

        _deposit_height[index] = min(
            new_height,
            max_height,
        )


def update_printing(dt):
    global _print_part
    global _print_part_distance
    global _print_y
    global _print_z
    global _printing
    global _data_saved

    if not _printing:
        if not _data_saved:
            save_interface_points()
            _data_saved = True

        return

    if _print_part >= len(_print_path) - 1:
        _printing = False
        print("Printing finished.")
        return

    start = _print_path[_print_part]
    end = _print_path[_print_part + 1]

    dy = end[0] - start[0]
    dz = end[1] - start[1]

    part_length = math.sqrt(
        dy * dy + dz * dz
    )

    if part_length <= 0.0:
        _print_part += 1
        _print_part_distance = 0.0
        return

    _print_part_distance += (
        _PRINT_SPEED * dt
    )

    amount = min(
        _print_part_distance / part_length,
        1.0,
    )

    _print_y = start[0] + dy * amount
    _print_z = start[1] + dz * amount

    add_material(
        _print_y,
        _print_z,
        dt,
    )

    if amount >= 1.0:
        _print_part += 1
        _print_part_distance = 0.0


def get_near_point_index(y, z):
    best_index = 0
    best_distance = None

    for index in range(
        len(_interface_start_points)
    ):
        dy = _interface_y[index] - y
        dz = _interface_z[index] - z

        distance = dy * dy + dz * dz

        if (
            best_distance is None
            or distance < best_distance
        ):
            best_distance = distance
            best_index = index

    return best_index


def update_print_head():
    stage = omni.usd.get_context().get_stage()

    prim = stage.GetPrimAtPath(
        _PRINT_HEAD_PATH
    )

    if not prim.IsValid():
        return

    index = get_near_point_index(
        _print_y,
        _print_z,
    )

    point = _interface_start_points[index]

    local_move = (
        get_local_move(
            _interface_y[index],
            _interface_z[index],
            _time,
        )
        * _interface_weight[index]
    )

    x = (
        float(point[0])
        + local_move
        + _deposit_height[index]
        + _PRINT_HEAD_GAP
    )

    y = float(point[1])
    z = float(point[2])

    xform = UsdGeom.Xformable(
        prim
    )

    ops = xform.GetOrderedXformOps()

    if len(ops) > 0:
        ops[0].Set(
            Gf.Vec3d(
                x,
                y,
                z,
            )
        )


def save_interface_points():
    stage = omni.usd.get_context().get_stage()

    prim = stage.GetPrimAtPath(
        _INTERFACE_PATH
    )

    if not prim.IsValid():
        print("Interface not found.")
        return

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()

    if points is None:
        print("No interface points.")
        return

    os.makedirs(
        _OUTPUT_DIR,
        exist_ok=True,
    )

    file_path = os.path.join(
        _OUTPUT_DIR,
        "upper_arm_interface_points.csv",
    )

    with open(
        file_path,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "index",
            "x",
            "y",
            "z",
            "deposit_height",
            "area_weight",
        ])

        for index, point in enumerate(points):
            writer.writerow([
                index,
                float(point[0]),
                float(point[1]),
                float(point[2]),
                float(_deposit_height[index]),
                float(_interface_weight[index]),
            ])

    print("Interface points saved:")
    print(file_path)


def update_interface(event):
    global _time

    if _interface_start_points is None:
        return

    dt = float(
        event.payload.get(
            "dt",
            0.0,
        )
    )

    _time += dt

    update_printing(dt)
    update_body_shift(dt)
    set_root_position()

    stage = omni.usd.get_context().get_stage()

    prim = stage.GetPrimAtPath(
        _INTERFACE_PATH
    )

    if not prim.IsValid():
        return

    mesh = UsdGeom.Mesh(prim)
    new_points = []

    for index, point in enumerate(
        _interface_start_points
    ):
        y = float(point[1])
        z = float(point[2])

        local_move = get_local_move(
            y,
            z,
            _time,
        )

        move_x = (
            local_move
            * _interface_weight[index]
        )

        x = (
            float(point[0])
            + move_x
            + _deposit_height[index]
        )

        new_points.append(
            Gf.Vec3f(
                x,
                y,
                z,
            )
        )

    point_array = Vt.Vec3fArray(
        new_points
    )

    mesh.GetPointsAttr().Set(
        point_array
    )

    mesh.GetExtentAttr().Set(
        UsdGeom.PointBased.ComputeExtent(
            point_array
        )
    )

    update_print_head()


def reset_shift():
    global _shift_wait_time
    global _shift_move_time
    global _is_shifting
    global _shift_start
    global _shift_target
    global _shift_current

    _shift_wait_time = random.uniform(
        _SHIFT_MIN_TIME,
        _SHIFT_MAX_TIME,
    )

    _shift_move_time = 0.0
    _is_shifting = False

    _shift_start = Gf.Vec3d(
        0.0,
        0.0,
        0.0,
    )

    _shift_target = Gf.Vec3d(
        0.0,
        0.0,
        0.0,
    )

    _shift_current = Gf.Vec3d(
        0.0,
        0.0,
        0.0,
    )


def start_animation():
    global _update_sub
    global _time

    stop_animation()

    _time = 0.0

    reset_shift()
    reset_printing()

    update_stream = (
        omni.kit.app
        .get_app()
        .get_update_event_stream()
    )

    _update_sub = (
        update_stream.create_subscription_to_pop(
            update_interface,
            name="Upper Arm Residual Update",
        )
    )

    print("Animation started.")


def main():
    stop_animation()
    create_scene()
    start_animation()


main()
