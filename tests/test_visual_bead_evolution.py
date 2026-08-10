from __future__ import annotations

import ast
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotic_printing_platform.exporters.isaac import export_isaac_bundle
from robotic_printing_platform.extrusion import (
    BeadEvolutionModel,
    MaterialProfile,
    TcpPose,
)
from robotic_printing_platform.robots.franka_panda import (
    IKConfig,
    IKReport,
    RobotTrajectory,
    TrajectoryPoint,
)


class _Attribute:
    def __init__(self, value=None) -> None:
        self.value = value
        self.set_calls = []

    def Set(self, value):
        # Copy array values so mutations that omit a USD Set call cannot make
        # the fake attribute appear to have updated successfully.
        self.value = list(value) if isinstance(value, list) else value
        self.set_calls.append(self.value)


class _Prim:
    pass


class _Curve:
    def __init__(self) -> None:
        self.counts = _Attribute()
        self.points = _Attribute()
        self.widths = _Attribute()
        self.prim = _Prim()

    def CreateTypeAttr(self, value):
        return _Attribute(value)

    def CreateWrapAttr(self, value):
        return _Attribute(value)

    def CreateCurveVertexCountsAttr(self, value):
        self.counts = _Attribute(value)
        return self.counts

    def CreatePointsAttr(self, value):
        self.points = _Attribute(value)
        return self.points

    def CreateWidthsAttr(self, value):
        self.widths = _Attribute(value)
        return self.widths

    def CreateDisplayColorAttr(self, value):
        return _Attribute(value)

    def SetWidthsInterpolation(self, value):
        self.widths_interpolation = value

    def GetPrim(self):
        return self.prim

    def GetCurveVertexCountsAttr(self):
        return self.counts

    def GetPointsAttr(self):
        return self.points

    def GetWidthsAttr(self):
        return self.widths


class _Sphere:
    def __init__(self) -> None:
        self.radius = _Attribute()
        self.prim = _Prim()

    def CreateRadiusAttr(self, value):
        self.radius = _Attribute(value)
        return self.radius

    def AddTranslateOp(self):
        return _Attribute()

    def CreateDisplayColorAttr(self, value):
        return _Attribute(value)

    def GetPrim(self):
        return self.prim

    def GetRadiusAttr(self):
        return self.radius


class _Stage:
    def __init__(self) -> None:
        self.curves = []
        self.spheres = []
        self.removed_paths = []

    def RemovePrim(self, path):
        self.removed_paths.append(path)


class _BasisCurves:
    @staticmethod
    def Define(stage, path):
        curve = _Curve()
        stage.curves.append(curve)
        return curve


class _Spheres:
    @staticmethod
    def Define(stage, path):
        sphere = _Sphere()
        stage.spheres.append(sphere)
        return sphere


class _Tokens:
    linear = "linear"
    nonperiodic = "nonperiodic"
    constant = "constant"
    uniform = "uniform"


class _UsdGeom:
    BasisCurves = _BasisCurves
    Sphere = _Spheres
    Tokens = _Tokens


class _Gf:
    @staticmethod
    def Vec3f(*values):
        return tuple(values)

    @staticmethod
    def Vec3d(*values):
        return tuple(values)


class _Sdf:
    @staticmethod
    def Path(value):
        return value


class _MaterialBinding:
    def __init__(self, prim) -> None:
        self.prim = prim

    def Bind(self, material):
        self.material = material


class _UsdShade:
    MaterialBindingAPI = _MaterialBinding


def _trajectory() -> RobotTrajectory:
    point = TrajectoryPoint(
        index=0,
        q=np.zeros(7),
        p=np.zeros(3),
        yaw=0.0,
        is_print=False,
        layer=0,
        seg_id=0,
        feed_m_s=0.01,
        de=0.0,
        material="test hydrogel",
        extrusion_volume_mm3=0.0,
        extrusion_mass_g=0.0,
        pos_error_m=0.0,
        rot_error_rad=0.0,
        time_from_start_s=0.0,
    )
    return RobotTrajectory(
        points=[point],
        report=IKReport(True, 1, 1, 1, [], [], 0.0, 0.0),
        config=IKConfig(),
    )


def _generated_visual_sink(source: str, *, segments_per_chunk: int):
    """Load only the generated visual sink against lightweight USD doubles."""

    selected_names = {
        "visual_bead_radius_m",
        "spawn_deposition_segment",
        "_VisualCurveChunk",
        "_VisualDroplet",
        "VisualBeadSink",
    }
    tree = ast.parse(source)
    selected_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in selected_names
    ]
    namespace = {
        "np": np,
        "math": math,
        "Sdf": _Sdf,
        "UsdGeom": _UsdGeom,
        "Gf": _Gf,
        "UsdShade": _UsdShade,
        "BEAD_COLOR": (1.0, 0.28, 0.03),
        "DEPOSITION_PRIM": "/World/PrintedMaterial",
        "PARTICLE_MATERIAL_PRIM": "/World/PrintedMaterial/Material",
        "MAX_DEPOSITION_MARKERS": 100,
        "VISUAL_SEGMENTS_PER_CHUNK": segments_per_chunk,
        "ensure_deposition_root": lambda stage: None,
        "create_preview_material": lambda stage, path: object(),
    }
    module = ast.Module(body=selected_nodes, type_ignores=[])
    exec(compile(module, "<generated-visual-sink>", "exec"), namespace)
    return namespace["VisualBeadSink"]


class GeneratedVisualBeadEvolutionTests(unittest.TestCase):
    def test_evolves_all_geometry_from_immutable_birth_state(self):
        material = MaterialProfile(
            profile_id="visual_test_hydrogel",
            name="visual test hydrogel",
            extrusion_mode="volumetric",
            spreading_ratio=1.5,
            spreading_time_s=2.0,
            shrinkage_fraction=0.2,
            shrinkage_time_s=4.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            bundle = export_isaac_bundle(
                _trajectory(),
                Path(directory),
                material_profile=material,
            )
            source = bundle["isaac_script"].read_text(encoding="utf-8")

        replay_loop = source.index("while simulation_app.is_running():")
        physics_step = source.index("world.step(render=True)", replay_loop)
        time_increment = source.index(
            "replay_time_s += world.get_physics_dt()", physics_step
        )
        evolution_update = source.index(
            "visual_bead_sink.advance_to(replay_time_s)", time_increment
        )
        deposition_update = source.index(
            "deposition_manager.update_from_schedule(", evolution_update
        )
        self.assertLess(physics_step, time_increment)
        self.assertLess(time_increment, evolution_update)
        self.assertLess(evolution_update, deposition_update)

        sink_type = _generated_visual_sink(source, segments_per_chunk=1)
        model = BeadEvolutionModel.from_material_profile(material)
        stage = _Stage()
        sink = sink_type(stage, model)
        first = TcpPose([0.0, 0.0, 0.0])
        second = TcpPose([0.01, 0.0, 0.0])
        third = TcpPose([0.02, 0.0, 0.0])

        sink.advance_to(1.0)
        sink.emit_segment(first, second, 10.0, 1.0)
        first_base_width = sink._curve_chunks[0].initial_widths[0]

        sink.advance_to(2.0)
        sink.emit_segment(second, third, 10.0, 1.0)
        sink.emit_point(third, 10.0, 1.0)
        second_base_width = sink._curve_chunks[1].initial_widths[0]
        droplet_base_radius = sink._active_droplets[0].initial_radius_m

        # The chunk size of one forces the first segment into a historical
        # chunk while the second remains current, with distinct birth ages.
        self.assertEqual(len(stage.curves), 2)
        self.assertEqual(len(stage.spheres), 1)
        sink.advance_to(3.0)
        first_scale = model.state_at(2.0).visual_radius_scale
        second_scale = model.state_at(1.0).visual_radius_scale
        self.assertAlmostEqual(
            stage.curves[0].widths.value[0], first_base_width * first_scale
        )
        self.assertAlmostEqual(
            stage.curves[1].widths.value[0], second_base_width * second_scale
        )
        self.assertAlmostEqual(
            stage.spheres[0].radius.value, droplet_base_radius * second_scale
        )

        # Repeating an absolute time is idempotent, and a later update is
        # always recomputed from the immutable deposited size rather than the
        # previously scaled value.
        values_at_three = (
            stage.curves[0].widths.value[0],
            stage.curves[1].widths.value[0],
            stage.spheres[0].radius.value,
        )
        updates_at_three = sink.geometry_update_count
        sink.advance_to(3.0)
        self.assertEqual(
            values_at_three,
            (
                stage.curves[0].widths.value[0],
                stage.curves[1].widths.value[0],
                stage.spheres[0].radius.value,
            ),
        )
        self.assertEqual(sink.geometry_update_count, updates_at_three)

        sink.advance_to(4.0)
        self.assertAlmostEqual(
            stage.curves[0].widths.value[0],
            first_base_width * model.state_at(3.0).visual_radius_scale,
        )
        self.assertAlmostEqual(
            stage.curves[1].widths.value[0],
            second_base_width * model.state_at(2.0).visual_radius_scale,
        )
        self.assertAlmostEqual(
            stage.spheres[0].radius.value,
            droplet_base_radius * model.state_at(2.0).visual_radius_scale,
        )

        created_path_count = len(sink._created_paths)
        sink.reset(clear_geometry=True)
        self.assertEqual(len(stage.removed_paths), created_path_count)
        self.assertEqual(sink.marker_count, 0)
        self.assertEqual(sink.geometry_update_count, 0)
        self.assertEqual(sink._curve_chunks, [])
        self.assertEqual(sink._active_droplets, [])
        self.assertEqual(sink._time_s, 0.0)


if __name__ == "__main__":
    unittest.main()
