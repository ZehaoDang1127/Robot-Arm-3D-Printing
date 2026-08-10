from __future__ import annotations

import ast
import math
import re
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
        self.value = self._copy(value)
        self.set_calls = []

    @staticmethod
    def _copy(value):
        if isinstance(value, list):
            return list(value)
        return value

    def Set(self, value):
        self.value = self._copy(value)
        self.set_calls.append(self._copy(value))


class _Prim:
    pass


class _Mesh:
    def __init__(self, path) -> None:
        self.path = path
        self.points = _Attribute()
        self.face_vertex_counts = _Attribute()
        self.face_vertex_indices = _Attribute()
        self.prim = _Prim()

    def CreatePointsAttr(self, value):
        self.points = _Attribute(value)
        return self.points

    def CreateFaceVertexCountsAttr(self, value):
        self.face_vertex_counts = _Attribute(value)
        return self.face_vertex_counts

    def CreateFaceVertexIndicesAttr(self, value):
        self.face_vertex_indices = _Attribute(value)
        return self.face_vertex_indices

    def CreateSubdivisionSchemeAttr(self, value):
        return _Attribute(value)

    def CreateDisplayColorAttr(self, value):
        return _Attribute(value)

    def CreateDoubleSidedAttr(self, value):
        return _Attribute(value)

    def CreateNormalsAttr(self, value):
        return _Attribute(value)

    def SetNormalsInterpolation(self, value):
        self.normals_interpolation = value

    def GetPointsAttr(self):
        return self.points

    def GetFaceVertexCountsAttr(self):
        return self.face_vertex_counts

    def GetFaceVertexIndicesAttr(self):
        return self.face_vertex_indices

    def GetPrim(self):
        return self.prim


class _Stage:
    def __init__(self) -> None:
        self.meshes = []
        self.removed_paths = []

    def RemovePrim(self, path):
        self.removed_paths.append(path)


class _Meshes:
    @staticmethod
    def Define(stage, path):
        mesh = _Mesh(path)
        stage.meshes.append(mesh)
        return mesh


class _Tokens:
    none = "none"
    faceVarying = "faceVarying"
    vertex = "vertex"


class _UsdGeom:
    Mesh = _Meshes
    Tokens = _Tokens


class _Gf:
    @staticmethod
    def Vec3f(*values):
        return tuple(float(value) for value in values)

    @staticmethod
    def Vec3d(*values):
        return tuple(float(value) for value in values)


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
        material="mesh test hydrogel",
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


def _generated_mesh_namespace(source: str):
    """Load the generated mesh sink against lightweight USD test doubles."""

    selected_names = {
        "bead_cross_section_frame",
        "elliptical_tube_points",
        "elliptical_tube_topology",
        "ellipsoid_points",
        "ellipsoid_topology",
        "mesh_enclosed_volume_m3",
        "_MeshBeadSegment",
        "_MeshBeadDroplet",
        "MeshBeadSink",
    }
    tree = ast.parse(source)
    selected_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in selected_names
    ]
    found_names = {node.name for node in selected_nodes}
    if found_names != selected_names:
        raise AssertionError(
            f"generated mesh definitions missing: {sorted(selected_names - found_names)}"
        )
    namespace = {
        "np": np,
        "math": math,
        "Sdf": _Sdf,
        "UsdGeom": _UsdGeom,
        "Gf": _Gf,
        "UsdShade": _UsdShade,
        "TcpPose": TcpPose,
        "BEAD_COLOR": (1.0, 0.28, 0.03),
        "DEPOSITION_PRIM": "/World/PrintedMaterial",
        "PARTICLE_MATERIAL_PRIM": "/World/PrintedMaterial/Material",
        "MAX_DEPOSITION_MARKERS": 100,
        "MESH_RING_SEGMENTS": 12,
        "MESH_DROPLET_LONGITUDE_SEGMENTS": 12,
        "MESH_DROPLET_LATITUDE_SEGMENTS": 6,
        "ensure_deposition_root": lambda stage: None,
        "create_preview_material": lambda stage, path: object(),
    }
    module = ast.Module(body=selected_nodes, type_ignores=[])
    exec(compile(module, "<generated-mesh-sink>", "exec"), namespace)
    return namespace


def _points_array(mesh: _Mesh) -> np.ndarray:
    return np.asarray(mesh.points.value, dtype=float)


def _assert_valid_topology(
    case: unittest.TestCase,
    topology,
    point_count: int,
) -> None:
    counts, indices = topology
    case.assertTrue(counts)
    case.assertTrue(all(int(count) >= 3 for count in counts))
    case.assertEqual(sum(int(count) for count in counts), len(indices))
    case.assertTrue(all(0 <= int(index) < point_count for index in indices))


class GeneratedMeshBeadGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        material = MaterialProfile(
            profile_id="mesh_test_hydrogel",
            name="mesh test hydrogel",
            extrusion_mode="volumetric",
            spreading_ratio=1.6,
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
            cls.source = bundle["isaac_script"].read_text(encoding="utf-8")
        cls.generated = _generated_mesh_namespace(cls.source)

    def test_mesh_is_default_visual_geometry_and_curve_remains_selectable(self):
        self.assertRegex(
            self.source,
            re.compile(
                r'os\.environ\.get\(\s*"RPP_VISUAL_BEAD_GEOMETRY"\s*,\s*'
                r'"mesh"\s*\)',
                re.MULTILINE,
            ),
        )
        self.assertIn('"curve"', self.source)
        self.assertIn("class MeshBeadSink", self.source)
        self.assertIn("class VisualBeadSink", self.source)
        self.assertIn("MeshBeadSink(world.stage, bead_evolution_model)", self.source)
        self.assertIn("VisualBeadSink(world.stage, bead_evolution_model)", self.source)

    def test_cross_section_frame_is_orthonormal_to_tcp_path(self):
        frame = self.generated["bead_cross_section_frame"]
        first = TcpPose([1.0, 2.0, 3.0])
        second = TcpPose([1.4, 2.3, 3.2])

        start, end, lateral, vertical, length = frame(first, second)
        tangent = (end - start) / length
        np.testing.assert_allclose(start, first.position_m)
        np.testing.assert_allclose(end, second.position_m)
        self.assertAlmostEqual(length, float(np.linalg.norm(end - start)))
        self.assertAlmostEqual(float(np.linalg.norm(lateral)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(vertical)), 1.0)
        self.assertAlmostEqual(float(np.dot(tangent, lateral)), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(tangent, vertical)), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(lateral, vertical)), 0.0, places=12)

        sloped_start = TcpPose([0.0, 0.0, 0.0])
        sloped_end = TcpPose([0.1, 0.0, 0.02])
        start, end, _, vertical, length = frame(sloped_start, sloped_end)
        tangent = (end - start) / length
        world_up = np.array([0.0, 0.0, 1.0])
        expected_vertical = world_up - np.dot(world_up, tangent) * tangent
        expected_vertical /= np.linalg.norm(expected_vertical)
        np.testing.assert_allclose(vertical, expected_vertical, atol=1.0e-12)

    def test_mesh_helpers_make_closed_elliptical_tube_and_ellipsoid(self):
        tube_points = self.generated["elliptical_tube_points"]
        tube_topology = self.generated["elliptical_tube_topology"]
        ellipsoid_points = self.generated["ellipsoid_points"]
        ellipsoid_topology = self.generated["ellipsoid_topology"]

        ring_segments = 8
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.1, 0.0, 0.0])
        lateral = np.array([0.0, 1.0, 0.0])
        vertical = np.array([0.0, 0.0, 1.0])
        half_width = 0.002
        half_height = 0.001
        points = np.asarray(
            tube_points(
                start,
                end,
                lateral,
                vertical,
                half_width,
                half_height,
                ring_segments,
            ),
            dtype=float,
        )
        self.assertEqual(points.shape, (2 * ring_segments, 3))
        for ring, center in ((points[:ring_segments], start), (points[ring_segments:], end)):
            offsets = ring - center
            ellipse = (
                (offsets @ lateral) / half_width
            ) ** 2 + ((offsets @ vertical) / half_height) ** 2
            np.testing.assert_allclose(ellipse, np.ones(ring_segments), atol=1.0e-12)
            np.testing.assert_allclose(offsets[:, 0], np.zeros(ring_segments))
        topology = tube_topology(ring_segments)
        _assert_valid_topology(self, topology, len(points))
        self.assertEqual(len(topology[0]), ring_segments + 2)

        center = np.array([0.03, -0.02, 0.01])
        initial_radius = 0.004
        horizontal_scale = 1.5
        vertical_scale = 0.6
        longitude_segments = 8
        latitude_segments = 4
        points = np.asarray(
            ellipsoid_points(
                center,
                initial_radius,
                horizontal_scale,
                vertical_scale,
                longitude_segments,
                latitude_segments,
            ),
            dtype=float,
        )
        offsets = points - center
        normalized_radius_squared = (
            offsets[:, 0] ** 2 + offsets[:, 1] ** 2
        ) / (initial_radius * horizontal_scale) ** 2 + (
            offsets[:, 2] / (initial_radius * vertical_scale)
        ) ** 2
        np.testing.assert_allclose(
            normalized_radius_squared,
            np.ones(len(points)),
            atol=1.0e-12,
        )
        self.assertAlmostEqual(
            float(np.max(np.abs(offsets[:, 0]))),
            initial_radius * horizontal_scale,
        )
        self.assertAlmostEqual(
            float(np.max(np.abs(offsets[:, 2]))),
            initial_radius * vertical_scale,
        )
        _assert_valid_topology(
            self,
            ellipsoid_topology(longitude_segments, latitude_segments),
            len(points),
        )

    def test_segment_birth_mesh_represents_exact_deposited_volume(self):
        sink_type = self.generated["MeshBeadSink"]
        stage = _Stage()
        sink = sink_type(stage, BeadEvolutionModel())
        start = TcpPose([0.0, 0.0, 0.0])
        end = TcpPose([0.1, 0.0, 0.0])
        deposited_volume_mm3 = 1000.0

        sink.emit_segment(start, end, deposited_volume_mm3, 1.0)

        self.assertEqual(len(stage.meshes), 1)
        self.assertEqual(len(sink._segments), 1)
        record = sink._segments[0]
        ring_segments = self.generated["MESH_RING_SEGMENTS"]
        polygon_area_factor = 0.5 * ring_segments * math.sin(
            2.0 * math.pi / ring_segments
        )
        represented_volume_m3 = (
            polygon_area_factor
            * record.initial_radius_m**2
            * float(np.linalg.norm(record.end_m - record.start_m))
        )
        self.assertAlmostEqual(
            represented_volume_m3,
            deposited_volume_mm3 * 1.0e-9,
            places=15,
        )
        self.assertAlmostEqual(record.half_width_m, record.initial_radius_m)
        self.assertAlmostEqual(record.half_height_m, record.initial_radius_m)
        expected_points = self.generated["elliptical_tube_points"](
            record.start_m,
            record.end_m,
            record.lateral_m,
            record.vertical_m,
            record.initial_radius_m,
            record.initial_radius_m,
            ring_segments,
        )
        np.testing.assert_allclose(_points_array(record.mesh), expected_points)
        _assert_valid_topology(
            self,
            (
                record.mesh.face_vertex_counts.value,
                record.mesh.face_vertex_indices.value,
            ),
            len(record.mesh.points.value),
        )

    def test_stationary_birth_mesh_represents_exact_deposited_volume(self):
        sink_type = self.generated["MeshBeadSink"]
        stage = _Stage()
        sink = sink_type(stage, BeadEvolutionModel())
        pose = TcpPose([0.03, -0.02, 0.01])
        deposited_volume_mm3 = 125.0

        sink.emit_point(pose, deposited_volume_mm3, 1.0)

        self.assertEqual(len(stage.meshes), 1)
        self.assertEqual(len(sink._droplets), 1)
        droplet = sink._droplets[0]
        represented_volume_m3 = self.generated["mesh_enclosed_volume_m3"](
            droplet.mesh.points.value,
            droplet.mesh.face_vertex_counts.value,
            droplet.mesh.face_vertex_indices.value,
        )
        self.assertAlmostEqual(
            represented_volume_m3,
            deposited_volume_mm3 * 1.0e-9,
            places=15,
        )
        self.assertAlmostEqual(
            droplet.horizontal_radius_m,
            droplet.initial_radius_m,
        )
        self.assertAlmostEqual(
            droplet.vertical_radius_m,
            droplet.initial_radius_m,
        )

    def test_marker_limit_accounts_skipped_volume_and_nonclearing_reset_retains_state(self):
        original_limit = self.generated["MAX_DEPOSITION_MARKERS"]
        self.generated["MAX_DEPOSITION_MARKERS"] = 2
        try:
            model = BeadEvolutionModel(
                spreading_ratio=1.2,
                spreading_time_s=1.0,
            )
            sink_type = self.generated["MeshBeadSink"]
            stage = _Stage()
            sink = sink_type(stage, model)
            first = TcpPose([0.0, 0.0, 0.0])
            second = TcpPose([0.01, 0.0, 0.0])

            sink.advance_to(2.5)
            sink.emit_segment(first, second, 10.0, 1.0)
            sink.emit_point(second, 20.0, 1.0)
            sink.emit_point(second, 30.0, 1.0)
            sink.advance_to(3.0)

            self.assertEqual(sink.marker_count, 2)
            self.assertEqual(len(stage.meshes), 2)
            self.assertEqual(len(sink._created_paths), 2)
            self.assertAlmostEqual(sink.skipped_volume_mm3, 30.0)

            retained_paths = list(sink._created_paths)
            retained_segments = list(sink._segments)
            retained_droplets = list(sink._droplets)
            retained_active_segments = list(sink._active_segments)
            retained_active_droplets = list(sink._active_droplets)
            retained_points = [
                _points_array(mesh).copy() for mesh in stage.meshes
            ]
            retained_updates = sink.geometry_update_count

            sink.reset(clear_geometry=False)

            self.assertEqual(stage.removed_paths, [])
            self.assertEqual(sink.marker_count, 2)
            self.assertAlmostEqual(sink.skipped_volume_mm3, 30.0)
            self.assertEqual(sink.geometry_update_count, retained_updates)
            self.assertEqual(sink._created_paths, retained_paths)
            self.assertEqual(sink._segments, retained_segments)
            self.assertEqual(sink._droplets, retained_droplets)
            self.assertEqual(sink._active_segments, retained_active_segments)
            self.assertEqual(sink._active_droplets, retained_active_droplets)
            self.assertEqual(sink._time_s, 3.0)
            for mesh, expected_points in zip(stage.meshes, retained_points):
                np.testing.assert_allclose(_points_array(mesh), expected_points)
        finally:
            self.generated["MAX_DEPOSITION_MARKERS"] = original_limit

    def test_evolution_is_per_bead_absolute_and_noncompounding(self):
        model = BeadEvolutionModel(
            spreading_ratio=1.6,
            spreading_time_s=2.0,
            shrinkage_fraction=0.2,
            shrinkage_time_s=4.0,
        )
        sink_type = self.generated["MeshBeadSink"]
        stage = _Stage()
        sink = sink_type(stage, model)
        first = TcpPose([0.0, 0.0, 0.0])
        second = TcpPose([0.01, 0.0, 0.0])
        third = TcpPose([0.02, 0.0, 0.0])

        sink.advance_to(1.0)
        sink.emit_segment(first, second, 10.0, 1.0)
        sink.advance_to(2.0)
        sink.emit_segment(second, third, 10.0, 1.0)
        sink.emit_point(third, 10.0, 1.0)

        initial_segment_radii = [record.initial_radius_m for record in sink._segments]
        initial_droplet_radius = sink._droplets[0].initial_radius_m
        sink.advance_to(3.0)

        for record, age_s in zip(sink._segments, (2.0, 1.0)):
            state = model.state_at(age_s)
            self.assertAlmostEqual(
                record.half_width_m,
                record.initial_radius_m * state.width_scale,
            )
            self.assertAlmostEqual(
                record.half_height_m,
                record.initial_radius_m * state.height_scale,
            )
            expected = self.generated["elliptical_tube_points"](
                record.start_m,
                record.end_m,
                record.lateral_m,
                record.vertical_m,
                record.half_width_m,
                record.half_height_m,
                self.generated["MESH_RING_SEGMENTS"],
            )
            np.testing.assert_allclose(_points_array(record.mesh), expected)
            represented_volume_m3 = self.generated["mesh_enclosed_volume_m3"](
                record.mesh.points.value,
                record.mesh.face_vertex_counts.value,
                record.mesh.face_vertex_indices.value,
            )
            self.assertAlmostEqual(
                represented_volume_m3,
                10.0e-9 * state.volume_scale,
                places=15,
            )

        droplet = sink._droplets[0]
        droplet_state = model.state_at(1.0)
        expected_horizontal_radius = (
            droplet.initial_radius_m * droplet_state.spreading_scale
        )
        expected_vertical_radius = (
            droplet.initial_radius_m
            * droplet_state.volume_scale
            / droplet_state.spreading_scale**2
        )
        self.assertAlmostEqual(
            droplet.horizontal_radius_m,
            expected_horizontal_radius,
        )
        self.assertAlmostEqual(droplet.vertical_radius_m, expected_vertical_radius)
        expected = self.generated["ellipsoid_points"](
            droplet.center_m,
            droplet.initial_radius_m,
            droplet_state.spreading_scale,
            droplet_state.volume_scale / droplet_state.spreading_scale**2,
            self.generated["MESH_DROPLET_LONGITUDE_SEGMENTS"],
            self.generated["MESH_DROPLET_LATITUDE_SEGMENTS"],
        )
        np.testing.assert_allclose(_points_array(droplet.mesh), expected)
        represented_droplet_volume_m3 = self.generated["mesh_enclosed_volume_m3"](
            droplet.mesh.points.value,
            droplet.mesh.face_vertex_counts.value,
            droplet.mesh.face_vertex_indices.value,
        )
        self.assertAlmostEqual(
            represented_droplet_volume_m3,
            10.0e-9 * droplet_state.volume_scale,
            places=15,
        )

        # Repeating the same absolute time is idempotent. A later update must
        # always derive from immutable birth radii, not previously scaled mesh.
        points_at_three = [
            _points_array(record.mesh).copy() for record in sink._segments
        ] + [_points_array(droplet.mesh).copy()]
        updates_at_three = sink.geometry_update_count
        sink.advance_to(3.0)
        for actual, expected_points in zip(
            [record.mesh for record in sink._segments] + [droplet.mesh],
            points_at_three,
        ):
            np.testing.assert_allclose(_points_array(actual), expected_points)
        self.assertEqual(sink.geometry_update_count, updates_at_three)

        sink.advance_to(4.0)
        self.assertEqual(
            [record.initial_radius_m for record in sink._segments],
            initial_segment_radii,
        )
        self.assertEqual(droplet.initial_radius_m, initial_droplet_radius)
        for record, age_s in zip(sink._segments, (3.0, 2.0)):
            state = model.state_at(age_s)
            self.assertAlmostEqual(
                record.half_width_m,
                record.initial_radius_m * state.width_scale,
            )
            self.assertAlmostEqual(
                record.half_height_m,
                record.initial_radius_m * state.height_scale,
            )

        created_paths = [mesh.path for mesh in stage.meshes]
        sink.reset(clear_geometry=True)
        self.assertEqual(stage.removed_paths, created_paths)
        self.assertEqual(sink.marker_count, 0)
        self.assertEqual(sink.geometry_update_count, 0)
        self.assertEqual(sink._segments, [])
        self.assertEqual(sink._droplets, [])
        self.assertEqual(sink._active_segments, [])
        self.assertEqual(sink._active_droplets, [])
        self.assertEqual(sink._time_s, 0.0)


if __name__ == "__main__":
    unittest.main()
