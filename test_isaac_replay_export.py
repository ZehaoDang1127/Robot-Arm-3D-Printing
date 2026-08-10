from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robotic_printing_platform.config import load_planner_config
from robotic_printing_platform.exporters.isaac import export_isaac_bundle
from robotic_printing_platform.extrusion import MaterialProfile
from robotic_printing_platform.robots.franka_panda import IKConfig, IKReport, RobotTrajectory, TrajectoryPoint


TEST_MATERIAL = MaterialProfile(
    profile_id="test_material",
    name="test material",
    extrusion_mode="volumetric",
    density_g_cm3=1.0,
    spreading_ratio=1.2,
    spreading_time_s=2.5,
    shrinkage_fraction=0.05,
    shrinkage_time_s=20.0,
)


class IsaacReplayExportTests(unittest.TestCase):
    def test_ur5e_extruder_bundle_includes_mount_payload(self):
        repo_root = Path(__file__).resolve().parent
        assembly = repo_root / "UR5e_extruder.usd"
        mount_payload = repo_root / "Mount_Extruder_Models" / "ur5_mount_extruder.usd"

        self.assertTrue(assembly.is_file(), "UR5e assembly USD is missing")
        self.assertTrue(
            mount_payload.is_file(),
            "UR5e_extruder.usd requires Mount_Extruder_Models/ur5_mount_extruder.usd",
        )
        with mount_payload.open("rb") as payload_file:
            self.assertEqual(payload_file.read(8), b"PXR-USDC")

    def test_generates_time_interpolated_position_target_replay(self):
        point = TrajectoryPoint(
            index=0,
            q=np.zeros(7),
            p=np.zeros(3),
            yaw=0.0,
            is_print=True,
            layer=0,
            seg_id=0,
            feed_m_s=0.01,
            de=0.0,
            material="PLA",
            extrusion_volume_mm3=0.0,
            extrusion_mass_g=0.0,
            pos_error_m=0.0,
            rot_error_rad=0.0,
            time_from_start_s=0.0,
        )
        trajectory = RobotTrajectory(
            points=[point],
            report=IKReport(True, 1, 1, 1, [], [], 0.0, 0.0),
            config=IKConfig(),
        )

        with tempfile.TemporaryDirectory() as directory:
            legacy_manager_path = Path(directory) / "deposition_manager.py"
            legacy_manager_path.write_text("legacy generated copy\n", encoding="utf-8")
            bundle = export_isaac_bundle(
                trajectory,
                Path(directory),
                material_profile=TEST_MATERIAL,
            )
            source = bundle["isaac_script"].read_text(encoding="utf-8")
            compile(source, str(bundle["isaac_script"]), "exec")
            self.assertNotIn("deposition_manager", bundle)
            self.assertFalse(legacy_manager_path.exists())

        self.assertIn("time_from_start_s", source)
        self.assertIn("interpolate_joint_target", source)
        self.assertIn("controller.apply_action(ArticulationAction", source)
        self.assertIn("joint_indices=joint_indices", source)
        self.assertIn("robot.get_dof_index(name)", source)
        self.assertIn("SETTLING_TIME_S", source)
        self.assertIn('os.environ.get("RPP_POST_DEPOSITION_TIME_S", "2.0")', source)
        self.assertIn("TRACKING_PLOT_SAMPLE_STRIDE", source)
        self.assertIn("joint_tracking.svg", source)
        self.assertIn("maximum_tracking_error_rad", source)
        self.assertIn("enable_robot_physics_variants", source)
        self.assertIn('variant_set.SetVariantSelection("PhysX")', source)
        self.assertIn("repair_extruder_mount", source)
        self.assertIn("UsdPhysics.CollisionAPI.Apply(mesh_prim)", source)
        self.assertIn("UsdGeom.Tokens.faceVarying", source)
        self.assertIn("MOUNT_TARGET_MAX_DIMENSION_M = 0.148", source)
        self.assertIn("DEFAULT_MOUNT_MASS_KG = 1.0", source)
        self.assertIn('os.environ.get("RPP_MOUNT_SCALE")', source)
        self.assertIn('os.environ.get("RPP_ENABLE_MOUNT_COLLISION", "0")', source)
        self.assertIn("UsdGeom.BBoxCache", source)
        self.assertIn("corrected mount scale", source)
        self.assertIn("corrected mount fixed-joint anchor", source)
        self.assertIn("mount_scale_correction", source)
        self.assertIn("find_or_create_articulation_root", source)
        self.assertIn("Usd.TraverseInstanceProxies()", source)
        self.assertIn("UsdPhysics.ArticulationRootAPI", source)
        self.assertIn("UsdPhysics.ArticulationRootAPI.Apply(reference_prim)", source)
        self.assertIn("prim.IsA(UsdPhysics.Joint)", source)
        self.assertIn("prim.IsA(UsdPhysics.RevoluteJoint)", source)
        self.assertIn("available articulation DOFs", source)
        self.assertIn("robot = SingleArticulation(prim_path=ARTICULATION_PRIM", source)
        self.assertIn("initialize_at_first_target", source)
        self.assertEqual(source.count("robot.set_joint_positions("), 1)
        self.assertIn("INITIALIZATION_TIMEOUT_S = 30.0", source)
        self.assertIn("INITIALIZATION_TOLERANCE_RAD", source)
        self.assertIn("def resolve_project_root(script_dir):", source)
        self.assertIn('os.environ.get("RPP_PROJECT_ROOT", "")', source)
        self.assertIn("while project_root_text in sys.path:", source)
        self.assertIn("sys.path.insert(0, project_root_text)", source)
        self.assertIn(
            "from robotic_printing_platform.extrusion import deposition as deposition_module",
            source,
        )
        self.assertIn(
            "actual_deposition_path = Path(deposition_module.__file__).resolve()",
            source,
        )
        self.assertIn(
            "actual_deposition_path != expected_deposition_path",
            source,
        )
        self.assertNotIn("from deposition_manager import (", source)
        project_resolution = source.index(
            "PROJECT_ROOT = resolve_project_root(SCRIPT_DIR)"
        )
        central_import = source.index(
            "from robotic_printing_platform.extrusion import deposition as deposition_module"
        )
        isaac_start = source.index("simulation_app = SimulationApp(")
        self.assertLess(project_resolution, central_import)
        self.assertLess(central_import, isaac_start)
        self.assertIn(
            "BeadEvolutionModel = deposition_module.BeadEvolutionModel",
            source,
        )
        self.assertIn(
            "DepositionManager = deposition_module.DepositionManager",
            source,
        )
        self.assertIn(
            "flow_schedule = FlowSchedule.from_trajectory_points(trajectory)",
            source,
        )
        self.assertIn("TCP_ANCHOR_CANDIDATES =", source)
        self.assertIn(
            'TCP_PRIM_OVERRIDE = os.environ.get("RPP_TCP_PRIM", "").strip()',
            source,
        )
        self.assertIn("class RobotTcpPoseReader", source)
        self.assertIn("def resolve_tcp_anchor", source)
        self.assertIn("ComputeLocalToWorldTransform", source)
        self.assertIn("RPP_TCP_PRIM must be below", source)
        self.assertIn("must select one of the configured TCP anchor", source)
        self.assertIn("Usd.TraverseInstanceProxies()", source)
        self.assertIn("deposition_manager.reset(", source)
        self.assertIn("deposition_manager.update_from_schedule(", source)
        manager_reset = source.index("deposition_manager.reset(")
        replay_loop = source.index("while simulation_app.is_running():")
        manager_update = source.index(
            "deposition_manager.update_from_schedule(", replay_loop
        )
        physics_step = source.rfind(
            "world.step(render=True)", replay_loop, manager_update
        )
        visual_evolution = source.index(
            "visual_bead_sink.advance_to(replay_time_s)", replay_loop
        )
        self.assertLess(manager_reset, replay_loop)
        self.assertGreater(physics_step, replay_loop)
        self.assertLess(physics_step, visual_evolution)
        self.assertLess(visual_evolution, manager_update)
        self.assertLess(physics_step, manager_update)
        self.assertIn("spawn_deposition_segment", source)
        self.assertIn("UsdGeom.BasisCurves.Define", source)
        self.assertIn("UsdGeom.Sphere.Define", source)
        self.assertIn("visual_bead_radius_m", source)
        self.assertIn("VISUAL_SEGMENTS_PER_CHUNK = 256", source)
        self.assertIn("bead_chunk_", source)
        self.assertIn("UsdGeom.Tokens.uniform", source)
        self.assertIn("class _VisualCurveChunk", source)
        self.assertIn("chunk.initial_widths.append(width)", source)
        self.assertIn("chunk.birth_times_s.append(self._time_s)", source)
        self.assertIn("for chunk in self._curve_chunks", source)
        self.assertIn("chunk.curve.GetWidthsAttr().Set(chunk.widths)", source)
        self.assertIn("class _VisualDroplet", source)
        self.assertIn("droplet.sphere.GetRadiusAttr().Set(radius_m)", source)
        self.assertIn("self._final_visual_radius_scale", source)
        self.assertIn("self._active_droplets = []", source)
        self.assertNotIn("MAX_VISUAL_BEAD_RADIUS_M", source)
        self.assertIn("start_pose.position_m", source)
        self.assertIn("end_pose.position_m", source)
        self.assertNotIn('deposition_row["p"]', source)
        self.assertNotIn('trajectory[deposition_index - 1]["p"]', source)
        self.assertIn('DEPOSITION_MODE = os.environ.get("RPP_DEPOSITION_MODE", "particles")', source)
        self.assertIn('"spreading_ratio",', source)
        self.assertIn('"spreading_time_s",', source)
        self.assertIn('"shrinkage_fraction",', source)
        self.assertIn('"shrinkage_time_s",', source)
        self.assertIn('MATERIAL_PROFILE.get("spreading_ratio", 1.0)', source)
        self.assertIn('MATERIAL_PROFILE.get("shrinkage_fraction", 0.0)', source)
        self.assertIn("bead_evolution_model = BeadEvolutionModel(", source)
        self.assertIn("class PhysxExtrusionEmitter", source)
        self.assertIn("particleUtils.add_physx_particle_system", source)
        self.assertIn("particleUtils.add_physx_particleset_points", source)
        self.assertIn("particleUtils.add_pbd_particle_material", source)
        self.assertIn("particleUtils.add_physx_particle_isosurface", source)
        self.assertIn('"RPP_PARTICLE_ISOSURFACE", "0"', source)
        self.assertIn("safe default; PBD physics remains enabled", source)
        self.assertIn("PhysxSchema.PhysxSceneAPI.Apply", source)
        self.assertIn("CreateEnableGPUDynamicsAttr().Set(True)", source)
        self.assertIn('CreateBroadphaseTypeAttr().Set("GPU")', source)
        self.assertIn("def create_print_bed", source)
        self.assertIn("UsdPhysics.CollisionAPI.Apply(bed.GetPrim())", source)
        self.assertIn("PARTICLE_VOLUME_MM3", source)
        self.assertIn("volume_remainder_mm3", source)
        self.assertIn('"deposited_particles"', source)
        self.assertLess(
            source.index("extend_array_attribute(simulation_points, positions)"),
            source.index("extend_array_attribute(self.points.GetPointsAttr(), positions)"),
        )
        self.assertNotIn("spawn_deposition_marker", source)
        self.assertIn("MAX_ACCEPTABLE_TRACKING_ERROR_RAD", source)
        self.assertIn('"tracking_passed"', source)
        self.assertIn('"initialization_duration_s"', source)
        self.assertIn('"tcp_pose_samples"', source)
        self.assertIn('"deposition_interval_updates"', source)
        self.assertIn('"deposition_discontinuities"', source)
        self.assertIn('"commanded_deposition_volume_mm3"', source)
        self.assertIn('"emitted_deposition_volume_mm3"', source)
        self.assertIn('"represented_deposition_volume_mm3"', source)
        self.assertIn('"visual_geometry_updates"', source)
        self.assertIn('"unrepresented_volume_mm3"', source)
        self.assertIn('os.environ.get("RPP_TRAJECTORY_CSV"', source)
        self.assertNotIn(str(bundle["csv"].resolve()), source)

        resolver_node = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "resolve_project_root"
        )
        resolver_namespace = {"os": os, "Path": Path}
        exec(
            compile(
                ast.Module(body=[resolver_node], type_ignores=[]),
                "<generated-project-root-resolver>",
                "exec",
            ),
            resolver_namespace,
        )
        resolve_project_root = resolver_namespace["resolve_project_root"]
        with tempfile.TemporaryDirectory() as directory:
            repository_root = Path(directory) / "cloned-repository"
            package_root = repository_root / "robotic_printing_platform"
            deposition_module = package_root / "extrusion" / "deposition.py"
            deposition_module.parent.mkdir(parents=True)
            (package_root / "__init__.py").write_text("", encoding="utf-8")
            deposition_module.write_text("", encoding="utf-8")
            nested_output = repository_root / "outputs" / "ur5e"
            nested_output.mkdir(parents=True)

            with mock.patch.dict(os.environ, {"RPP_PROJECT_ROOT": ""}):
                self.assertEqual(
                    resolve_project_root(nested_output),
                    repository_root,
                )
                with self.assertRaisesRegex(RuntimeError, "Set RPP_PROJECT_ROOT"):
                    resolve_project_root(Path(directory) / "external-output")
            with mock.patch.dict(
                os.environ,
                {"RPP_PROJECT_ROOT": str(repository_root)},
            ):
                self.assertEqual(
                    resolve_project_root(Path(directory) / "external-output"),
                    repository_root.resolve(),
                )
            with mock.patch.dict(
                os.environ,
                {"RPP_PROJECT_ROOT": str(Path(directory) / "not-a-repository")},
            ):
                with self.assertRaisesRegex(RuntimeError, "RPP_PROJECT_ROOT"):
                    resolve_project_root(nested_output)

        bootstrap_source = source.split("from isaacsim import SimulationApp", 1)[0]
        bootstrap_source += "\nprint(DepositionManager.__module__)\n"
        repository_root = Path(__file__).resolve().parent

        def run_isolated_bootstrap(script_path, environment):
            return subprocess.run(
                [sys.executable, "-I", "-B", str(script_path)],
                cwd=script_path.parent,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        with tempfile.TemporaryDirectory(dir=repository_root) as directory:
            script_path = Path(directory) / "outputs" / "ur5e" / "bootstrap.py"
            script_path.parent.mkdir(parents=True)
            script_path.write_text(bootstrap_source, encoding="utf-8")
            environment = os.environ.copy()
            environment.pop("RPP_PROJECT_ROOT", None)
            result = run_isolated_bootstrap(script_path, environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "robotic_printing_platform.extrusion.deposition",
                result.stdout,
            )

        with tempfile.TemporaryDirectory() as directory:
            script_path = Path(directory) / "bootstrap.py"
            script_path.write_text(bootstrap_source, encoding="utf-8")
            environment = os.environ.copy()
            environment["RPP_PROJECT_ROOT"] = str(repository_root)
            result = run_isolated_bootstrap(script_path, environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "robotic_printing_platform.extrusion.deposition",
                result.stdout,
            )

    def test_ur5e_replay_embeds_fixed_link_tcp_anchor_fallbacks(self):
        repo_root = Path(__file__).resolve().parent
        planner_config = load_planner_config(
            repo_root / "planner_config.json",
            robot_config_dir=(
                repo_root
                / "robotic_printing_platform"
                / "robots"
                / "robot_configs"
                / "ur5e"
            ),
        )
        config = planner_config.make_ik_config()
        point = TrajectoryPoint(
            index=0,
            q=np.zeros(len(config.joint_names)),
            p=np.zeros(3),
            yaw=0.0,
            is_print=False,
            layer=0,
            seg_id=0,
            feed_m_s=0.01,
            de=0.0,
            material="test material",
            extrusion_volume_mm3=0.0,
            extrusion_mass_g=0.0,
            pos_error_m=0.0,
            rot_error_rad=0.0,
            time_from_start_s=0.0,
        )
        trajectory = RobotTrajectory(
            points=[point],
            report=IKReport(True, 1, 1, 1, [], [], 0.0, 0.0),
            config=config,
        )

        with tempfile.TemporaryDirectory() as directory:
            bundle = export_isaac_bundle(
                trajectory,
                Path(directory),
                material_profile=TEST_MATERIAL,
            )
            source = bundle["isaac_script"].read_text(encoding="utf-8")

        candidates_line = next(
            line
            for line in source.splitlines()
            if line.startswith("TCP_ANCHOR_CANDIDATES = ")
        )
        candidates = ast.literal_eval(candidates_line.partition("=")[2].strip())

        self.assertEqual(
            [candidate["link_name"] for candidate in candidates],
            ["tool0", "flange", "wrist_3_link"],
        )
        for candidate in candidates:
            self.assertEqual(len(candidate["translation_m"]), 3)
            self.assertEqual(len(candidate["rotation_xyzw"]), 4)

    def test_writes_runtime_material_profile_and_bed_configuration(self):
        point = TrajectoryPoint(
            index=0,
            q=np.zeros(7),
            p=np.array([0.31, -0.04, 0.23]),
            yaw=0.0,
            is_print=True,
            layer=0,
            seg_id=0,
            feed_m_s=0.01,
            de=0.0,
            material="test paste",
            extrusion_volume_mm3=0.0,
            extrusion_mass_g=0.0,
            pos_error_m=0.0,
            rot_error_rad=0.0,
            time_from_start_s=0.0,
        )
        trajectory = RobotTrajectory(
            points=[point],
            report=IKReport(True, 1, 1, 1, [], [], 0.0, 0.0),
            config=IKConfig(),
        )
        material = MaterialProfile(
            profile_id="test_paste",
            name="test paste",
            extrusion_mode="volumetric",
            density_g_cm3=1.6,
            physx_particle_contact_offset_m=0.0007,
            physx_viscosity=432.0,
            physx_cohesion=6.0,
            physx_adhesion=7.0,
            physx_surface_tension=0.03,
            physx_friction=8.0,
            physx_damping=0.8,
            spreading_ratio=1.4,
            spreading_time_s=3.0,
            shrinkage_fraction=0.12,
            shrinkage_time_s=40.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            bundle = export_isaac_bundle(
                trajectory,
                Path(directory),
                material_profile=material,
                bed_center_xyz_m=(0.31, -0.04, 0.23),
            )
            source = bundle["isaac_script"].read_text(encoding="utf-8")
            resolved_material = json.loads(
                bundle["material_profile"].read_text(encoding="utf-8")
            )
            alternate_bundle = export_isaac_bundle(
                trajectory,
                Path(directory) / "alternate_material",
                material_profile=MaterialProfile(
                    profile_id="alternate_paste",
                    name="alternate paste",
                    extrusion_mode="volumetric",
                    density_g_cm3=1.2,
                    physx_particle_contact_offset_m=0.0005,
                    physx_viscosity=12.0,
                    physx_cohesion=2.0,
                    physx_adhesion=3.0,
                    physx_surface_tension=0.01,
                    physx_friction=4.0,
                    physx_damping=0.5,
                    spreading_ratio=1.1,
                    spreading_time_s=4.0,
                    shrinkage_fraction=0.02,
                    shrinkage_time_s=50.0,
                ),
                bed_center_xyz_m=(0.31, -0.04, 0.23),
            )
            alternate_source = alternate_bundle["isaac_script"].read_text(
                encoding="utf-8"
            )
            alternate_resolved_material = json.loads(
                alternate_bundle["material_profile"].read_text(encoding="utf-8")
            )

        self.assertIn("BED_CENTER_M = (0.31, -0.04, 0.23)", source)
        self.assertIn(
            'MATERIAL_PROFILE_PATH = SCRIPT_DIR / "resolved_material_profile.json"',
            source,
        )
        self.assertIn("MATERIAL_PROFILE = load_material_profile", source)
        self.assertNotIn("test_paste", source)
        self.assertEqual(source, alternate_source)
        self.assertEqual(resolved_material["profile_id"], "test_paste")
        self.assertEqual(resolved_material["physx_viscosity"], 432.0)
        self.assertEqual(resolved_material["spreading_ratio"], 1.4)
        self.assertEqual(resolved_material["spreading_time_s"], 3.0)
        self.assertEqual(resolved_material["shrinkage_fraction"], 0.12)
        self.assertEqual(resolved_material["shrinkage_time_s"], 40.0)
        self.assertEqual(alternate_resolved_material["spreading_ratio"], 1.1)
        self.assertEqual(alternate_resolved_material["shrinkage_fraction"], 0.02)

    def test_custom_robot_usd_overrides_configured_asset(self):
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
            material="PLA",
            extrusion_volume_mm3=0.0,
            extrusion_mass_g=0.0,
            pos_error_m=0.0,
            rot_error_rad=0.0,
            time_from_start_s=0.0,
        )
        trajectory = RobotTrajectory(
            points=[point],
            report=IKReport(True, 1, 1, 1, [], [], 0.0, 0.0),
            config=IKConfig(),
        )

        with tempfile.TemporaryDirectory() as directory:
            custom_usd = Path(directory) / "UR5e_extruder.usd"
            custom_usd.touch()
            bundle = export_isaac_bundle(
                trajectory,
                Path(directory),
                robot_usd_path=custom_usd,
                material_profile=TEST_MATERIAL,
            )
            source = bundle["isaac_script"].read_text(encoding="utf-8")

        self.assertIn('os.environ.get("RPP_ROBOT_USD", ROBOT_USD_DEFAULT)', source)
        self.assertIn("ROBOT_USD_RELATIVE = 'UR5e_extruder.usd'", source)
        self.assertNotIn(str(custom_usd.resolve()), source)

        with tempfile.TemporaryDirectory() as directory:
            custom_usd = Path(directory) / "UR5e_extruder.usd"
            custom_usd.touch()
            with mock.patch(
                "robotic_printing_platform.exporters.isaac.os.path.relpath",
                side_effect=ValueError("different drives"),
            ):
                bundle = export_isaac_bundle(
                    trajectory,
                    Path(directory),
                    robot_usd_path=custom_usd,
                    material_profile=TEST_MATERIAL,
                )
            fallback_source = bundle["isaac_script"].read_text(encoding="utf-8")

        self.assertIn("ROBOT_USD_RELATIVE = ''", fallback_source)
        self.assertIn(repr(str(custom_usd.resolve()))[1:-1], fallback_source)


if __name__ == "__main__":
    unittest.main()
