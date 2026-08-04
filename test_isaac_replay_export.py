from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from robotic_printing_platform.exporters.isaac import export_isaac_bundle
from robotic_printing_platform.extrusion import MaterialProfile
from robotic_printing_platform.robots.franka_panda import IKConfig, IKReport, RobotTrajectory, TrajectoryPoint


TEST_MATERIAL = MaterialProfile(
    profile_id="test_material",
    name="test material",
    extrusion_mode="volumetric",
    density_g_cm3=1.0,
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
            bundle = export_isaac_bundle(
                trajectory,
                Path(directory),
                material_profile=TEST_MATERIAL,
            )
            source = bundle["isaac_script"].read_text(encoding="utf-8")
            compile(source, str(bundle["isaac_script"]), "exec")

        self.assertIn("time_from_start_s", source)
        self.assertIn("interpolate_joint_target", source)
        self.assertIn("controller.apply_action(ArticulationAction", source)
        self.assertIn("joint_indices=joint_indices", source)
        self.assertIn("robot.get_dof_index(name)", source)
        self.assertIn("SETTLING_TIME_S", source)
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
        self.assertIn("DEPOSITION_MAX_JOINT_ERROR_RAD", source)
        self.assertIn("spawn_deposition_segment", source)
        self.assertIn("UsdGeom.BasisCurves.Define", source)
        self.assertIn('DEPOSITION_MODE = os.environ.get("RPP_DEPOSITION_MODE", "particles")', source)
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
        self.assertIn("segment_start", source)
        self.assertNotIn("spawn_deposition_marker", source)
        self.assertIn("MAX_ACCEPTABLE_TRACKING_ERROR_RAD", source)
        self.assertIn('"tracking_passed"', source)
        self.assertIn('"initialization_duration_s"', source)
        self.assertIn('"skipped_deposition_points_due_to_tracking"', source)
        self.assertIn('os.environ.get("RPP_TRAJECTORY_CSV"', source)
        self.assertNotIn(str(bundle["csv"].resolve()), source)

    def test_embeds_physical_material_and_bed_configuration(self):
        point = TrajectoryPoint(
            index=0,
            q=np.zeros(7),
            p=np.array([0.31, -0.04, 0.23]),
            yaw=0.0,
            is_print=True,
            layer=0,
            seg_id=0,
            feed_m_s=0.01,
            de=0.2,
            material="test paste",
            extrusion_volume_mm3=0.48,
            extrusion_mass_g=0.0006,
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

        self.assertIn("BED_CENTER_M = (0.31, -0.04, 0.23)", source)
        self.assertIn("MATERIAL_PROFILE_ID = 'test_paste'", source)
        self.assertIn("MATERIAL_NAME = 'test paste'", source)
        self.assertIn("MATERIAL_EXTRUSION_MODE = 'volumetric'", source)
        self.assertIn("MATERIAL_DENSITY_KG_M3 = 1600.0", source)
        self.assertIn("PARTICLE_CONTACT_OFFSET_M = 0.0007", source)
        self.assertIn("PARTICLE_VISCOSITY = 432.0", source)
        self.assertIn("PARTICLE_COHESION = 6.0", source)
        self.assertIn("PARTICLE_ADHESION = 7.0", source)
        self.assertEqual(resolved_material["profile_id"], "test_paste")
        self.assertEqual(resolved_material["physx_viscosity"], 432.0)

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
