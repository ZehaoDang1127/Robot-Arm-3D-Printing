from __future__ import annotations

import math
import unittest
from pathlib import Path

from robotic_printing_platform.config import load_planner_config
from robotic_printing_platform.extrusion import MaterialProfile, load_material_profile


class MaterialProfileTests(unittest.TestCase):
    def test_filament_mode_uses_filament_cross_section(self):
        material = MaterialProfile(
            profile_id="test_filament",
            name="test filament",
            extrusion_mode="filament_length",
            filament_diameter_mm=2.0,
        )

        self.assertAlmostEqual(material.volume_mm3(3.0), 3.0 * math.pi)

    def test_volumetric_mode_treats_each_e_unit_as_one_cubic_millimetre(self):
        material = MaterialProfile(
            profile_id="test_hydrogel",
            name="research hydrogel",
            extrusion_mode="volumetric",
            flow_multiplier=0.8,
        )

        self.assertAlmostEqual(material.volume_mm3(3.25), 2.6)
        self.assertEqual(material.volume_mm3(-1.0), 0.0)

    def test_syringe_mode_uses_plunger_cross_section(self):
        material = MaterialProfile(
            profile_id="test_syringe",
            name="syringe paste",
            extrusion_mode="syringe_plunger",
            syringe_inner_diameter_mm=4.0,
        )

        self.assertAlmostEqual(material.volume_mm3(2.0), 8.0 * math.pi)

    def test_syringe_mode_requires_a_bore_diameter(self):
        with self.assertRaisesRegex(ValueError, "syringe_inner_diameter_mm"):
            MaterialProfile(
                profile_id="invalid_syringe",
                name="invalid syringe",
                extrusion_mode="syringe_plunger",
            )

    def test_default_planner_selects_research_hydrogel_profile(self):
        repo_root = Path(__file__).resolve().parent
        material_config = load_planner_config(repo_root / "planner_config.json").material
        profile = material_config.profile

        self.assertEqual(material_config.profile_id, profile.profile_id)
        self.assertEqual(Path(material_config.profile_path).stem, profile.profile_id)
        self.assertEqual(profile.name, "alginate_chitosan_pic_al1ch1_research")
        self.assertEqual(profile.extrusion_mode, "volumetric")
        self.assertAlmostEqual(profile.density_g_cm3, 1.05)
        self.assertAlmostEqual(profile.physx_particle_contact_offset_m, 0.0002)
        self.assertAlmostEqual(profile.physx_adhesion, 15.0)

    def test_cli_style_override_selects_pla_without_changing_planner_file(self):
        repo_root = Path(__file__).resolve().parent
        profile = load_planner_config(
            repo_root / "planner_config.json",
            material_profile_id="pla",
        ).material.profile

        self.assertEqual(profile.profile_id, "pla")
        self.assertEqual(profile.name, "PLA")
        self.assertEqual(profile.extrusion_mode, "filament_length")
        self.assertAlmostEqual(profile.filament_diameter_mm, 1.75)

    def test_unknown_profile_lists_available_profiles(self):
        repo_root = Path(__file__).resolve().parent
        with self.assertRaisesRegex(ValueError, "available profiles"):
            load_planner_config(
                repo_root / "planner_config.json",
                material_profile_id="not_a_material",
            )

    def test_profile_id_must_match_filename(self):
        repo_root = Path(__file__).resolve().parent
        profile_path = repo_root / "material_profiles" / "pla.json"

        profile = load_material_profile(profile_path)

        self.assertEqual(profile.profile_id, profile_path.stem)


if __name__ == "__main__":
    unittest.main()
