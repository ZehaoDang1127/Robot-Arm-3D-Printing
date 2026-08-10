from __future__ import annotations

import math
import unittest
from pathlib import Path

from robotic_printing_platform.config import load_planner_config
from robotic_printing_platform.extrusion import (
    MaterialProfile,
    load_material_profile,
    material_profile_from_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        material_config = load_planner_config(
            PROJECT_ROOT / "planner_config.json"
        ).material
        profile = material_config.profile

        self.assertEqual(material_config.profile_id, profile.profile_id)
        self.assertEqual(Path(material_config.profile_path).stem, profile.profile_id)
        self.assertEqual(profile.name, "alginate_chitosan_pic_al1ch1_research")
        self.assertEqual(profile.extrusion_mode, "volumetric")
        self.assertAlmostEqual(profile.density_g_cm3, 1.05)
        self.assertAlmostEqual(profile.physx_particle_contact_offset_m, 0.0002)
        self.assertAlmostEqual(profile.physx_adhesion, 15.0)
        self.assertAlmostEqual(profile.spreading_ratio, 1.35)
        self.assertAlmostEqual(profile.spreading_time_s, 2.0)
        self.assertAlmostEqual(profile.shrinkage_fraction, 0.08)
        self.assertAlmostEqual(profile.shrinkage_time_s, 30.0)

    def test_legacy_profile_data_gets_neutral_evolution_defaults(self):
        profile = material_profile_from_dict(
            {
                "profile_id": "legacy",
                "name": "legacy material",
                "extrusion_mode": "volumetric",
            }
        )

        self.assertEqual(profile.spreading_ratio, 1.0)
        self.assertEqual(profile.spreading_time_s, 1.0)
        self.assertEqual(profile.shrinkage_fraction, 0.0)
        self.assertEqual(profile.shrinkage_time_s, 1.0)

    def test_rejects_invalid_evolution_parameters(self):
        invalid_profiles = (
            ({"spreading_ratio": 0.9}, "spreading_ratio"),
            ({"spreading_time_s": 0.0}, "spreading_time_s"),
            ({"shrinkage_fraction": -0.1}, "shrinkage_fraction"),
            ({"shrinkage_fraction": 1.0}, "shrinkage_fraction"),
            ({"shrinkage_time_s": 0.0}, "shrinkage_time_s"),
        )
        for keyword_arguments, field_name in invalid_profiles:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    MaterialProfile(
                        profile_id="invalid_evolution",
                        name="invalid evolution",
                        extrusion_mode="volumetric",
                        **keyword_arguments,
                    )

    def test_cli_style_override_selects_pla_without_changing_planner_file(self):
        profile = load_planner_config(
            PROJECT_ROOT / "planner_config.json",
            material_profile_id="pla",
        ).material.profile

        self.assertEqual(profile.profile_id, "pla")
        self.assertEqual(profile.name, "PLA")
        self.assertEqual(profile.extrusion_mode, "filament_length")
        self.assertAlmostEqual(profile.filament_diameter_mm, 1.75)

    def test_unknown_profile_lists_available_profiles(self):
        with self.assertRaisesRegex(ValueError, "available profiles"):
            load_planner_config(
                PROJECT_ROOT / "planner_config.json",
                material_profile_id="not_a_material",
            )

    def test_profile_id_must_match_filename(self):
        profile_path = PROJECT_ROOT / "material_profiles" / "pla.json"

        profile = load_material_profile(profile_path)

        self.assertEqual(profile.profile_id, profile_path.stem)


if __name__ == "__main__":
    unittest.main()
