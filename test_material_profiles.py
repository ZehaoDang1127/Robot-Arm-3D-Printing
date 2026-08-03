from __future__ import annotations

import math
import unittest
from pathlib import Path

from robotic_printing_platform.config import load_planner_config
from robotic_printing_platform.extrusion import MaterialProfile


class MaterialProfileTests(unittest.TestCase):
    def test_default_filament_mode_uses_filament_cross_section(self):
        material = MaterialProfile(filament_diameter_mm=2.0)

        self.assertAlmostEqual(material.volume_mm3(3.0), 3.0 * math.pi)

    def test_volumetric_mode_treats_each_e_unit_as_one_cubic_millimetre(self):
        material = MaterialProfile(
            name="research hydrogel",
            extrusion_mode="volumetric",
            flow_multiplier=0.8,
        )

        self.assertAlmostEqual(material.volume_mm3(3.25), 2.6)
        self.assertEqual(material.volume_mm3(-1.0), 0.0)

    def test_syringe_mode_uses_plunger_cross_section(self):
        material = MaterialProfile(
            name="syringe paste",
            extrusion_mode="syringe_plunger",
            syringe_inner_diameter_mm=4.0,
        )

        self.assertAlmostEqual(material.volume_mm3(2.0), 8.0 * math.pi)

    def test_syringe_mode_requires_a_bore_diameter(self):
        with self.assertRaisesRegex(ValueError, "syringe_inner_diameter_mm"):
            MaterialProfile(extrusion_mode="syringe_plunger")

    def test_loads_research_hydrogel_preset(self):
        config_path = Path(__file__).resolve().parent / "planner_config_hydrogel.json"
        profile = load_planner_config(config_path).material.profile

        self.assertEqual(profile.name, "alginate_chitosan_pic_al1ch1_research")
        self.assertEqual(profile.extrusion_mode, "volumetric")
        self.assertAlmostEqual(profile.density_g_cm3, 1.05)
        self.assertAlmostEqual(profile.physx_particle_contact_offset_m, 0.0002)
        self.assertAlmostEqual(profile.physx_adhesion, 15.0)


if __name__ == "__main__":
    unittest.main()
