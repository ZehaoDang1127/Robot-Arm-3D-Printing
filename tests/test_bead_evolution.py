from __future__ import annotations

import math
import unittest

from robotic_printing_platform.extrusion import (
    BeadEvolutionModel,
    MaterialProfile,
)


class BeadEvolutionModelTests(unittest.TestCase):
    def test_neutral_defaults_preserve_geometry_at_every_age(self):
        model = BeadEvolutionModel()

        state = model.state_at(123.0)

        self.assertEqual(state.spreading_scale, 1.0)
        self.assertEqual(state.volume_scale, 1.0)
        self.assertEqual(state.width_scale, 1.0)
        self.assertEqual(state.height_scale, 1.0)
        self.assertEqual(state.visual_radius_scale, 1.0)

    def test_exponential_state_uses_seconds_and_initial_deposit_as_baseline(self):
        model = BeadEvolutionModel(
            spreading_ratio=1.5,
            spreading_time_s=2.0,
            shrinkage_fraction=0.2,
            shrinkage_time_s=4.0,
        )

        initial = model.state_at(0.0)
        state = model.state_at(2.0)
        same_state = model.state_at(2.0)

        expected_spreading = 1.0 + 0.5 * (1.0 - math.exp(-1.0))
        expected_volume = 1.0 - 0.2 * (1.0 - math.exp(-0.5))
        self.assertEqual(initial.spreading_scale, 1.0)
        self.assertEqual(initial.volume_scale, 1.0)
        self.assertAlmostEqual(state.spreading_scale, expected_spreading)
        self.assertAlmostEqual(state.volume_scale, expected_volume)
        self.assertEqual(state, same_state)

    def test_directional_scales_conserve_remaining_cross_section(self):
        state = BeadEvolutionModel(
            spreading_ratio=1.8,
            spreading_time_s=1.2,
            shrinkage_fraction=0.15,
            shrinkage_time_s=8.0,
        ).state_at(3.0)

        self.assertAlmostEqual(
            state.width_scale * state.height_scale,
            state.volume_scale,
        )
        self.assertAlmostEqual(
            state.visual_radius_scale,
            math.sqrt(state.spreading_scale * state.volume_scale),
        )

    def test_large_age_approaches_configured_limits(self):
        state = BeadEvolutionModel(
            spreading_ratio=1.4,
            spreading_time_s=2.0,
            shrinkage_fraction=0.1,
            shrinkage_time_s=3.0,
        ).state_at(1.0e6)

        self.assertAlmostEqual(state.spreading_scale, 1.4)
        self.assertAlmostEqual(state.volume_scale, 0.9)

    def test_builds_from_material_profile(self):
        profile = MaterialProfile(
            profile_id="evolving_hydrogel",
            name="evolving hydrogel",
            extrusion_mode="volumetric",
            spreading_ratio=1.25,
            spreading_time_s=2.5,
            shrinkage_fraction=0.05,
            shrinkage_time_s=20.0,
        )

        model = BeadEvolutionModel.from_material_profile(profile)

        self.assertEqual(model.spreading_ratio, 1.25)
        self.assertEqual(model.spreading_time_s, 2.5)
        self.assertEqual(model.shrinkage_fraction, 0.05)
        self.assertEqual(model.shrinkage_time_s, 20.0)

    def test_rejects_invalid_parameters(self):
        invalid_models = (
            ({"spreading_ratio": 0.99}, "spreading_ratio"),
            ({"spreading_time_s": 0.0}, "spreading_time_s"),
            ({"shrinkage_fraction": -0.01}, "shrinkage_fraction"),
            ({"shrinkage_fraction": 1.0}, "shrinkage_fraction"),
            ({"shrinkage_time_s": 0.0}, "shrinkage_time_s"),
        )
        for keyword_arguments, field_name in invalid_models:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    BeadEvolutionModel(**keyword_arguments)

    def test_rejects_negative_or_non_finite_age(self):
        model = BeadEvolutionModel()
        for age in (-0.01, math.inf, math.nan):
            with self.subTest(age=age):
                with self.assertRaisesRegex(ValueError, "age_s"):
                    model.state_at(age)


if __name__ == "__main__":
    unittest.main()
