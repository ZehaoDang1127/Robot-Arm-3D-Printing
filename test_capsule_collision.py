from __future__ import annotations

import unittest

import numpy as np

from robotic_printing_platform.validation.collision import (
    AxisAlignedBox,
    LinkCapsule,
    capsule_box_distance_m,
    collision_warnings,
    segment_segment_distance_m,
)


class CapsuleCollisionTests(unittest.TestCase):
    def test_detects_bed_self_and_printed_material_warnings(self):
        capsules = [
            LinkCapsule("link_0", np.array([0.0, 0.0, 0.11]), np.array([0.1, 0.0, 0.11]), 0.01),
            LinkCapsule("link_1", np.array([2.0, 0.0, 1.0]), np.array([2.1, 0.0, 1.0]), 0.01),
            LinkCapsule("link_2", np.array([0.05, -0.1, 0.11]), np.array([0.05, 0.1, 0.11]), 0.01),
            LinkCapsule("tool", np.array([0.0, 0.0, 0.2]), np.array([0.0, 0.0, 0.1]), 0.005),
        ]
        bed = AxisAlignedBox(np.array([-0.2, -0.2, 0.08]), np.array([0.2, 0.2, 0.1]))
        printed = AxisAlignedBox(np.array([-0.01, -0.01, 0.09]), np.array([0.01, 0.01, 0.11]))

        warnings = collision_warnings(
            capsules,
            bed_box=bed,
            bed_z_m=0.1,
            bed_clearance_m=0.005,
            printed_volume=printed,
        )

        self.assertTrue(any("bed clearance" in warning for warning in warnings))
        self.assertTrue(any("self-collision" in warning for warning in warnings))
        self.assertTrue(any("previously printed material" in warning for warning in warnings))

    def test_segment_and_capsule_box_distance(self):
        self.assertAlmostEqual(
            segment_segment_distance_m(
                np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                np.array([0.5, -1.0, 0.0]), np.array([0.5, 1.0, 0.0]),
            ),
            0.0,
        )
        capsule = LinkCapsule("link", np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0]), 0.1)
        box = AxisAlignedBox(np.array([0.4, -0.1, 0.8]), np.array([0.6, 0.1, 0.9]))
        self.assertAlmostEqual(capsule_box_distance_m(capsule, box), 0.0)


if __name__ == "__main__":
    unittest.main()
