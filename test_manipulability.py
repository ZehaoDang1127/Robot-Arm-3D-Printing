from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from robotic_printing_platform.robots.franka_panda import IKConfig, jacobian_quality


class ManipulabilityTests(unittest.TestCase):
    @patch("robotic_printing_platform.robots.franka_panda.urdf_geometric_jacobian")
    def test_uses_product_of_singular_values_for_yoshikawa_manipulability(self, jacobian):
        matrix = np.zeros((6, 7), dtype=float)
        matrix[:, :6] = np.diag([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        jacobian.return_value = matrix

        manipulability, sigma_min = jacobian_quality(np.zeros(7), IKConfig())

        self.assertEqual(manipulability, 720.0)
        self.assertEqual(sigma_min, 1.0)


if __name__ == "__main__":
    unittest.main()
