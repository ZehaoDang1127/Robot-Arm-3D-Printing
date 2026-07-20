from __future__ import annotations

import unittest

from robotic_printing_platform.gcode.parser import Move, ParseResult
from robotic_printing_platform.path_planning.layered import build_waypoints


def _move(x, y, z, de, feed, layer, *, rapid=False) -> Move:
    return Move(
        x=x,
        y=y,
        z=z,
        e=0.0,
        de=de,
        f=feed,
        has_e=de != 0.0,
        is_print=de > 0.0,
        layer=layer,
        rapid=rapid,
    )


class LayeredPathMetadataTests(unittest.TestCase):
    def test_preserves_layer_feedrate_and_extrusion_through_densification(self):
        moves = [
            _move(0, 0, 0.2, 0.0, 1200, 0),
            _move(2, 0, 0.2, 2.0, 1200, 0),
            _move(4, 0, 0.2, 4.0, 2400, 0),
            _move(4, 2, 0.2, 0.0, 6000, 0, rapid=True),
            _move(4, 2, 0.4, 0.0, 6000, 1, rapid=True),
            _move(6, 2, 0.4, 3.0, 1800, 1),
        ]
        parsed = ParseResult(
            moves=moves,
            layer_count=2,
            bbox_min=(0, 0, 0.2),
            bbox_max=(6, 2, 0.4),
            n_print=3,
            n_travel=3,
            units="mm",
            used_z_inference=False,
        )

        path = build_waypoints(parsed, (0, 2), max_seg_len_mm=1.0, simplify_deg=5.0)

        self.assertEqual({waypoint.layer for waypoint in path.waypoints}, {0, 1})
        self.assertEqual(path.source_extrusion_mm, 9.0)
        self.assertAlmostEqual(path.waypoint_extrusion_mm, 9.0)
        self.assertTrue(all(waypoint.feed_m_s > 0.0 for waypoint in path.waypoints))
        self.assertEqual(
            {waypoint.feed_m_s for waypoint in path.waypoints if waypoint.is_print},
            {0.02, 0.03, 0.04},
        )
        self.assertIn(0.1, {waypoint.feed_m_s for waypoint in path.waypoints if not waypoint.is_print})
        self.assertEqual(path.layer_statistics()[0]["print_waypoints"], 5)
        self.assertEqual(path.layer_statistics()[1]["print_waypoints"], 3)


if __name__ == "__main__":
    unittest.main()
