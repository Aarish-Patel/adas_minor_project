"""Reverse-camera ghost car and assists (TODO Q1-Q3)."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.markers import Camera
from adas.vision.guidelines import draw_reverse_assist, first_contact, floor_patches, footprint_at
from adas.vision.sim_camera import SimCamera

REAR = Camera(x=-0.05, z=0.12, pitch_deg=32.0, yaw_deg=180.0, hfov_deg=70.0)
W, RX, FX = 0.20, -0.05, 0.28


class GhostTest(unittest.TestCase):
    def test_footprint_moves_back_when_reversing_straight(self):
        p0, p1 = footprint_at(0.0, 0.0, W, RX, FX), footprint_at(0.0, 0.5, W, RX, FX)
        self.assertAlmostEqual(p0[:, 0].min() - p1[:, 0].min(), 0.5, places=6)
        self.assertAlmostEqual(abs(p1[:, 1]).max(), W / 2, places=6)

    def test_contact_distance_straight(self):
        # a point 0.6 m behind the origin, on the axis: the rear bumper (x=-0.05) reaches it after 0.55 m
        self.assertAlmostEqual(first_contact(0.0, [(-0.6, 0.0)], W, RX, FX), 0.55, delta=0.03)
        self.assertIsNone(first_contact(0.0, [(-0.6, 0.4)], W, RX, FX))            # beside the path
        self.assertIsNone(first_contact(0.0, [(-1.5, 0.0)], W, RX, FX, length=1.2))    # beyond the look-ahead

    def test_steering_swings_the_path_into_or_away_from_a_hazard(self):
        h = [(-0.7, 0.25)]
        self.assertIsNone(first_contact(0.0, h, W, RX, FX))
        hits = [first_contact(k, h, W, RX, FX) for k in (-2.0, -1.0, 1.0, 2.0)]
        self.assertTrue(any(x is not None for x in hits))                            # one of the arcs must sweep into it

    def test_overlay_draws_and_reports(self):
        sim = SimCamera(REAR, (320, 240), seed=1)
        f = sim.render((0.0, 0.0, 0.0))
        img, info = draw_reverse_assist(f, sim.cam, 0.0, W, RX, FX, hazards=[(-0.5, 0.0)], speed=0.3)
        self.assertEqual(img.shape, f.shape)
        self.assertGreater(np.abs(img.astype(int) - f.astype(int)).sum(), 0)
        self.assertAlmostEqual(info["contact_m"], 0.45, delta=0.03)
        self.assertTrue(info["stop"] is False)
        self.assertGreater(info["ttc_s"], 1.0)
        img2, info2 = draw_reverse_assist(f, sim.cam, 0.0, W, RX, FX, hazards=[(-0.3, 0.0)], speed=0.3)
        self.assertTrue(info2["stop"])
        _, info3 = draw_reverse_assist(f, sim.cam, 0.0, W, RX, FX, hazards=[])
        self.assertIsNone(info3["contact_m"])

    def test_floor_patch_found_in_the_path_and_not_on_a_clean_floor(self):
        import cv2
        sim = SimCamera(REAR, (320, 240), seed=2)
        f = sim.render((0.0, 0.0, 0.0))
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        cam = sim.cam
        self.assertEqual(floor_patches(g, cam, 0.0, W, RX), [])
        from adas.markers import project
        uv, z = project(cam, np.array([[-0.6, 0.0, 0.0]]))
        wet = g.copy()
        cv2.circle(wet, (int(uv[0][0]), int(uv[0][1])), 14, int(g.mean() * 0.3), -1)   # a dark puddle-sized disc
        found = floor_patches(wet, cam, 0.0, W, RX)
        self.assertTrue(found)
        self.assertAlmostEqual(found[0][0], 0.55, delta=0.25)


if __name__ == "__main__":
    unittest.main()
