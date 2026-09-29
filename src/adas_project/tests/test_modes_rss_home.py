"""Drive modes, the RSS safe distance and the return-to-start goal."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.rss import RSSParams, lateral_min_distance, longitudinal_min_distance, margin


class RSSTest(unittest.TestCase):
    def test_paper_formula_for_a_stationary_object(self):
        p = RSSParams()
        v = 0.8
        want = v * p.rho + 0.5 * p.a_accel * p.rho ** 2 + (v + p.rho * p.a_accel) ** 2 / (2 * p.b_min)
        self.assertAlmostEqual(longitudinal_min_distance(v, 0.0, p), want, places=9)

    def test_grows_with_speed_and_shrinks_with_a_moving_leader(self):
        self.assertGreater(longitudinal_min_distance(0.8), longitudinal_min_distance(0.4))
        self.assertLess(longitudinal_min_distance(0.8, 0.6), longitudinal_min_distance(0.8, 0.0))
        self.assertEqual(longitudinal_min_distance(0.0), 0.5 * RSSParams().a_accel * RSSParams().rho ** 2 + (RSSParams().rho * RSSParams().a_accel) ** 2 / (2 * RSSParams().b_min))

    def test_margin_sign(self):
        need = longitudinal_min_distance(0.6)
        self.assertGreater(margin(need + 0.1, 0.6), 0)
        self.assertLess(margin(need - 0.1, 0.6), 0)

    def test_lateral_distance_is_at_least_mu(self):
        self.assertGreaterEqual(lateral_min_distance(0.0, 0.0), RSSParams().mu)
        self.assertGreater(lateral_min_distance(0.3, -0.3), lateral_min_distance(0.0, 0.0))


class ModesTest(unittest.TestCase):
    def setUp(self):
        from adas.config import load_tuning
        from pi.relay_assists import RelayAssists
        self.a = RelayAssists(load_tuning(os.path.join(os.path.dirname(__file__), "..", "pi", "tuning_real_car.json")))
        self.c = f"{self.a.centre:.0f}"

    def tearDown(self):
        self.a.planner.shutdown()

    def throttle(self, pwm=-255):
        out = self.a.process([f"A {self.c} {self.c}", f"M {pwm}"], [(0.0, 3.0)], 1, now=0.0)
        return int(next(l for l in out if l.startswith("M ")).split()[1])

    def test_eco_caps_the_throttle_and_normal_does_not(self):
        from pi.relay_assists import DRIVE_MODES
        self.assertEqual(self.throttle(), -255)
        self.assertTrue(self.a.set_mode("eco"))
        self.assertLessEqual(abs(self.throttle()), 255 * DRIVE_MODES["eco"]["cap"] + 1)
        self.assertEqual(self.throttle(-100), -100)                       # below the cap: untouched
        self.assertTrue(self.a.set_mode("sport"))
        self.assertEqual(self.throttle(), -255)
        self.assertFalse(self.a.set_mode("warp"))


class HomeTest(unittest.TestCase):
    def test_goal_and_heading_from_a_world_pose(self):
        from pi.relay_assists import home_goal
        x, y, h = home_goal((0.0, 0.0, 0.0))
        self.assertAlmostEqual(math.hypot(x, y), 0.0)
        # 2 m ahead of the origin, facing away from it (heading 0): the origin is 2 m behind
        x, y, h = home_goal((2.0, 0.0, 0.0))
        self.assertAlmostEqual(x, -2.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(h, 0.0)
        # at (1, 1), turned 90 deg left: the origin is behind-right... (-1,-1) rotated by -90 -> (-1, 1)
        x, y, h = home_goal((1.0, 1.0, math.pi / 2))
        self.assertAlmostEqual(x, -1.0)
        self.assertAlmostEqual(y, 1.0)
        self.assertAlmostEqual(h, -90.0)


if __name__ == "__main__":
    unittest.main()
