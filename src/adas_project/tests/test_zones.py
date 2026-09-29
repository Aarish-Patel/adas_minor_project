"""Speed-limit zones (pi/zones.py) and the relay's throttle cap (pi/relay_assists.RelayAssists)."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pi.zones import SpeedZones, car_to_kph, contains, kph_to_car


class Speed:
    """Stands in for RelaySpeed: a world pose and a speed."""

    def __init__(self, pose, v):
        self.pose, self.v = pose, v


class ZonesTest(unittest.TestCase):
    def test_scaling(self):
        self.assertAlmostEqual(kph_to_car(50), 0.992, places=3)     # 1:14 -> about full speed on this car
        self.assertAlmostEqual(car_to_kph(kph_to_car(20)), 20)

    def test_shapes(self):
        self.assertTrue(contains({"kind": "rect", "x0": 2, "y0": 1, "x1": 0, "y1": -1}, 1.0, 0.0))
        self.assertFalse(contains({"kind": "circle", "x": 0, "y": 0, "r": 0.5}, 0.6, 0.0))
        tri = {"kind": "poly", "pts": [[0, 0], [2, 0], [0, 2]]}
        self.assertTrue(contains(tri, 0.4, 0.4))
        self.assertFalse(contains(tri, 1.5, 1.5))

    def test_lowest_limit_wins_and_bad_zones_are_dropped(self):
        z = SpeedZones()
        n = z.set('[{"kind": "rect", "x0": 0, "y0": -1, "x1": 3, "y1": 1, "kph": 30},'
                  ' {"kind": "circle", "x": 1, "y": 0, "r": 0.4, "kph": 10}, {"kind": "blob", "kph": 5},'
                  ' {"kind": "rect", "kph": 5}]')
        self.assertEqual(n, 2)
        self.assertEqual(z.limit_kph(1.0, 0.0), 10)
        self.assertEqual(z.limit_kph(2.5, 0.0), 30)
        self.assertIsNone(z.limit_kph(5.0, 0.0))

    def test_slows_before_entering(self):
        z = SpeedZones()
        z.set([{"kind": "rect", "x0": 1.0, "y0": -1, "x1": 3, "y1": 1, "kph": 10}])
        self.assertEqual(z.limit_ahead((0.7, 0.0, 0.0), 0.8), 10)   # 0.4 m ahead at 0.8 m/s: inside
        self.assertIsNone(z.limit_ahead((0.7, 0.0, math.pi), 0.8))  # driving away from it


class RelayCapTest(unittest.TestCase):
    def setUp(self):
        from adas.config import load_tuning
        from pi.relay_assists import RelayAssists
        self.a = RelayAssists(load_tuning(os.path.join(os.path.dirname(__file__), "..", "pi", "tuning_real_car.json")))
        self.c = f"{self.a.centre:.0f}"

    def tearDown(self):
        self.a.planner.shutdown()

    def throttle(self, pose, v, pwm=-250):
        self.a.speed = Speed(pose, v)
        out = self.a.process([f"A {self.c} {self.c}", f"M {pwm}"], [(0.0, 3.0)], 1, now=0.0)
        return int(next(l for l in out if l.startswith("M ")).split()[1])

    def test_no_zone_full_throttle(self):
        self.assertEqual(self.throttle((0, 0, 0), 0.5), -250)

    def test_inside_a_zone_the_throttle_is_capped(self):
        self.a.zones.set([{"kind": "circle", "x": 0, "y": 0, "r": 1.0, "kph": 10}])
        w = self.throttle((0, 0, 0), 0.5)
        self.assertLess(abs(w), 250)
        self.assertAlmostEqual(abs(w), self.a.model.pwm_for_speed(kph_to_car(10)), delta=1)
        self.assertEqual(self.a.zone_kph, 10)
        self.assertEqual(self.throttle((0, 0, 0), 0.5, pwm=60), 60)    # slower than the limit: untouched


if __name__ == "__main__":
    unittest.main()
