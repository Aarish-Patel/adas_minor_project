"""Predictive speed choice for moving obstacles (adas/crossing.py): pass, yield, stop, back away, clear."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.config import load_tuning
from adas.crossing import CrossingConfig, arc_path, decide
from pi.relay_assists import car_params

P = car_params(load_tuning(os.path.join(os.path.dirname(__file__), "..", "pi", "tuning_real_car.json")).mount)
STRAIGHT = arc_path(0.0)


class CrossingTest(unittest.TestCase):
    def test_nothing_moving(self):
        self.assertEqual(decide([(1.0, 0.0, 0.0, 0.0, 0.1)], STRAIGHT, 0.4, 0.4, P)[0], "clear")

    def test_crossing_far_away_in_time_is_ignored(self):
        # walking across 2.5 m ahead but still 3 m to the side at 0.3 m/s: arrives after the car has gone by
        self.assertEqual(decide([(2.5, 3.0, 0.0, -0.3, 0.1)], STRAIGHT, 0.5, 0.5, P)[0], "clear")

    def test_yields_to_someone_about_to_cross(self):
        # a person 1.2 m ahead, 0.8 m to the left, walking across at 0.3 m/s: the car at 0.3 m/s would meet them
        act, v, info = decide([(1.2, 0.8, 0.0, -0.3, 0.12)], STRAIGHT, 0.3, 0.3, P)
        self.assertIn(act, ("yield", "wait"))
        self.assertLess(v, 0.3)

    def test_speeds_up_to_pass_a_late_crosser(self):
        # someone 1.2 m ahead, still 1.6 m to the side, walking across at 0.5 m/s: at 0.5 m/s the car would arrive
        # as they do - a little faster and it is through first
        act, v, _ = decide([(1.2, 1.6, 0.0, -0.5, 0.12)], STRAIGHT, 0.5, 0.5, P)
        self.assertEqual(act, "pass")
        self.assertGreater(v, 0.5)
        self.assertLessEqual(v, 0.5 + CrossingConfig().pass_gain_max + 1e-9)

    def test_head_on_backs_away_if_it_can(self):
        head_on = [(1.2, 0.0, -0.6, 0.0, 0.12)]                 # coming straight down the car's path
        self.assertEqual(decide(head_on, STRAIGHT, 0.0, 0.3, P, rear_free=1.0)[0], "away")
        self.assertEqual(decide(head_on, STRAIGHT, 0.0, 0.3, P, rear_free=0.1)[0], "stop")


if __name__ == "__main__":
    unittest.main()
