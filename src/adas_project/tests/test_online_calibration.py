"""Online recalibration of the crash predictor (adas/online_calibration.py)."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.online_calibration import OnlineCalibrator


def simulate(true_a, true_b, n=3000, seed=0, cal=None):
    """A model whose logits are mis-calibrated: the truth is sigmoid(true_a * x + true_b) instead of sigmoid(x). Outcomes are fed
    straight to the update (what the resolution of 'event within 2 s' delivers)."""
    rng = np.random.default_rng(seed)
    cal = cal or OnlineCalibrator()
    for _ in range(n):
        x = rng.normal(-1.0, 1.8)
        y = 1.0 if rng.random() < 1 / (1 + math.exp(-(true_a * x + true_b))) else 0.0
        cal._update(x, y)
    return cal


class CalibratorTest(unittest.TestCase):
    def test_learns_a_shift_of_the_base_rate(self):
        cal = simulate(1.0, 1.2)                                  # this driver is in trouble far more often than the model thinks
        self.assertGreater(cal.probability(-1.0), 1 / (1 + math.exp(1.0)) + 0.15)
        self.assertAlmostEqual(cal.theta[1], 1.2, delta=0.5)

    def test_learns_overconfidence(self):
        cal = simulate(0.5, 0.0)                                  # the model's probabilities are too extreme
        self.assertLess(cal.theta[0], 0.8)

    def test_identity_when_the_model_is_right(self):
        cal = simulate(1.0, 0.0)
        for x in (-3.0, -1.0, 0.5):
            self.assertAlmostEqual(cal.probability(x), 1 / (1 + math.exp(-x)), delta=0.08)

    def test_inactive_until_enough_feedback(self):
        cal = OnlineCalibrator()
        for k in range(10):
            cal.observe(k * 0.25, 1.0)
        self.assertFalse(cal.active)
        self.assertAlmostEqual(cal.probability(1.0), 1 / (1 + math.exp(-1.0)))

    def test_outcomes_come_from_safety_events_two_seconds_later(self):
        cal = OnlineCalibrator(min_updates=1)
        cal.observe(0.0, 0.0)          # will be followed by an event 1 s later -> outcome 1
        cal.observe(10.0, 0.0)         # no event within 2 s -> outcome 0
        cal.event(1.0)
        cal.observe(12.5, 0.0)         # resolves the prediction made at 10.0
        self.assertEqual(cal.n, 2)     # 0.0 was resolved when t=10 arrived, 10.0 when t=12.5 arrived
        self.assertEqual(len(cal.pending), 1)

    def test_follows_a_driver_who_changes(self):
        cal = simulate(1.0, 1.0, n=1500)
        b1 = cal.theta[1]
        simulate(1.0, -1.0, n=2500, seed=1, cal=cal)              # continues the same calibrator (clock restarts: fine)
        self.assertLess(cal.theta[1], b1 - 0.5)


if __name__ == "__main__":
    unittest.main()
