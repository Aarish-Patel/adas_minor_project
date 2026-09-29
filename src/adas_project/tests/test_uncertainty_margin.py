"""Uncertainty-aware brake margins (PathGate.v_effective, RelaySpeed.sigma_v)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.config import load_tuning
from pi.path_gate import PathGate
from pi.relay_assists import car_params


class Src:
    def __init__(self, s):
        self.sigma_v = s


class UncertaintyTest(unittest.TestCase):
    def setUp(self):
        tun = load_tuning(os.path.join(os.path.dirname(__file__), "..", "pi", "tuning_real_car.json"))
        self.gate = PathGate(car_params(tun.mount), tun.speed_model)

    def test_no_source_or_sharp_estimate_changes_nothing(self):
        self.assertEqual(self.gate.v_effective(0.5), 0.5)
        self.gate.uncertainty_source = Src(0.02)                 # the EKF's normal spread
        self.assertEqual(self.gate.v_effective(0.5), 0.5)
        self.assertEqual(self.gate.v_effective(-0.4), -0.4)

    def test_uncertain_speed_is_planned_for_at_its_95th_percentile(self):
        self.gate.uncertainty_source = Src(0.13)
        self.assertAlmostEqual(self.gate.v_effective(0.5), 0.5 + 1.645 * (0.13 - 0.03), places=6)
        self.assertAlmostEqual(self.gate.v_effective(-0.5), -(0.5 + 1.645 * 0.10), places=6)
        self.assertGreater(self.gate.v_effective(0.0), 0.0)      # even 'standing' is not trusted when the estimate is that loose

    def test_allowed_speed_shrinks_with_the_extra_margin(self):
        base = self.gate.allowed_speed(1.0)
        self.gate.delay_source = Src(0.0)
        self.gate.delay_source.excess = 0.2
        self.assertLess(self.gate.allowed_speed(1.0), base)


if __name__ == "__main__":
    unittest.main()
