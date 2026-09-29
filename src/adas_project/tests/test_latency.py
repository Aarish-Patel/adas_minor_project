"""Online scan-latency estimation (adas/latency.py)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.latency import DelayEstimator


def speed(t):
    """A throttle-like profile: rest, ramp up, hold, ramp down."""
    return float(np.interp(t, [0, 0.5, 1.5, 2.5, 3.0, 4.0], [0, 0, 0.7, 0.7, 0.2, 0.2]))


class DelayTest(unittest.TestCase):
    def run_est(self, delay, noise=0.02, seed=0):
        rng = np.random.default_rng(seed)
        est = DelayEstimator(nominal_s=0.0)
        t = 0.0
        while t < 4.0:
            est.push_model(t, speed(t))
            if int(round(t / 0.05)) % 2 == 0:                      # a flow measurement every 0.1 s
                est.push_flow(t, speed(t - delay) + rng.normal(0, noise))
            t += 0.05
        return est

    def test_recovers_the_delay(self):
        for d in (0.0, 0.1, 0.2, 0.35):
            est = self.run_est(d)
            self.assertAlmostEqual(est.delay, d, delta=0.06, msg=f"delay {d}: got {est.delay:.3f}")

    def test_constant_speed_gives_no_estimate(self):
        est = DelayEstimator(nominal_s=0.1)
        for i in range(120):
            t = i * 0.05
            est.push_model(t, 0.5)
            if i % 2 == 0:
                est.push_flow(t, 0.5)
        self.assertEqual(est.tested, 0)
        self.assertEqual(est.excess, 0.0)

    def test_excess_is_capped_and_never_negative(self):
        est = self.run_est(0.45)
        self.assertLessEqual(est.excess, est.cap)
        est2 = self.run_est(0.0)
        est2.nominal = 0.3
        self.assertEqual(est2.excess, 0.0)


if __name__ == "__main__":
    unittest.main()
