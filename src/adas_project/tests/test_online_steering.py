"""Online steering calibration (adas/online_steering.py): recovers the servo centre and gain of a simulated car."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.online_steering import OnlineSteering

K0, C0 = 0.0656, 87.0


def drive(true_c, true_k, seconds=40.0, v=0.4, seed=0, noise_deg=0.15):
    """A bicycle car steered through varied angles; poses with heading noise (scan matching's error)."""
    rng = np.random.default_rng(seed)
    est = OnlineSteering(C0, K0, prior_weight=2)
    x = y = th = 0.0
    t, dt = 0.0, 0.05
    servo_cmd = C0
    while t < seconds:
        servo_cmd = C0 + 14.0 * math.sin(0.35 * t) + 6.0 * math.sin(1.3 * t)          # varied, mostly steady steering
        est.command(t, servo_cmd)
        servo_true = servo_cmd                                                        # (the delay is modelled by DELAY_S)
        kappa = -true_k * (servo_true - true_c)
        th += kappa * v * dt
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        t += dt
        est.update(t, v, (x, y, th + math.radians(rng.normal(0, noise_deg))))
    return est


class OnlineSteeringTest(unittest.TestCase):
    def test_recovers_centre_and_gain(self):
        for true_c, true_k in ((87.0 + 3.0, K0 * 1.1), (87.0 - 2.5, K0 * 0.9), (87.0, K0)):
            est = drive(true_c, true_k)
            self.assertLess(abs(est.centre - true_c), 0.8, (true_c, est.status()))
            self.assertLess(abs(est.k / true_k - 1), 0.08, (true_k, est.status()))

    def test_stays_near_the_prior_with_no_data_and_is_clipped(self):
        est = OnlineSteering(C0, K0)
        self.assertAlmostEqual(est.centre, C0, delta=0.05)
        est2 = drive(C0 + 12.0, K0, seconds=30)            # an absurd true centre: the estimate is clipped
        self.assertLessEqual(abs(est2.centre - C0), OnlineSteering.MAX_CENTRE_DEG + 1e-6)


if __name__ == "__main__":
    unittest.main()
