"""Closed-loop speed control (adas/speed_control.py)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.aeb import SpeedModel
from adas.speed_control import SpeedController


class Plant:
    """First-order motor lag; the car is `scale` of what the model believes."""

    def __init__(self, model, scale, tau=0.12):
        self.m, self.scale, self.tau, self.v = model, scale, tau, 0.0

    def step(self, pwm, dt):
        target = self.m.speed(pwm) * self.scale
        self.v += (target - self.v) * dt / self.tau
        return self.v


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.model = SpeedModel(v_max=0.9, deadband=60.0)

    def settle(self, scale, closed, target=0.5, secs=6.0):
        ctl, plant, dt = SpeedController(self.model), Plant(self.model, scale), 0.05
        v = 0.0
        for _ in range(int(secs / dt)):
            u = ctl.pwm(target, v, dt) if closed else self.model.pwm_for_speed(target)
            v = plant.step(u, dt)
        return v

    def test_removes_the_steady_state_error_of_a_slow_car(self):
        open_err = abs(self.settle(0.75, False) - 0.5)
        closed_err = abs(self.settle(0.75, True) - 0.5)
        self.assertGreater(open_err, 0.08)
        self.assertLess(closed_err, 0.025)

    def test_matches_the_model_when_the_model_is_right(self):
        self.assertAlmostEqual(self.settle(1.0, True), 0.5, delta=0.02)

    def test_antiwindup_and_limits(self):
        ctl = SpeedController(self.model)
        for _ in range(400):                                   # a car that cannot reach the target (blocked wheel)
            u = ctl.pwm(0.8, 0.0, 0.05)
        self.assertLessEqual(abs(u), 255.0)
        self.assertLessEqual(abs(ctl.i), ctl.i_max)
        # released: recovers without a long overshoot because the integrator is bounded
        self.assertLess(abs(ctl.pwm(0.8, 0.8, 0.05)), 255.0)

    def test_zero_target_resets_and_direction_change_restarts(self):
        ctl = SpeedController(self.model)
        for _ in range(50):
            ctl.pwm(0.4, 0.2, 0.05)
        self.assertNotEqual(ctl.i, 0.0)
        self.assertEqual(ctl.pwm(0.0, 0.2, 0.05), 0.0)
        self.assertEqual(ctl.i, 0.0)
        ctl.pwm(0.4, 0.3, 0.05)
        u = ctl.pwm(-0.2, 0.0, 0.05)                           # reverse creep: starts from the feedforward, negative
        self.assertLess(u, 0.0)


if __name__ == "__main__":
    unittest.main()
