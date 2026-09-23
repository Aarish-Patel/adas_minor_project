"""Obstacle-bypass controller + scan-matching odometry, in the ray-cast simulator (pi/sim_bypass.py)."""
import io
import math
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pi import sim_bypass as S  # noqa: E402


def run(*a, **k):
    with redirect_stdout(io.StringIO()):
        return S.run(*a, **k)


def wall(x0, y0, x1, y1):
    return S.box(x0, y0, x1, y1)


class Bypass(unittest.TestCase):
    def check_rejoined(self, out, max_y=0.06, max_th=4.0, min_clear=0.02):
        state, clear, fin = out
        self.assertEqual(state, "DONE")
        self.assertGreater(clear, min_clear, "body came within %.0f mm of the obstacle" % (clear * 1000))
        self.assertLess(abs(fin[1]), max_y)
        self.assertLess(abs(math.degrees(fin[2])), max_th)

    def test_dead_ahead(self):
        self.check_rejoined(run((1.4, -0.10, 1.6, 0.10), [], "open"))

    def test_offset_obstacle(self):
        self.check_rejoined(run((1.4, 0.00, 1.6, 0.22), [], "offset"))

    def test_wide_obstacle(self):
        self.check_rejoined(run((1.4, -0.20, 1.6, 0.20), [], "wide"), max_y=0.08)

    def test_chooses_open_side(self):
        # wall close on +y: must go around on -y (and vice versa) and still rejoin
        self.check_rejoined(run((1.4, -0.10, 1.6, 0.10), wall(0.8, 0.45, 2.6, 0.5), "+y blocked"), max_y=0.12)
        self.check_rejoined(run((1.4, -0.10, 1.6, 0.10), wall(0.8, -0.5, 2.6, -0.45), "-y blocked"), max_y=0.12)

    def test_refuses_when_boxed_in(self):
        state, clear, fin = run((1.4, -0.10, 1.6, 0.10), wall(0.8, 0.35, 2.6, 0.4) + wall(0.8, -0.4, 2.6, -0.35), "boxed")
        self.assertEqual(state, "ABORT")
        self.assertGreater(clear, 0.5)          # it must not have driven toward the obstacle

    def test_robust_to_steering_mismatch(self):
        self.check_rejoined(run((1.4, -0.10, 1.6, 0.10), [], "weak", k_true=0.75, servo_bias=1.5), max_y=0.08, max_th=5)
        self.check_rejoined(run((1.4, 0.00, 1.6, 0.22), [], "strong", k_true=1.3, servo_bias=-1.5), max_y=0.08, max_th=5)

    def test_repeatable_over_noise_seeds(self):
        for seed in range(6):
            self.check_rejoined(run((1.4, -0.10, 1.6, 0.10), [], "seed", seed=seed), max_y=0.08, max_th=5)


if __name__ == "__main__":
    unittest.main()
