import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pi.scanmatch import icp, polar_to_xy, rot  # noqa: E402


def room_scan(pose, rng):
    """Points of a 3 x 2 m room seen from `pose` (x, y, th) - a crude ray cast."""
    x, y, th = pose
    pts = []
    for a in np.linspace(-np.pi, np.pi, 360, endpoint=False):
        d = np.array([np.cos(a + th), np.sin(a + th)])
        best = 9
        for axis, lo, hi in ((0, -1.0, 2.0), (1, -1.0, 1.0)):
            for wall in (lo, hi):
                if abs(d[axis]) > 1e-9:
                    t = (wall - (x, y)[axis]) / d[axis]
                    if 0.05 < t < best:
                        best = t
        pts.append((np.degrees(a), best + rng.normal(0, 0.005)))
    return polar_to_xy(pts)


class ICP(unittest.TestCase):
    def test_recovers_motion(self):
        rng = np.random.default_rng(1)
        A = room_scan((0.0, 0.0, 0.0), rng)
        for dx, dy, dth in ((0.03, 0.0, 0.0), (0.05, 0.02, 0.05), (0.04, -0.03, -0.08)):
            B = room_scan((dx, dy, dth), rng)
            res = icp(A, B, init=(dx * 0.8, 0.0, 0.0))
            self.assertIsNotNone(res)
            R, t, th, resid, n = res
            self.assertAlmostEqual(t[0], dx, delta=0.012)
            self.assertAlmostEqual(t[1], dy, delta=0.012)
            self.assertAlmostEqual(th, dth, delta=np.radians(1.0))

    def test_rejects_too_few_points(self):
        self.assertIsNone(icp(np.zeros((10, 2)), np.zeros((10, 2))))

    def test_rot(self):
        self.assertTrue(np.allclose(rot(np.pi / 2) @ np.array([1, 0]), [0, 1]))


if __name__ == "__main__":
    unittest.main()
