"""Moving-object tracking: cluster ends, the likelihood-ratio moving test, and its twin evaluation (adas/tracking.py, sim/tracking_eval.py)."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.tracking import Tracker, clusters_ex, cluster_points


def arc_points(cx, cy, r, n=7, lidar_x=0.12):
    """The front half of a circle as the LiDAR sees it: n points on the arc facing the sensor."""
    a0 = math.atan2(cy, cx - lidar_x)
    ang = a0 + math.pi + np.linspace(-1.2, 1.2, n)
    return np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])


class ClusterEnds(unittest.TestCase):
    def scene(self):
        """A far wall (x = 6) with a small object 4 m in front of it, all seen from the LiDAR: points ordered by angle."""
        wall = np.column_stack([np.full(120, 6.0), np.arange(-3.0, 3.0, 0.05)])
        obj = arc_points(2.0, 0.6, 0.06)
        shadow = np.arctan2(0.6, 2.0 - 0.12)                            # the object hides the wall behind it (one return per beam)
        wall = wall[np.abs(np.arctan2(wall[:, 1], wall[:, 0] - 0.12) - shadow) > 0.07]
        allp = np.vstack([wall, obj])
        ang = np.arctan2(allp[:, 1], allp[:, 0] - 0.12)
        return allp[np.argsort(ang)]

    def test_an_object_in_front_of_a_background_has_real_edges(self):
        cl = [c for c in clusters_ex(self.scene()) if math.hypot(c[0][0] - 2.0, c[0][1] - 0.6) < 0.2]
        self.assertEqual(len(cl), 1)
        self.assertEqual(cl[0][3], 0)                                   # no cut end

    def test_a_wall_cut_into_pieces_has_cut_ends(self):
        # a wall seen at a grazing angle with the gaps of the LiDAR's beam spacing: pieces of 4 points, 0.1 m gaps between them
        xs = np.concatenate([np.arange(0, 0.15, 0.05) + 0.4 * k for k in range(6)])
        pts = np.column_stack([1.0 + xs, np.full(len(xs), 2.0)])
        soft = [c[3] for c in clusters_ex(pts)]
        self.assertTrue(soft and all(s >= 1 for s in soft))
        self.assertTrue(any(s == 2 for s in soft))

    def test_cluster_points_still_returns_centre_and_radius(self):
        out = cluster_points(arc_points(1.5, 0.3, 0.08))
        self.assertEqual(len(out), 1)
        self.assertLess(math.hypot(out[0][0][0] - 1.5, out[0][0][1] - 0.3), 0.1)


class WindowFit(unittest.TestCase):
    def test_weighted_line_fit_matches_a_direct_solution(self):
        tk = Tracker()
        rng = np.random.default_rng(3)
        from adas.tracking import Track
        tr = Track(1)
        v_true = np.array([0.2, -0.4])
        for k in range(7):
            t = 0.1 * k
            p = np.array([2.0, 0.5]) + v_true * t + rng.normal(0, 0.01, 2)
            tr.hist.append((t, float(p[0]), float(p[1])))
            tr.meas.append((0.02, 0.015, 0.3))
        ks, vs, cvs = tk._windows(tr)
        self.assertEqual(list(ks), list(range(3, 8)))
        # direct weighted least squares over all 7 samples
        ts = np.array([h[0] for h in tr.hist])
        P = np.array([[h[1], h[2]] for h in tr.hist])
        c, s = math.cos(0.3), math.sin(0.3)
        R = np.array([[c, -s], [s, c]])
        W = R @ np.diag([1 / 0.02 ** 2, 1 / 0.015 ** 2]) @ R.T
        tau = ts - ts[-1]
        A, b = np.zeros((4, 4)), np.zeros(4)
        for k in range(7):
            H = np.hstack([np.eye(2), tau[k] * np.eye(2)])
            A += H.T @ W @ H
            b += H.T @ W @ P[k]
        theta = np.linalg.solve(A, b)
        self.assertTrue(np.allclose(vs[-1], theta[2:], atol=1e-9))
        self.assertTrue(np.allclose(cvs[-1], np.linalg.inv(A)[2:, 2:], atol=1e-12))


class MovingTest(unittest.TestCase):
    """The moving test on twin scans at 20 mm range noise (sim/tracking_eval.py)."""

    def test_static_furniture_is_never_flagged_at_20mm_noise(self):
        from sim.tracking_eval import static_run
        for vc, k, seed in ((0.0, 0.0, 0), (0.4, 0.0, 1), (0.4, 0.5, 2), (0.6, 0.0, 4)):
            raw, assist, total, ever = static_run(lambda: Tracker(), vc, k, 0.02, seed, seconds=6.0)
            self.assertGreater(total, 300)
            self.assertEqual(raw, 0, f"static object flagged moving (car {vc} m/s, arc {k}, seed {seed})")
            self.assertEqual(len(ever), 0)

    def test_a_walker_is_flagged_within_a_second_and_faster_when_faster(self):
        from sim.tracking_eval import crossing_run
        lat = {}
        for speed in (0.5, 1.0):
            (l_assist, l_flag), _, _, _ = crossing_run(lambda: Tracker(), speed, 0.0, 0.0, 0.02, 0, 1.0)
            self.assertIsNotNone(l_flag)
            lat[speed] = l_flag
        self.assertLess(lat[0.5], 1.0)
        self.assertLess(lat[1.0], 0.7)
        self.assertLessEqual(lat[1.0], lat[0.5])

    def test_a_walker_is_flagged_while_the_car_turns_with_a_yaw_rate_error(self):
        from sim.tracking_eval import crossing_run
        (_, l_flag), flagged, total, _ = crossing_run(lambda: Tracker(), 0.5, 0.4, 0.5, 0.02, 0, 1.0)   # seed 0: +0.05 rad/s bias
        self.assertIsNotNone(l_flag)
        self.assertLess(l_flag, 1.5)

    def test_the_legacy_estimator_is_still_available(self):
        tk = Tracker(model="legacy")
        self.assertAlmostEqual(tk.gate, 0.30)
        self.assertEqual(tk.update(arc_points(1.5, 0.3, 0.08), 0.0, 0.0, 0.0)[0].moving, False)


if __name__ == "__main__":
    unittest.main()
