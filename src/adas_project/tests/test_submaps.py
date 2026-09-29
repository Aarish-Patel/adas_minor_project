"""Submaps, loop closure and return to start (adas/submaps.py, sim/slam_eval.py, sim/home_eval.py)."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.submaps import PoseGraphMap, between, compose, inverse


class PoseAlgebra(unittest.TestCase):
    def test_compose_inverse_between(self):
        a, b = np.array([1.0, 2.0, 0.7]), np.array([-0.5, 0.3, -1.1])
        self.assertTrue(np.allclose(compose(a, between(a, b)), b))
        self.assertTrue(np.allclose(compose(a, inverse(a)), [0, 0, 0], atol=1e-12))


class LoopClosure(unittest.TestCase):
    def test_a_drifting_loop_is_pulled_back_onto_itself(self):
        from sim.slam_eval import run
        res = run(verbose=False)[0]
        self.assertGreaterEqual(res["loops"], 3)
        self.assertLess(res["final_err_corrected_cm"], 2.5)
        self.assertLess(res["final_err_corrected_cm"], res["final_err_odometry_cm"])
        self.assertLess(res["final_heading_err_corrected_deg"], res["final_heading_err_odometry_deg"])

    def test_without_a_return_nothing_is_changed(self):
        # a straight corridor drive never revisits a submap: the graph must leave the pose alone
        g = PoseGraphMap()
        rng = np.random.default_rng(0)
        pts = np.column_stack([np.linspace(0.3, 3, 200), 0.5 + 0 * np.linspace(0, 1, 200)])
        pts = np.vstack([pts, pts * [1, -1]])
        pose = None
        for k in range(60):
            pose = g.add_scan(np.array([k * 0.05, 0.0, 0.0]), pts)
        self.assertEqual(len(g.loops), 0)
        self.assertTrue(np.allclose(pose, [59 * 0.05, 0.0, 0.0], atol=1e-6))


class ReturnToStart(unittest.TestCase):
    def test_comes_back_to_the_start_with_loop_closure(self):
        from sim.home_eval import drive_out_and_home
        # a bare room: the scan-to-scan front end alone loses the start (1 m off in the twin); loop closure brings it back
        r = drive_out_and_home("open", slam=True)
        self.assertFalse(r["crashed"])
        self.assertLess(r["err_cm"], 30.0)                      # (mean 11 cm, worst 21 cm over runs; the front end alone: ~110 cm)
        self.assertLess(r["heading_err_deg"], 15.0)
        self.assertGreater(r["farthest_m"], 1.0)
        self.assertGreater(r["loops"], 3)


if __name__ == "__main__":
    unittest.main()
