"""Parking validation framework (sim/parking_validation.py): scenario coverage, metrics on two real drives, summary."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sim import parking_validation as pv


def case(name):
    for n, g, p in pv.all_cases():
        if n == name:
            return n, g, p
    raise KeyError(name)


class CoverageTest(unittest.TestCase):
    def test_required_scenarios_exist(self):
        names = {n for n, _, _ in pv.all_cases()}
        for req in ("perp_perfect", "perp_rot5", "perp_rot10", "perp_rot20", "perp_drop50", "perp_occluder_full", "perp_dist1.2",
                    "perp_dist3.0", "perp_lidar2cm", "perp_drift_strong", "par_tight", "par_standard", "par_wide", "par_too_short",
                    "par_all_full", "par_skip_short_gap", "par_long_click_end", "par_edge_first", "par_edge_last", "par_dead_end",
                    "par_click_car", "par_shallow_lane"):
            self.assertIn(req, names)
        streets = {n[:-1] for n in names if n.startswith("par_street")}
        self.assertGreaterEqual(len(streets), 20)

    def test_streets_are_deterministic_and_varied(self):
        """Continuous lanes (user, 30 Sep: no spaces): cars and gaps of many different lengths, every street has a viable gap."""
        self.assertEqual(pv.street_spec(3), pv.street_spec(3))
        gaps = {round(ln, 2) for i in range(24) for k, ln in pv.street_spec(i) if k == "gap"}
        self.assertGreater(len(gaps), 60)
        for i in range(24):
            self.assertTrue(any(k == "gap" and ln >= 1.0 for k, ln in pv.street_spec(i)[1:-1]))

    def test_parallel_world_is_a_continuous_lane(self):
        _, _, p = case("par_standard")
        world, start, strip, cars = pv.build_parallel(p)
        self.assertEqual(len(cars), 2)                                               # two parked cars, no spaces
        gaps = pv._true_gaps(strip, cars)
        self.assertAlmostEqual(gaps[1][1] - gaps[1][0], 1.2, delta=0.03)             # the 1.2 m gap between them (skew adds a little)
        np.testing.assert_allclose(world.layout["target"], (0.7, 1.9), atol=1e-9)   # the named case clicks into it


class DriveTest(unittest.TestCase):
    def test_perfect_marker_parks_within_tolerance(self):
        r = pv.run_case(case("perp_perfect") + (0, False))
        self.assertEqual(r["success"], 1, r["failure"])
        self.assertEqual(r["collisions"], 0)
        self.assertLessEqual(r["pos_err_cm"], 10)
        self.assertLessEqual(r["head_err_deg"], 10)
        self.assertGreater(r["path_m"], 1.0)

    def test_hidden_marker_is_refused(self):
        r = pv.run_case(case("perp_occluder_full") + (0, False))
        self.assertEqual(r["success"], 1)
        self.assertEqual(r["planned"], 0)
        self.assertEqual(r["path_m"], 0.0)

    def test_too_short_gap_is_refused_and_car_stays(self):
        r = pv.run_case(case("par_too_short") + (0, False))
        self.assertEqual(r["success"], 1)
        self.assertEqual(r["planned"], 0)

    def test_summary_counts(self):
        rows = [dict(name="a", group="parallel", mc=1, success=1, collisions=0, time_s=5.0, expect="park", failure="", pos_err_cm=2.0,
                     head_err_deg=1.0, corrections=0, path_m=3.0, selection_ok=1),
                dict(name="a", group="parallel", mc=1, success=0, collisions=0, time_s=None, expect="park", failure="no plan: x",
                     pos_err_cm=None, head_err_deg=None, corrections=None, path_m=0.0, selection_ok=0)]
        s = pv.summarise(rows)[0]
        self.assertEqual((s["runs"], s["success"], s["parked"]), (2, 1, 1))
        self.assertEqual(s["failures"], {"no plan": 1})


if __name__ == "__main__":
    unittest.main()
