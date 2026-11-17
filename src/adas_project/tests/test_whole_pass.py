"""Regression tests of the whole-project pass (TODO AR, 14-17 Nov 2026): the smooth brake approach, a start beside a noisy wall, a
pull-up pose the route planner can reach, click-to-go stopping on its goal."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class SmoothBrakeApproachTest(unittest.TestCase):
    def test_full_throttle_at_a_wall_is_one_smooth_stop(self):
        """Throttle held at a wall: the cap leads the limit, so no brake pulse is needed and the car does not stop, go and stop again
        (before: brake at ~33 cm, re-acceleration, brake at ~12 cm). It still ends a few cm short without contact."""
        from sim.stopping_eval import one
        for pwm in (150, 255):
            r = one(pwm, 0)
            self.assertFalse(r["collided"], r)
            self.assertEqual(r["brake_pulses"], 0, r)
            self.assertLessEqual(r["stops"], 2, r)
            self.assertGreater(r["min_gap_cm"], 2.5, r)
            self.assertLess(r["gap_cm"], 10.0, r)


class StartBesideAWallTest(unittest.TestCase):
    def test_a_first_move_exists_whatever_the_scan_noise(self):
        """Car 5 cm off a wall along its side, the wall's returns scattered by a centimetre: every noise draw must still give a path
        (with 5 mm of escape slack some draws had no first move: 'no way through' after one node)."""
        from adas.plan_service import plan_point_job
        from adas.vehicle_params import VehicleParams
        vp = VehicleParams()
        xs = np.linspace(-1.0, 3.0, 400)
        fails = 0
        for seed in range(12):
            rng = np.random.default_rng(seed)
            wall = np.column_stack([xs, -(vp.width / 2 + 0.05) + rng.normal(0, 0.008, len(xs))])
            far = np.column_stack([xs, np.full(len(xs), 1.2)])
            path = plan_point_job(vp, 2.0, np.vstack([wall, far]), (0.0, 0.0, 0.0), (1.8, 0.5), budget_s=2.0, goal_heading=0.0)
            fails += path is None
        self.assertEqual(fails, 0)


class PullUpPoseTest(unittest.TestCase):
    def test_pull_up_keeps_room_for_the_approach_beside_a_wall_at_the_lane_end(self):
        """A block across the lane's end reaches the road edge (the home bay): the pull-up pose stands clear of it by APPROACH_CLEAR_M,
        not 10 cm beside its corner."""
        from adas.park_select import APPROACH_CLEAR_M, ParkingStrip, _obb_clearance, plan_parallel
        from adas.vehicle_params import VehicleParams
        car = VehicleParams()
        depth = 0.45
        strip = ParkingStrip([(0.0, 0.0), (4.0, 0.0)], 1, depth)
        blocks = [(1.0, 1.3), (2.2, 2.5)]                                    # full-depth blocks, flush with the road edge; gap 0.9 m
        pts = []
        for x0, x1 in blocks:
            pts += [(x, 0.0) for x in np.linspace(x0, x1, 16)] + [(x0, y) for y in np.linspace(0, depth, 20)] + [(x1, y) for y in np.linspace(0, depth, 20)]
        pts += [(x, depth) for x in np.linspace(0.0, 4.0, 160)]
        pts = np.array(pts, float)
        plan = plan_parallel(strip, (1.75, 0.2), car, points=pts, approach=False)
        self.assertIsNotNone(plan.planned_path)
        clear = float(_obb_clearance(np.array([plan.pull_up], float), pts, car)[0])
        self.assertGreaterEqual(clear, APPROACH_CLEAR_M - 1e-6)


class GotoArrivalTest(unittest.TestCase):
    def test_click_to_go_stops_on_the_goal(self):
        from sim.autonav_eval import SUITE, run_scenario
        case = [c for c in SUITE if c[0] == "straight_open"][0]
        out = run_scenario(case, 0)
        self.assertTrue(out["ok"], out)
        self.assertLess(out["err_cm"], 3.5, out)                              # (was 4-6 cm: it stopped as soon as it was within 8 cm)


class HeldByTheGateReplanTest(unittest.TestCase):
    def test_goal_on_a_box_ends_beside_it_not_stuck(self):
        """Goal clicked on a box: the car stops in front of it and re-plans. With 2 cm of escape slack that re-plan went 2 cm nearer
        the box - the path the brake gate had just refused - six times, then 'stuck' (seed 1). A re-plan after a gate hold is strict
        now (plan_point_job strict_escape)."""
        from sim.autonav_eval import SUITE, run_scenario
        case = [c for c in SUITE if c[0] == "goal_in_box"][0]
        out = run_scenario(case, 1)
        self.assertTrue(out["ok"], out)
        self.assertTrue(out["msg"].startswith("arrived"), out)

    def test_strict_escape_falls_back_to_the_normal_slack(self):
        """Beside a noisy wall the strict search can have no first move: the job then still returns the normal-slack path."""
        from adas.plan_service import plan_point_job
        from adas.vehicle_params import VehicleParams
        vp = VehicleParams()
        xs = np.linspace(-1.0, 3.0, 400)
        for seed in range(6):
            rng = np.random.default_rng(seed)
            wall = np.column_stack([xs, -(vp.width / 2 + 0.05) + rng.normal(0, 0.008, len(xs))])
            far = np.column_stack([xs, np.full(len(xs), 1.2)])
            path = plan_point_job(vp, 2.0, np.vstack([wall, far]), (0.0, 0.0, 0.0), (1.8, 0.5), budget_s=2.0, goal_heading=0.0,
                                  strict_escape=True)
            self.assertIsNotNone(path, seed)


class BayStopTest(unittest.TestCase):
    def test_nose_in_bay_ends_at_the_wall_not_short_of_it(self):
        """Nose-in bay parking on the twin's relay loop: the car stops a few cm from the back wall (goal 3 cm), straight and centred.
        Before 17 Nov it ended 8 cm off: the twin's loop never gave the gate the bay crawl limit (the car's relay does), and the
        follower's arrival radius ended the run up to 3 cm short, once in the middle of a straightening stroke."""
        import sim.parking_validation as pv
        case = [c for c in pv.all_cases() if c[0] == "front_nominal"][0]
        row = pv.run_case((case[0], case[1], case[2], 0, False))
        self.assertEqual(row["failure"], "", row)
        self.assertEqual(int(row["collisions"]), 0, row)
        self.assertLess(row["long_err_cm"], 3.5, row)                         # front bumper under 6.5 cm from the wall, never past the 3 cm goal by much
        self.assertLess(row["lat_err_cm"], 1.5, row)
        self.assertLess(row["head_err_deg"], 3.0, row)

    def test_creep_held_by_the_gate_ends_the_run(self):
        """The final creep that makes no progress with the throttle held is over after CREEP_STALL_S (it used to 'centre in the slot' for
        30 s); a released throttle does not count."""
        from adas.autonav import AutoNav
        nav = AutoNav.__new__(AutoNav)
        nav.x, nav.y = 1.0, 0.0
        nav._creep_ref, nav._creep_still_s = None, 0.0
        self.assertFalse(nav._creep_stalled(0.05, True))
        for _ in range(40):                                                    # 2 s without the throttle: paused
            self.assertFalse(nav._creep_stalled(0.05, False))
        stalled = [nav._creep_stalled(0.05, True) for _ in range(30)]          # 1.5 s held and still
        self.assertTrue(stalled[-1])
        self.assertFalse(stalled[10])
        nav.x = 1.02                                                           # it moved: the timer starts again
        self.assertFalse(nav._creep_stalled(0.05, True))


if __name__ == "__main__":
    unittest.main()
