"""Parking planning (adas/park_select.py): gaps on a continuous parking lane, nearest viable slot to a click, reverse S and multi-stroke
parallel parking, re-planning alongside the gap, the ArUco bay."""
import math
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.markers import MarkerObs, marker_corners_3d, pose_from_corners, project
from adas.park_select import (KERB_CLEAR_M, KERB_FAR_M, KERB_GAP_M, KERB_MIN_M, ParkingStrip, find_gaps, fuse_markers, min_space_length, plan_parallel,
                              plan_perpendicular, rear_camera, refine_parallel)
from adas.vehicle_params import VehicleParams

CAR = VehicleParams(wheelbase=0.20, width=0.20, front_overhang=0.08, rear_overhang=0.05)   # the laptop car: 0.33 x 0.20 m
DEPTH = 0.38


def outline(x0, length, y0=0.16, w=0.20, n=24):
    """LiDAR-like returns of a parked car: its road-side face, its two end faces and its kerb side (lane frame, x along the lane)."""
    pts = [(x0 + t * length, y0) for t in np.linspace(0, 1, n)] + [(x0 + t * length, y0 + w) for t in np.linspace(0, 1, n)]
    pts += [(x0, y0 + t * w) for t in np.linspace(0, 1, 8)] + [(x0 + length, y0 + t * w) for t in np.linspace(0, 1, 8)]
    return pts


def street(cars, side=1, rot=0.0, lane_y=0.0, y0=0.16):
    """A straight lane (road edge along x from -2 to 10, rotated by `rot`) with parked cars [(x0, length)], their road-side faces `y0`
    from the road edge (0.16: 2 cm off the kerb): strip, points, car blocks (x0, x1, y0, y1) in the lane frame."""
    c, s = math.cos(rot), math.sin(rot)
    tf = lambda p: (c * p[0] - s * side * p[1], s * p[0] + c * side * p[1] + lane_y)
    strip = ParkingStrip([tf((-2.0, 0.0)), tf((10.0, 0.0))], side, DEPTH)
    pts = np.array([tf(q) for x0, ln in cars for q in outline(x0, ln, y0=y0)])
    blocks = [(x0, x0 + ln, y0, y0 + 0.20) for x0, ln in cars]
    return strip, pts, blocks


def to_lane(strip, path):
    """Path poses (N, 3+) of the working frame -> lane frame (x along, y toward the kerb, heading relative to the lane)."""
    s, y = strip.to_strip(path[:, :2])
    return np.column_stack([s - 2.0, y, [math.remainder(a - strip.heading[0], 2 * math.pi) * strip.side for a in path[:, 2]]])


def slot_x(strip, plan):
    """The chosen slot's centre, along the lane (lane frame x)."""
    return float(strip.to_strip(plan.slot.mean(axis=0)[None])[0][0] - 2.0)


def body_clearance(path, blocks, car=CAR, kerb=False):
    """Independent check: smallest distance of body outline samples to the car blocks (kerb=False) or to the kerb (kerb=True), lane
    frame, over the path."""
    hw = car.width / 2
    best = math.inf
    for x, y, th in path[:, :3]:
        c, s = math.cos(th), math.sin(th)
        for lx in np.linspace(car.rear_x, car.front_x, 12):
            for ly in (-hw, hw):
                px, py = x + c * lx - s * ly, y + s * lx + c * ly
                if kerb:
                    best = min(best, DEPTH - py)
                    continue
                for x0, x1, y0, y1 in blocks:
                    dx, dy = max(x0 - px, 0.0, px - x1), max(y0 - py, 0.0, py - y1)
                    best = min(best, math.hypot(dx, dy) if dx or dy else -min(px - x0, x1 - px, py - y0, y1 - py))
    return best


class GapTest(unittest.TestCase):
    def test_gaps_between_the_parked_cars(self):
        strip, pts, _ = street([(1.0, 0.4), (2.4, 0.4), (3.5, 0.4)])
        gaps = [(round(a - 2.0, 2), round(b - 2.0, 2)) for a, b in find_gaps(strip, pts)]
        self.assertEqual(gaps[1:3], [(1.4, 2.4), (2.8, 3.5)])                    # between the cars, from their end faces
        self.assertEqual(gaps[0][0], -2.0)                                       # the lane before the first car is free too

    def test_minimum_gap(self):
        self.assertAlmostEqual(min_space_length(CAR), 0.63, places=2)            # car length + 0.30 m


class SelectionTest(unittest.TestCase):
    def test_click_on_a_car_takes_the_nearest_viable_gap(self):
        strip, pts, blocks = street([(0.0, 0.4), (1.5, 0.4), (2.4, 0.4), (4.0, 0.4)])   # gaps 1.1, 0.5 (too short), 1.2
        plan = plan_parallel(strip, (2.6 - 2.0 + 2.0, 0.3), CAR, pose=(-1.0, -0.4, 0.0), points=pts)   # click on the 3rd car
        cx = slot_x(strip, plan)
        self.assertTrue(2.8 <= cx <= 4.0, cx)                                    # 0.2 m to the gap after it, 0.5 m to the one before

    def test_click_in_a_short_gap_skips_it(self):
        strip, pts, _ = street([(0.0, 0.4), (1.5, 0.4), (2.4, 0.4), (3.6, 0.4)])        # gaps 1.1, 0.5, 0.8
        plan = plan_parallel(strip, (2.15, 0.3), CAR, pose=(-1.0, -0.4, 0.0), points=pts)
        cx = slot_x(strip, plan)
        self.assertTrue(2.8 <= cx <= 3.6, cx)                                    # the 0.8 m gap is nearer than the 1.1 m one

    def test_long_gap_parks_where_clicked(self):
        strip, pts, _ = street([(0.0, 0.4), (3.0, 0.4)])                           # a 2.6 m gap
        plan = plan_parallel(strip, (2.4, 0.3), CAR, pose=(-1.0, -0.4, 0.0), points=pts)   # click 0.4 m before the front car
        cx = slot_x(strip, plan)
        self.assertAlmostEqual(cx, 3.0 - 0.8, delta=0.02)                        # the slot (1.6 m) as near to the click as fits

    def test_long_gap_offers_more_spots_nearest_first(self):
        from adas.park_select import SLOT_STEP_M, choose_slots
        slots = choose_slots([(2.4, 5.4)], 3.9, CAR, DEPTH)                              # a 3 m gap, clicked in its middle
        mids = [sp.mid for sp in slots]
        self.assertAlmostEqual(mids[0], 3.9)
        self.assertEqual(sorted(mids, key=lambda m: abs(m - 3.9)), mids)                 # nearest to the click first
        self.assertAlmostEqual(abs(mids[1] - mids[0]), SLOT_STEP_M)
        self.assertTrue(all(2.4 <= sp.s0 and sp.s1 <= 5.4 for sp in slots))              # all inside the gap

    def test_spot_with_no_way_in_takes_the_next_along_the_gap(self):
        """The spot at the click has no manoeuvre (e.g. something beside it on the road): the nearest other spot along the same gap is
        taken - never a refusal while one can be parked."""
        import adas.park_select as PS
        strip, pts, _ = street([(0.0, 0.4), (3.4, 0.4)])
        click = tuple(strip.pts[0] + (2.0 + 1.9) * (strip.pts[1] - strip.pts[0]) / 12.0)
        orig = PS._plan_space
        PS._plan_space = lambda strip_, space, *a, **k: None if abs(space.mid - 3.9) < 0.3 else orig(strip_, space, *a, **k)
        try:
            plan = plan_parallel(strip, click, CAR, points=pts, approach=False)
        finally:
            PS._plan_space = orig
        self.assertIsNotNone(plan.planned_path)
        self.assertAlmostEqual(abs(slot_x(strip, plan) - 1.9), 0.5, delta=0.01)          # 0.5 m along (0.25 was ruled out too)

    def test_no_viable_gap_gives_no_plan(self):
        strip, pts, _ = street([(x, 0.4) for x in np.arange(-2.0, 10.0, 0.9)])  # 0.5 m gaps everywhere
        plan = plan_parallel(strip, (3.0, 0.3), CAR, points=pts)
        self.assertIsNone(plan.planned_path)
        self.assertIsNone(plan.parking_goal)

    def test_shallow_lane_is_refused(self):
        strip, pts, _ = street([(0.0, 0.4), (1.5, 0.4)])
        shallow = ParkingStrip(strip.pts, 1, CAR.width + 0.06)
        self.assertIsNone(plan_parallel(shallow, (1.0, 0.2), CAR, points=pts).planned_path)


class ParallelPlanTest(unittest.TestCase):
    face = 0.16                                                            # the neighbours' road-side faces (outline y0)

    def park(self, gap, start=(-1.0, -0.4, 0.0), **kw):
        strip, pts, blocks = street([(0.0, 0.4), (0.4 + gap, 0.4)], y0=self.face, **kw)
        t = time.perf_counter()
        click = strip.pts[0] + (2.0 + 0.4 + gap / 2) * (strip.pts[1] - strip.pts[0]) / 12.0
        if kw.get("side", 1) < 0 or kw.get("rot"):
            click = click + 0.1 * np.array([-math.sin(strip.heading[0]), math.cos(strip.heading[0])]) * strip.side
        plan = plan_parallel(strip, tuple(click), CAR, pose=start, points=pts)
        dt = time.perf_counter() - t
        self.assertIsNotNone(plan.planned_path, gap)
        return strip, blocks, plan, dt

    def check(self, strip, blocks, plan, gap):
        P = to_lane(strip, plan.planned_path)
        g = to_lane(strip, np.array([[*plan.parking_goal[:2], math.radians(plan.parking_goal[2])]]))[0]
        self.assertLess(np.hypot(*(P[-1, :2] - g[:2])), 0.01)                   # the path ends on the goal
        self.assertLess(abs(P[-1, 2]), math.radians(1.0))                        # parallel to the lane
        centre = g[0] + 0.5 * (CAR.front_x + CAR.rear_x)
        self.assertAlmostEqual(centre, 0.4 + gap / 2, delta=0.02)                # centred between the two cars
        kerb_gap = DEPTH - (g[1] + CAR.width / 2)
        # in line with the neighbours (their road-side faces at y 0.16: 2 cm off the kerb), but never nearer the kerb than KERB_MIN_M
        # nor further out than KERB_FAR_M
        self.assertAlmostEqual(kerb_gap, float(np.clip(DEPTH - (self.face + CAR.width), KERB_MIN_M, KERB_FAR_M)), delta=0.01)
        self.assertGreater(body_clearance(P, blocks), 0.03)                      # never closer than 3 cm to a car
        self.assertGreater(body_clearance(P, blocks, kerb=True), min(KERB_CLEAR_M, kerb_gap - 0.005) - 0.002)   # nor to the kerb than the goal
        return int((np.diff(plan.planned_path[:, 3]) != 0).sum())

    def test_roomy_gap_single_reverse_s(self):
        strip, blocks, plan, dt = self.park(1.6)
        self.assertLessEqual(self.check(strip, blocks, plan, 1.6), 1)            # pull up alongside, one reverse move

    def test_normal_gap(self):
        strip, blocks, plan, dt = self.park(1.1)
        self.assertLessEqual(self.check(strip, blocks, plan, 1.1), 2)

    def test_tight_gap_several_strokes_fast(self):
        strip, blocks, plan, dt = self.park(0.68)
        self.assertGreaterEqual(self.check(strip, blocks, plan, 0.68), 3)       # in, forward, back ... as a driver in a tiny spot
        self.assertLess(dt, 0.5)                                                  # deterministic exit plan: milliseconds, not a search

    def test_from_ahead_of_the_gap(self):
        strip, blocks, plan, dt = self.park(1.1, start=(3.5, -0.35, 0.0))         # the car is already past it: reverses back
        self.check(strip, blocks, plan, 1.1)

    def test_neighbours_standing_further_out(self):
        """The neighbours stand 5 / 8 cm off the kerb: the car parks in line with them, but never further out than KERB_FAR_M."""
        self.face = 0.13
        strip, blocks, plan, dt = self.park(1.3)
        self.check(strip, blocks, plan, 1.3)
        self.face = 0.10
        strip, blocks, plan, dt = self.park(1.3)
        self.check(strip, blocks, plan, 1.3)

    def test_kerb_on_the_right_and_a_rotated_lane(self):
        strip, blocks, plan, dt = self.park(1.2, side=-1, rot=math.radians(30), start=(-1.0, 0.4, math.radians(30)))
        self.check(strip, blocks, plan, 1.2)


class PhasesTest(unittest.TestCase):
    """Parking in phases (ADR-038): the slot is chosen independently of where the car stands, the manoeuvre starts at its pull-up pose, a
    parked car pulls out first."""

    def test_slot_independent_of_the_car_and_starts_at_the_pull_up_pose(self):
        strip, pts, _ = street([(0.0, 0.4), (1.5, 0.4), (3.0, 0.4)])                   # gaps 1.1 and 1.1
        click = tuple(strip.pts[0] + (2.0 + 2.1) * (strip.pts[1] - strip.pts[0]) / 12.0)
        slots = []
        for pose in ((-1.5, -0.5, 0.0), (5.0, -0.5, 0.0), (3.0, -0.6, math.pi)):          # behind, past it, facing the other way
            plan = plan_parallel(strip, click, CAR, pose=pose, points=pts, approach=False)
            self.assertIsNotNone(plan.planned_path)
            self.assertFalse(plan.starts_at_car)
            self.assertLess(math.hypot(*(plan.planned_path[0, :2] - np.array(plan.pull_up[:2]))), 0.03)   # it starts at the pull-up
            slots.append(round(slot_x(strip, plan), 2))
        self.assertEqual(len(set(slots)), 1, slots)                                      # the same slot wherever the car is

    def _parked(self, gap, kerb_gap=0.05):
        """A car parked in a gap of `gap` m between two cars, centred, `kerb_gap` off the kerb. Returns the strip and the scan in the
        parked car's vehicle frame (the working frame of a request made there), the car blocks in the lane frame and the car's rear
        axle (x0, y0) in the lane frame (lane frame = the street() working frame: x along, y from the road edge toward the kerb)."""
        x0 = 0.4 + gap / 2 - 0.5 * (CAR.front_x + CAR.rear_x)
        y0 = DEPTH - kerb_gap - CAR.width / 2
        strip, pts, blocks = street([(0.0, 0.4), (0.4 + gap, 0.4)])
        off = np.array([x0, y0])
        return ParkingStrip(strip.pts - off, 1, DEPTH), pts - off, blocks, off

    def _check_pull_out(self, gap):
        from adas.park_select import plan_pull_out
        strip, pts, blocks, off = self._parked(gap)
        P = plan_pull_out(strip, CAR, pts)
        self.assertIsNotNone(P, gap)
        self.assertLess(np.hypot(*P[0, :2]), 0.03)                                       # from where the car stands
        c = 0.5 * (CAR.front_x + CAR.rear_x)
        end = P[-1:, :2] + c * np.column_stack([np.cos(P[-1:, 2]), np.sin(P[-1:, 2])])
        self.assertLess(strip.to_strip(end)[1][0], 0.0)                                   # the body's centre out on the road
        lane = np.column_stack([P[:, :2] + off, P[:, 2]])
        self.assertGreater(body_clearance(lane, blocks), 0.03)                            # never closer than 3 cm to a car
        self.assertGreater(body_clearance(lane, blocks, kerb=True), KERB_CLEAR_M - 0.002)
        return int((np.diff(P[:, 3]) != 0).sum())

    def test_pull_out_of_a_roomy_gap(self):
        self.assertLessEqual(self._check_pull_out(1.3), 1)

    def test_pull_out_of_a_tight_gap_with_strokes(self):
        self.assertGreaterEqual(self._check_pull_out(0.86), 1)                          # back up first, then out

    def test_side_offset_is_shifted_out_within_the_gap(self):
        """Parked 5 cm too deep in a 1.3 m gap (the S never ends exactly where planned with real steering): the correction is clear of
        both cars and of the kerb, and ends on the goal line (AutoNav._lateral_shift)."""
        from adas.autonav import AutoNav
        strip, pts, blocks, off = self._parked(1.3, kerb_gap=0.02)                        # 2 cm from the kerb instead of 5 (a goal 3 cm left)
        nav = AutoNav(CAR, None)
        nav.goal, nav.goal_heading = (0.0, -0.03), 0.0                                    # the goal: 3 cm toward the road (right)
        nav.park_request = {"strip": (strip.pts, strip.side, strip.depth)}
        P = nav._lateral_shift(-0.03, pts)
        self.assertIsNotNone(P)
        self.assertLess(abs(P[-1, 1] - (-0.03)), 0.01)                                     # on the goal line
        lane = np.column_stack([P[:, :2] + off, P[:, 2]])
        self.assertGreater(body_clearance(lane, blocks), 0.025)
        self.assertGreater(body_clearance(lane, blocks, kerb=True), 0.015)

    def test_not_parked_no_pull_out(self):
        from adas.park_select import plan_pull_out
        strip, pts, _ = street([(0.0, 0.4), (1.5, 0.4)])
        self.assertIsNone(plan_pull_out(strip, CAR, pts))                                 # the car (at the origin) is on the road


class RefineTest(unittest.TestCase):
    """The car has driven up to the pull-up point beside the gap (the working frame is the vehicle there): lane edge 0.3 m to its left,
    rear neighbour at x -1.6 .. -1.2, front neighbour at -0.1 .. 0.3 (true gap 1.1 m)."""
    CARS = [(-1.6, 0.4), (-0.1, 0.4)]

    def test_gap_measured_again_alongside(self):
        """From afar the rear neighbour was half hidden (the gap looked longer); alongside it, the real gap gives the slot."""
        strip, pts, _ = street(self.CARS, lane_y=0.3)
        hidden = np.array([p for p in pts if not (-1.5 < p[0] < -1.2)])           # the rear car's front part was not seen
        planned = plan_parallel(strip, (-0.65, 0.5), CAR, points=hidden)
        s0, _ = strip.to_strip(planned.slot)
        self.assertLess(s0.min() - 2.0, -1.2)                                      # it looked longer than it is
        ref = refine_parallel(strip, planned.request["click"], planned.slot, CAR, points=pts)
        self.assertIsNotNone(ref.planned_path)
        s, _ = strip.to_strip(ref.slot)
        self.assertGreaterEqual(s.min() - 2.0, -1.2 - 1e-6)                        # inside the true gap
        self.assertLessEqual(s.max() - 2.0, -0.1 + 1e-6)


MARKER_M = 0.16                                                        # the printed marker (sim.parking_worlds.MARKER_SIZE)
REAR = rear_camera()                                                   # the planner's rear camera model (pitch 8 deg)


def observe(marker_xy, facing, size=MARKER_M, z=0.09, noise=0.0, rng=None):
    """The rear camera's view of a wall marker (vehicle frame) -> the pose estimated from its noisy corners. `facing` is the direction
    the marker's normal points (toward the car for a wall behind it: 0 with the wall behind an x-forward car)."""
    c = marker_corners_3d(size)                                        # in the marker frame: normal = +z
    n = np.array([math.cos(facing), math.sin(facing), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, n)
    pts = np.array([[marker_xy[0], marker_xy[1], z]]) + c[:, :1] * right + c[:, 1:2] * up
    px, depth = project(REAR, pts)
    if noise:
        px = px + rng.normal(0, noise, px.shape)
    return pose_from_corners(px, 7, size, REAR)


def marker_at(x, y, facing):
    return MarkerObs(id=7, corners=np.zeros((4, 2)), x=x, y=y, z=0.09, facing=facing)


class PerpendicularTest(unittest.TestCase):
    def test_pose_estimate_and_goal_within_5_cm(self):
        rng = np.random.default_rng(0)
        worst = 0.0
        for k in range(60):
            facing = rng.uniform(-0.2, 0.2)                            # the wall behind the car faces the car (+x)
            m = (-rng.uniform(1.2, 2.2), rng.uniform(-0.2, 0.2))
            obs = fuse_markers([observe(m, facing, noise=0.5, rng=rng) for _ in range(5)])
            truth = plan_perpendicular(marker_at(m[0], m[1], facing), CAR)
            est = plan_perpendicular(obs, CAR)
            gt, ge = truth.parking_goal, est.parking_goal
            worst = max(worst, math.hypot(gt[0] - ge[0], gt[1] - ge[1]))
            self.assertIsNotNone(est.planned_path)
        self.assertLess(worst, 0.05)

    def test_path_ends_aligned_with_the_centre_line(self):
        for start in ((0, 0, 0.0), (0.1, 0.2, 0.5), (-0.1, -0.2, -0.4)):
            m = observe((-1.0, 0.1), 0.0)
            plan = plan_perpendicular(m, CAR, pose=start)
            self.assertEqual(plan.selected_space_id, 7)
            path = plan.planned_path
            self.assertIsNotNone(path, start)
            gx, gy, gh = plan.parking_goal
            self.assertLess(math.hypot(path[-1, 0] - gx, path[-1, 1] - gy), 0.01)
            self.assertLess(abs(math.degrees(_wrap(path[-1, 2] - math.radians(gh)))), 1.0)
            self.assertEqual(path[-1, 3], -1)                            # the last move is a reverse
            line = path[-8:, :2]                                          # the tail is straight and on the marker's line
            self.assertLess(np.abs(np.cross(line[-1] - line[0], line[0] - np.array([gx, gy]))).max() / 0.1, 0.01)

    def test_goal_keeps_the_rear_bumper_off_the_wall(self):
        obs = observe((-1.0, 0.0), 0.0)
        gx, gy, gh = plan_perpendicular(obs, CAR, stop_gap=0.06).parking_goal
        bumper_x = gx + CAR.rear_x * math.cos(math.radians(gh))
        self.assertAlmostEqual(bumper_x - obs.x, 0.06, delta=0.005)

    def test_marker_ahead_of_the_car_still_ends_with_the_straight_reverse(self):
        for d in (2.4, 3.0):                                             # the wall is in front of the car and faces it: turn round, reverse in
            plan = plan_perpendicular(marker_at(d, 0.0, math.pi), CAR)
            self.assertIsNotNone(plan.planned_path, d)
            self.assertEqual(plan.planned_path[-1, 3], -1)

    def test_marker_is_seen_from_the_bay_entrance_range(self):
        """The rear camera model + the 16 cm marker: visible at 0.6-2.4 m (the perpendicular bay's start is ~2.5 m from the wall)."""
        for d in (0.6, 1.0, 1.6, 2.2):
            self.assertIsNotNone(observe((-d, 0.0), 0.0), d)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


if __name__ == "__main__":
    unittest.main()
