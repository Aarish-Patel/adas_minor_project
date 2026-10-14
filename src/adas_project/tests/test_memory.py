"""Obstacle memory for the LiDAR's blind ring (adas/memory.py, TODO AB4): what the scan can no longer see is remembered, moved with the
car, dropped when the scan proves it gone or after the car has driven on, and kept on the scan-matched pose."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.memory import ObstacleMemory
from adas.vehicle_params import VehicleParams

P = VehicleParams(wheelbase=0.20, width=0.20, front_overhang=0.08, rear_overhang=0.05, lidar_x=0.12)


def box_face(x, y0, y1, n=12):
    """Returns of an object's face at x (vehicle frame), from y0 to y1."""
    return np.column_stack([np.full(n, x), np.linspace(y0, y1, n)])


def memory():
    return ObstacleMemory(P, blind_radius=0.27, keep_radius=1.2, max_age=60.0, max_points=700, max_travel=1.5)


def drive(m, dist, v=0.3, dt=0.05, delta=0.0):
    for _ in range(int(round(dist / abs(v * dt)))):
        m.advance(dt, v, delta)


class MemoryTest(unittest.TestCase):
    def test_an_object_in_the_blind_ring_is_remembered_where_it_is(self):
        m = memory()
        m.add_scan(box_face(0.60, 0.05, 0.20))                    # seen 0.48 m ahead of the LiDAR
        drive(m, 0.40)                                            # now 8 cm in front of the LiDAR: inside its minimum range
        m.add_scan(np.empty((0, 2)))                              # the scan has nothing there any more
        blind = m.blind_points()
        self.assertGreater(len(blind), 5)
        self.assertAlmostEqual(float(blind[:, 0].mean()), 0.20, delta=0.01)   # 0.60 - 0.40, in the car's frame now

    def test_the_scan_overrules_the_memory(self):
        m = memory()
        m.add_scan(box_face(0.30, -0.02, 0.02))
        drive(m, 0.10)                                            # inside the ring, but the scan sees FURTHER along those bearings
        m.prune_contradicted(box_face(1.50, -0.45, 0.45, n=60))
        self.assertEqual(len(m.blind_points()), 0)

    def test_a_bearing_with_no_return_keeps_its_points(self):
        """No return at all is what a truly close object looks like (inside the minimum range): the memory keeps it."""
        m = memory()
        m.add_scan(box_face(0.30, -0.02, 0.02))
        drive(m, 0.10)
        m.prune_contradicted(box_face(1.50, 0.8, 1.2))           # returns elsewhere only
        self.assertGreater(len(m.blind_points()), 0)

    def test_forgotten_after_the_car_has_driven_on(self):
        m = memory()
        m.add_scan(box_face(0.35, -0.05, 0.05))
        drive(m, 1.6, delta=0.3)                                  # max_travel 1.5 m
        m.add_scan(box_face(3.0, -1.0, 1.0))
        self.assertEqual(len(m.pts[np.hypot(m.pts[:, 0] - (0.35), m.pts[:, 1]) < 0.1]), 0)

    def test_the_scan_matched_pose_removes_dead_reckoning_error(self):
        """The speed estimate is 10 % high: dead reckoning alone puts the remembered object 4 cm off after 0.4 m; with the scan-matched
        pose it is where it is."""
        a, b = memory(), memory()
        for m in (a, b):
            m.add_scan(box_face(0.60, 0.05, 0.20))
            drive(m, 0.44, v=0.33)                                # believed 0.44 m, truly 0.40 m
        b.correct_pose((0.40, 0.0, 0.0), stamp=1)
        err_dr = abs(float(a.blind_points()[:, 0].mean()) - 0.20)
        err_sm = abs(float(b.blind_points()[:, 0].mean()) - 0.20)
        self.assertGreater(err_dr, 0.035)
        self.assertLess(err_sm, 0.005)

    def test_a_stale_pose_is_ignored(self):
        m = memory()
        m.add_scan(box_face(0.60, 0.05, 0.20))
        m.correct_pose((0.0, 0.0, 0.0), stamp=7)
        drive(m, 0.40)
        m.correct_pose((0.0, 0.0, 0.0), stamp=7)                 # no new scan matched: must not pull the car back
        self.assertAlmostEqual(m.x, 0.40, delta=0.01)

    def test_origin_reset_keeps_the_objects_around_the_car(self):
        m = memory()
        m.add_scan(box_face(0.60, 0.05, 0.20))
        drive(m, 0.40)
        before = m.blind_points().copy()
        m.correct_pose((0.0, 0.0, 0.0), stamp=2)                 # the world frame restarted at the car (ORIGIN): a jump, not motion
        np.testing.assert_allclose(m.blind_points(), before, atol=1e-9)

    def test_moving_objects_are_not_remembered(self):
        m = memory()
        m.add_scan(box_face(0.40, -0.05, 0.05), exclude=[(0.40, 0.0, 0.08)])
        self.assertEqual(len(m.pts), 0)


def a3_scan(objects, step_deg=0.33, lidar_x=0.12):
    """An A3-like scan (one return per step_deg) of flat objects given as (x, y0, y1) faces in the vehicle frame; the nearest face on
    each bearing returns."""
    out = []
    for b in np.arange(-179.9, 180.0, step_deg):
        c, s = math.cos(math.radians(b)), math.sin(math.radians(b))
        best = None
        for x, y0, y1 in objects:
            if c <= 1e-6:
                continue
            t = (x - lidar_x) / c                     # ray: (lidar_x + t c, t s)
            y = t * s
            if t > 0.2 and min(y0, y1) <= y <= max(y0, y1) and (best is None or t < best):
                best = t
        if best is not None:
            out.append((lidar_x + best * c, best * s))
    return np.array(out)


class ThinObjectTest(unittest.TestCase):
    """User, 14 Oct: thin objects were remembered close up and deleted once the car moved away (not seen from further)."""
    def setUp(self):
        self.wall = (2.6, -1.5, 1.5)

    def test_a_thin_post_stays_when_the_car_backs_away(self):
        m = memory()
        post = (0.55, -0.01, 0.01)                                   # a 2 cm post 0.43 m ahead of the LiDAR
        for _ in range(6):
            sc = a3_scan([post, self.wall])
            m.prune_contradicted(sc); m.add_scan(sc)
        drive(m, 0.6, v=-0.3)                                        # backs away: the post is now ~1.0 m from the LiDAR
        for _ in range(40):                                          # 3 s of scans that miss the post (wall seen behind it)
            sc = a3_scan([self.wall])
            m.prune_contradicted(sc); m.add_scan(sc)
        v = m._to_vehicle(m.pts)
        self.assertTrue(np.any(np.hypot(v[:, 0] - 1.15, v[:, 1]) < 0.04))

    def test_a_removed_box_goes(self):
        m = memory()
        box = (0.75, -0.10, 0.10)                                    # a 20 cm box 0.63 m ahead
        for _ in range(6):
            sc = a3_scan([box, self.wall])
            m.prune_contradicted(sc); m.add_scan(sc)
        for _ in range(15):                                          # the box is taken away: the wall is seen through its place
            sc = a3_scan([self.wall])
            m.prune_contradicted(sc); m.add_scan(sc)
        v = m._to_vehicle(m.pts)
        self.assertFalse(np.any(np.hypot(v[:, 0] - 0.75, v[:, 1]) < 0.12))


if __name__ == "__main__":
    unittest.main()
