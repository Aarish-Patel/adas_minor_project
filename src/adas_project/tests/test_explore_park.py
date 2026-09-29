"""Exploration (frontiers) and auto-park (bay detection + reverse-in) - unit tests and end-to-end runs on the twin."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.explore import FREE, OCC, UNKNOWN, OccupancyGrid, frontiers, pick_frontier
from adas.park import find_bays, park_goal


def ring(radius, cx=0.12, n=720):
    a = np.radians(np.linspace(0, 360, n, endpoint=False))
    return np.column_stack([cx + radius * np.cos(a), radius * np.sin(a)])


class GridTest(unittest.TestCase):
    def test_free_inside_occupied_on_the_ring_unknown_outside(self):
        g = OccupancyGrid()
        g.update((0, 0, 0), ring(2.0))
        st = g.state()
        i, j = g.cell(0.5, 0.3)
        self.assertEqual(st[i, j], FREE)
        i, j = g.cell(0.12 + 2.0, 0.0)
        self.assertEqual(st[i - 1:i + 2, j - 1:j + 2].max(), OCC)
        i, j = g.cell(0.0, 3.5)
        self.assertEqual(st[i, j], UNKNOWN)

    def test_frontier_lies_at_the_edge_of_what_is_known(self):
        g = OccupancyGrid()
        pts = ring(3.0)
        pts = pts[(np.arctan2(pts[:, 1], pts[:, 0] - 0.12) < 0) | (pts[:, 1] > 0.0)]     # everything: no gap yet
        gap = ring(3.0)
        gap = gap[np.abs(np.degrees(np.arctan2(gap[:, 1], gap[:, 0] - 0.12))) > 25]       # a 50 deg opening straight ahead
        g.update((0, 0, 0), gap)
        goal = pick_frontier(g, (0, 0, 0))
        self.assertIsNotNone(goal)
        ang = math.degrees(math.atan2(goal[1], goal[0] - 0.12))
        self.assertLess(abs(ang), 35.0)                                                   # on the side of the opening (ahead)
        self.assertGreater(goal[0], 0.3)

    def test_nothing_left_when_a_closed_room_is_seen(self):
        g = OccupancyGrid(max_range=5.0)
        g.update((0, 0, 0), ring(2.5))
        self.assertIsNone(pick_frontier(g, (0, 0, 0)))


class BayTest(unittest.TestCase):
    def test_finds_a_gap_in_a_row_and_none_in_a_solid_wall(self):
        x = np.arange(0.7, 2.0, 0.02)
        row = [(xi, 0.6) for xi in x if not 1.15 < xi < 1.6]
        pts = np.array(row + [(1.0, 0.9)])                                  # a box face, the gap is x 1.15..1.6
        bays = find_bays(pts, side=1)
        self.assertEqual(len(bays), 1)
        self.assertAlmostEqual(bays[0].x, 1.38, delta=0.08)
        wall = np.array([(xi, 0.6) for xi in x])
        self.assertEqual(find_bays(wall, side=1), [])
        narrow = np.array([(xi, 0.6) for xi in x if not 1.3 < xi < 1.46])   # 16 cm: narrower than the car
        self.assertEqual(find_bays(narrow, side=1), [])

    def test_goal_backs_in_with_the_nose_out(self):
        x = np.arange(0.7, 2.0, 0.02)
        bays = find_bays(np.array([(xi, 0.6) for xi in x if not 1.15 < xi < 1.6]), side=1)
        gx, gy, hd = park_goal(bays[0])
        self.assertAlmostEqual(hd, -90.0)                                   # left bay: nose pointing right (out)
        self.assertGreater(gy, 0.6)


class ParallelTest(unittest.TestCase):
    def test_finds_the_gap_in_a_parallel_row_and_ignores_short_or_solid_ones(self):
        from adas.park import find_parallel_slots, park_goal_parallel
        xs = np.arange(0.3, 3.0, 0.02)
        row = np.array([(x, 0.62) for x in xs if not 1.2 < x < 2.2])                       # a 1.0 m gap in a row of parked objects
        slots = find_parallel_slots(row, side=1)
        self.assertEqual(len(slots), 1)
        self.assertAlmostEqual(slots[0].x, 1.7, delta=0.1)
        gx, gy, gh = park_goal_parallel(slots[0])
        self.assertAlmostEqual(gh, 0.0)
        self.assertGreater(gy, 0.62)
        short = np.array([(x, 0.62) for x in xs if not 1.2 < x < 1.5])                     # 30 cm: shorter than the car
        self.assertEqual(find_parallel_slots(short, side=1), [])
        self.assertEqual(find_parallel_slots(np.array([(x, 0.62) for x in xs]), side=1), [])


class EndToEnd(unittest.TestCase):
    def test_explores_a_furnished_room_without_contact(self):
        from sim.explore_eval import explore
        r = explore("room")
        self.assertFalse(r["crashed"])
        self.assertGreater(r["coverage"], 0.9)
        self.assertLess(r["t"], 90.0)

    def test_parallel_parks_between_two_objects(self):
        from sim.park_eval import parallel
        ok, r = parallel()
        self.assertFalse(r["crashed"])
        self.assertLess(r["err"], 0.12)
        self.assertLess(r["heading_err_deg"], 10.0)

    def test_parks_backwards_into_the_bay(self):
        from sim.autonav_eval import drive
        from sim.park_eval import bay_from_scan
        bays, goal = bay_from_scan("parking")
        self.assertEqual(len(bays), 1)
        r = drive("parking", goal[:2], t_max=60.0, heading_deg=goal[2], reverse_first=True)
        self.assertFalse(r["crashed"])
        self.assertLess(r["err"], 0.10)
        self.assertLess(r["heading_err_deg"], 12.0)


if __name__ == "__main__":
    unittest.main()
