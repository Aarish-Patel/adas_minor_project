"""The dashboard's map (gui/ev.py WorldStore): points the LiDAR sees past are removed (visibility-based removal, Kim & Kim 2020);
points it cannot see now - too close, hidden behind a nearer return, or in a direction without a return - are kept."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gui.ev import WorldStore


def wall(dist, centre=0.0, half=30.0, step=0.5):
    """A flat wall `dist` from the LiDAR, square to the bearing `centre` (deg clockwise from ahead), spanning +-half deg, in the
    relay's polar form (deg clockwise from ahead, range from the LiDAR)."""
    a = np.arange(-half, half + 1e-9, step)
    return np.column_stack([centre + a, dist / np.cos(np.radians(a))])


def scan(store, polar, pose=(0.0, 0.0, 0.0), times=1):
    for _ in range(3 * times):                     # add() uses every third packet
        store.add({"world": {"pose": list(pose)}, "_pts": polar})


def has_wall_at(store, x, tol=0.06):
    p = store.points()
    return bool(len(p)) and bool(np.any((np.abs(p[:, 0] - x) < tol) & (np.abs(p[:, 1]) < 0.3)))


class WorldStoreTest(unittest.TestCase):
    def setUp(self):
        from gui.dashboard import GEO
        self.lx = GEO["lidar_x"]
        self.s = WorldStore()
        scan(self.s, wall(2.0), times=4)                                       # a wall 2 m ahead, seen 4 times
        self.assertTrue(has_wall_at(self.s, self.lx + 2.0))

    def test_points_the_lidar_sees_past_are_removed(self):
        scan(self.s, wall(3.0), times=6)                                       # the wall is now 3 m away: seen past (a miss at 2 m counts half)
        self.assertFalse(has_wall_at(self.s, self.lx + 2.0))
        scan(self.s, wall(3.0), times=1)
        self.assertTrue(has_wall_at(self.s, self.lx + 3.0))

    def test_points_hidden_behind_a_nearer_return_stay(self):
        scan(self.s, wall(1.0), times=4)                                       # a box in front hides the wall
        self.assertTrue(has_wall_at(self.s, self.lx + 2.0))
        self.assertTrue(has_wall_at(self.s, self.lx + 1.0))

    def test_points_in_a_direction_without_returns_stay(self):
        scan(self.s, wall(1.0, centre=75.0, half=15.0), times=4)                     # returns only off to the right
        self.assertTrue(has_wall_at(self.s, self.lx + 2.0))

    def test_points_too_close_to_see_stay(self):
        # the car drives up to 0.12 m from the wall: the LiDAR cannot see it (under 0.20 m), only a wall far behind it to the side
        pose = (2.0 - 0.12, 0.0, 0.0)
        scan(self.s, np.vstack([wall(0.12), wall(2.5, centre=90.0, half=10.0)]), pose=pose, times=4)
        self.assertTrue(has_wall_at(self.s, self.lx + 2.0))

    def test_rotated_pose(self):
        # the same wall seen from a car turned 90 deg left (the wall is then to its right, bearing +90 clockwise)
        s = WorldStore()
        scan(s, wall(2.0, centre=90.0), pose=(0.0, 0.0, math.pi / 2), times=4)
        p = s.points()
        self.assertTrue(len(p) and np.all(np.abs(p[:, 0] - 2.0) < 0.06))      # facing +y, the car's right is world +x
        scan(s, wall(3.0, centre=90.0), pose=(0.0, 0.0, math.pi / 2), times=6)
        self.assertFalse(np.any(np.abs(s.points()[:, 0] - 2.0) < 0.06))

    def test_a_thin_post_seen_close_stays_from_far_away(self):
        """User, 14 Oct: thin objects seen close were deleted once the car moved away. A 2 cm post seen at 0.4 m, then from 2 m with the
        wall 3 m away seen right beside it on every scan: kept (too thin to be judged from there)."""
        s = WorldStore()
        post = np.array([[0.0, 0.40]] * 3)                                    # 0.40 m straight ahead
        scan(s, np.vstack([post, wall(3.0)]), times=4)
        lx = self.lx
        self.assertTrue(np.any(np.hypot(s.points()[:, 0] - (lx + 0.40), s.points()[:, 1]) < 0.06))
        back = (-1.6, 0.0, 0.0)                                                # the car backed 1.6 m: the post is 2 m ahead now
        scan(s, wall(4.6), pose=back, times=20)
        self.assertTrue(np.any(np.hypot(s.points()[:, 0] - (lx + 0.40), s.points()[:, 1]) < 0.06))

    def test_clear_and_cap(self):
        s = WorldStore(max_cells=50)
        scan(s, wall(2.0), times=2)
        self.assertLessEqual(len(s.score), 50)
        s.clear()
        self.assertEqual(len(s.points()), 0)


if __name__ == "__main__":
    unittest.main()
