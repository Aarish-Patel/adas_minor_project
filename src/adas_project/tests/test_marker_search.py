"""Park at the marker from a general area (adas/marker_search.py, TODO AQ3): the decision logic with a fake car."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adas.markers import MarkerObs                    # noqa: E402
from adas.marker_search import MarkerSearch, PARK_DIST_M, STANDOFF_M   # noqa: E402
import numpy as np                                    # noqa: E402


def obs(dist, bearing=0.0, mid=1):
    return MarkerObs(id=mid, corners=np.zeros((4, 2)), x=dist * math.cos(bearing), y=dist * math.sin(bearing), z=0.0, facing=math.pi,
                     dist=dist, bearing=bearing, t=0.0, size_px=40.0)


class FakeCar:
    def __init__(self):
        self.gotos, self.parked = [], 0

    def goto(self, x, y, h):
        self.gotos.append((x, y, h))
        return True

    def park(self):
        self.parked += 1
        return "planning the bay manoeuvre"


class MarkerSearchTest(unittest.TestCase):
    def run_step(self, ms, car, t, target=(), cands=(), front=2.0, active=False, msg="arrived"):
        return ms.step(t, (0, 0, 0), active, list(target), list(cands), front, car.goto, car.park, msg)

    def test_area_first_then_park_when_the_marker_is_near(self):
        car, ms = FakeCar(), MarkerSearch((2.0, 0.5, 10.0), 0.0, 1)
        self.run_step(ms, car, 0.1)
        self.assertEqual(car.gotos[0], (2.0, 0.5, None))                     # the area point (its heading is not asked for)
        self.run_step(ms, car, 0.2, active=True)                             # driving there: nothing new
        self.assertEqual(len(car.gotos), 1)
        self.run_step(ms, car, 5.0, target=[obs(PARK_DIST_M - 0.2)])
        self.assertEqual(car.parked, 1)
        self.assertTrue(ms.done and ms.ok)

    def test_a_far_cue_is_approached_in_a_short_step(self):
        car, ms = FakeCar(), MarkerSearch((1.0, 0.0, 0.0), 0.0, 1)
        self.run_step(ms, car, 0.1)
        self.run_step(ms, car, 3.0, cands=[{"dist": 3.0, "bearing": math.radians(10), "x": 0, "y": 0}])
        x, y, h = car.gotos[-1]
        self.assertAlmostEqual(math.hypot(x, y), min(3.0 - STANDOFF_M, 0.60), places=3)
        self.assertGreater(y, 0.0)                                           # towards it (left)
        self.assertEqual(car.parked, 0)

    def test_nothing_seen_sweeps_then_gives_up_at_the_wall(self):
        car, ms = FakeCar(), MarkerSearch((1.0, 0.0, 0.0), 0.0, 1)
        self.run_step(ms, car, 0.1)
        self.run_step(ms, car, 2.0, front=1.5)
        self.run_step(ms, car, 4.0, front=1.2)
        self.assertNotEqual(car.gotos[-1][1], 0.0)                           # the second search step swings the heading
        self.run_step(ms, car, 6.0, front=0.4)
        self.assertTrue(ms.done and not ms.ok)

    def test_seen_on_the_way_parks_at_once(self):
        car, ms = FakeCar(), MarkerSearch((2.0, 0.0, 0.0), 0.0, 1)
        self.run_step(ms, car, 0.1)
        self.run_step(ms, car, 1.0, target=[obs(1.2)], active=True)
        self.assertEqual(car.parked, 1)

    def test_operator_cancel_ends_it(self):
        car, ms = FakeCar(), MarkerSearch((1.0, 0.0, 0.0), 0.0, 1)
        self.run_step(ms, car, 0.1)
        self.run_step(ms, car, 2.0, msg="cancelled from the GUI")
        self.assertTrue(ms.done and not ms.ok)
        self.assertEqual(len(car.gotos), 1)


if __name__ == "__main__":
    unittest.main()
