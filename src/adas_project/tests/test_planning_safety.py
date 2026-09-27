"""Hybrid A* evasive planning, the path-predicted brake gate, obstacle memory and the learned intent model."""
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from adas.config import load_tuning  # noqa: E402
from adas.hybrid_astar import HybridAStar  # noqa: E402
from pi.path_gate import PathGate  # noqa: E402
from pi.relay_assists import RelayAssists, car_params  # noqa: E402

TUN = load_tuning(os.path.join(ROOT, "pi", "tuning_real_car.json"))
P = car_params(TUN.mount)


def world_points(world, spacing=0.02):
    pts = []
    for x1, y1, x2, y2 in world.segments():
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / spacing))
        for t in np.linspace(0, 1, n):
            pts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    return np.array(pts)


def wall(x0, y0, x1, y1, n=200):
    return np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)])


class HybridAStarTests(unittest.TestCase):
    def setUp(self):
        from sim.hw_worlds import doorway
        w, _ = doorway()
        self.pts = world_points(w)

    def test_doorway_path_goes_round_the_box_and_back_through_the_door(self):
        path = HybridAStar(P).plan(self.pts, (0.0, 0.0, 0.0), x_goal=2.3)
        self.assertIsNotNone(path)
        self.assertGreater(path[-1, 0], 2.8)                       # past the dividing wall
        self.assertLess(abs(path[-1, 1]), 0.05)                     # back on the line
        self.assertTrue((path[:, 3] > 0).all())                     # forward only

    def test_too_close_backs_off_first(self):
        ha = HybridAStar(P)
        self.assertIsNone(ha.plan(self.pts, (1.1, -0.05, 0.1), x_goal=2.3))
        path = ha.plan(self.pts, (1.1, -0.05, 0.1), x_goal=2.3, allow_reverse=True, max_nodes=4000)
        self.assertIsNotNone(path)
        self.assertTrue((path[:, 3] < 0).any())

    def test_boxed_in_has_no_path(self):
        pts = np.vstack([self.pts, wall(0.5, 0.3, 3.0, 0.3), wall(0.5, -0.3, 3.0, -0.3)])
        self.assertIsNone(HybridAStar(P).plan(pts, (0.0, 0.0, 0.0), x_goal=2.3))


class ClickToGoTests(unittest.TestCase):
    def test_point_goal_through_the_doorway_forward_only(self):
        from sim.hw_worlds import doorway
        path = HybridAStar(P).plan_to_point(world_points(doorway()[0]), (0.0, 0.0, 0.0), (3.4, 0.0))
        self.assertIsNotNone(path)
        self.assertLess(math.hypot(path[-1, 0] - 3.4, path[-1, 1]), 0.05)
        self.assertTrue((path[:, 3] > 0).all())

    def test_goal_behind_in_a_corridor_reverses(self):
        from sim.hw_worlds import corridor
        path = HybridAStar(P).plan_to_point(world_points(corridor()[0]), (0.0, 0.0, 0.0), (-0.6, 0.0),
                                            allow_reverse=True)
        self.assertIsNotNone(path)
        self.assertTrue((path[:, 3] < 0).all())

    def test_goal_inside_an_obstacle_is_refused(self):
        from sim.hw_worlds import doorway
        self.assertIsNone(HybridAStar(P).plan_to_point(world_points(doorway()[0]), (0.0, 0.0, 0.0), (1.7, 0.0)))

    def test_drives_there_in_the_twin_without_crashing(self):
        from sim.autonav_eval import drive
        r = drive("gap", (3.2, -0.7))
        self.assertTrue(r["ok"], r["why"])
        self.assertFalse(r["crashed"])

    def test_holds_after_arrival_until_the_throttle_is_released(self):
        a = RelayAssists(TUN)
        a.nav.threaded = False
        self.assertTrue(a.goto(1.0, 0.0, points=[(0.0, 3.0)]))
        c = f"{a.centre:.0f}"
        held = [f"A {c} {c}", "M -150"]
        a.process(held, [(0.0, 3.0)], 1, now=0.0)
        a.nav.cancel("cancelled from the GUI")
        self.assertIn("M 0", a.process(held, [(0.0, 3.0)], 2, now=0.05))           # still held: stay stopped
        a.process([f"A {c} {c}", "M 0"], [(0.0, 3.0)], 3, now=0.10)                  # operator lets go
        self.assertIn("M -150", a.process(held, [(0.0, 3.0)], 4, now=0.15))         # the driver has it again

    def test_steering_hands_back_at_once(self):
        a = RelayAssists(TUN)
        a.nav.threaded = False
        a.goto(1.5, 0.0, points=[(0.0, 3.0)])
        out = a.process([f"A {a.centre + 30:.0f} {a.centre + 30:.0f}", "M -150"], [(0.0, 3.0)], 1, now=0.0)
        self.assertFalse(a.nav.active)
        self.assertIn("M -150", out)


class GateTests(unittest.TestCase):
    def gate(self, pts):
        g = PathGate(P, TUN.speed_model)
        g.on_scan(pts, 1)
        return g

    def test_passing_beside_a_wall_is_not_braking(self):
        g = self.gate(wall(-0.5, P.width / 2 + 0.18, 2.0, P.width / 2 + 0.18))
        out, brk = g.decide(0.05, 150, 0.0, 0.45)
        self.assertEqual(out, 150)
        self.assertFalse(brk)

    def test_head_on_wall_brakes_close_in(self):
        g = self.gate(wall(P.front_x + 0.12, -1, P.front_x + 0.12, 1))
        out, brk = g.decide(0.05, 150, 0.0, 0.45)
        self.assertTrue(brk or out <= 0)

    def test_not_frozen_next_to_a_box_can_back_away(self):
        box = wall(P.front_x + 0.02, -0.13, P.front_x + 0.02, -0.25, 30)        # at the front-right corner
        g = self.gate(box)
        out, _ = g.decide(0.05, -120, 0.0, 0.0)
        self.assertLess(out, 0)                                                  # reversing is allowed

    def test_phantom_memory_is_pruned_by_the_live_scan(self):
        g = PathGate(P, TUN.speed_model)
        far = wall(P.lidar_x + 0.69, -1, P.lidar_x + 0.69, 1)
        g.on_scan(far, 1)
        g.memory.advance(1.0, 0.5, 0.0)                     # dead reckoning claims 0.5 m that never happened
        g.on_scan(far, 2)
        out, _ = g.decide(0.05, 150, 0.0, 0.3)
        self.assertGreater(out, 100)

    def test_brake_latch_holds_until_released(self):
        g = self.gate(wall(P.front_x + 0.10, -1, P.front_x + 0.10, 1))
        g.decide(0.05, 200, 0.0, 0.6)                        # brakes
        out, _ = g.decide(0.05, 200, 0.0, 0.0)               # driver still pushing: hold, no throttle
        self.assertEqual(out, 0)
        g.decide(0.05, 0, 0.0, 0.0)                          # driver lets go -> released
        self.assertIsNone(g.latch)


class IntentTests(unittest.TestCase):
    def test_model_loads_and_scores(self):
        from adas.intent_net import DriverProfile, IntentNet, features
        net = IntentNet(os.path.join(ROOT, "pi", "intent_net.json"))
        self.assertGreater(net.report.get("test_auc", 0), 0.75)
        hist = [87.0] * 80
        pts = wall(P.front_x + 0.4, -1, P.front_x + 0.4, 1)              # wall 40 cm ahead, stick frozen
        f = features(hist, 150, 0.45, pts, P, 87.0, 0.0656, profile=DriverProfile())
        self.assertIsNotNone(f)
        self.assertGreater(net.crash_probability(f), 0.3)                 # frozen stick at a wall: risky


class RelayScenarioTests(unittest.TestCase):
    """The car's relay code on the digital twin (sim/relay_scenarios.py)."""

    def test_no_needless_slowing_beside_a_wall(self):
        from sim.relay_scenarios import passing_beside_a_wall
        ok, detail = passing_beside_a_wall()
        self.assertTrue(ok, detail)

    def test_full_speed_at_a_wall_stops_close_without_contact(self):
        from sim.relay_scenarios import wall_full_speed
        ok, detail = wall_full_speed()
        self.assertTrue(ok, detail)


class MonteCarloSmoke(unittest.TestCase):
    def test_adas_prevents_a_crash_the_driver_alone_has(self):
        SEED = 1                                            # a room where the lapsing driver crashes alone
        from sim.relay_mc import run
        off = run((SEED, "off", "lapsing"))
        on = run((SEED, "adas+intent", "lapsing"))
        self.assertTrue(off["crashed"])
        self.assertFalse(on["crashed"])


if __name__ == "__main__":
    unittest.main()
