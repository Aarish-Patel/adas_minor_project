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

    def test_goal_heading_is_reached_forward(self):
        """A goal pose (position + heading): the path ends on the pose, driving forward (Dubins approach)."""
        from sim.hw_worlds import open_room
        path = HybridAStar(P).plan_to_point(world_points(open_room()[0]), (0.0, 0.0, 0.0), (1.8, 1.0),
                                            goal_heading=math.radians(90))
        self.assertIsNotNone(path)
        self.assertLess(math.hypot(path[-1, 0] - 1.8, path[-1, 1] - 1.0), 0.05)
        self.assertLess(abs(math.degrees(path[-1, 2]) - 90), 3)
        self.assertTrue((path[:, 3] > 0).all())

    def test_prefers_forward_even_for_a_goal_behind(self):
        """Click-to-go goal behind the car in an open room: a forward U-turn, no reversing (user request)."""
        from adas.plan_service import plan_point_job
        from sim.hw_worlds import open_room
        path = plan_point_job(P, 1.5, world_points(open_room()[0]), (0.0, 0.0, 0.0), (-0.3, 1.6))
        self.assertIsNotNone(path)
        self.assertTrue((path[:, 3] > 0).all())

    def test_reverses_when_the_heading_needs_it(self):
        """Arrive pointing backwards in a corridor too narrow to turn round: the planner has to reverse."""
        from adas.plan_service import plan_point_job
        from sim.hw_worlds import corridor
        path = plan_point_job(P, 1.5, world_points(corridor()[0]), (0.0, 0.0, 0.0), (-0.5, 0.0),
                              goal_heading=0.0)
        self.assertIsNotNone(path)
        self.assertTrue((path[:, 3] < 0).any())
        self.assertLess(math.hypot(path[-1, 0] + 0.5, path[-1, 1]), 0.06)

    def test_drives_to_a_goal_pose_in_the_twin(self):
        from sim.autonav_eval import drive
        r = drive("open", (1.8, 0.8), heading_deg=90)
        self.assertTrue(r["ok"], r["why"])
        self.assertFalse(r["crashed"])
        self.assertLess(r["heading_err_deg"], 20)

    def test_goal_inside_an_obstacle_is_refused(self):
        from sim.hw_worlds import doorway
        self.assertIsNone(HybridAStar(P).plan_to_point(world_points(doorway()[0]), (0.0, 0.0, 0.0), (1.7, 0.0)))

    def test_drives_there_in_the_twin_without_crashing(self):
        from sim.autonav_eval import drive
        r = drive("gap", (3.2, -0.7))
        self.assertTrue(r["ok"], r["why"])
        self.assertFalse(r["crashed"])

    def test_holds_after_arrival_until_the_throttle_is_released(self):
        a = RelayAssists(TUN)                                   # inline planning: the plan is there at once
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
        a.goto(1.5, 0.0, points=[(0.0, 3.0)])
        out = a.process([f"A {a.centre + 30:.0f} {a.centre + 30:.0f}", "M -150"], [(0.0, 3.0)], 1, now=0.0)
        self.assertFalse(a.nav.active)
        self.assertIn("M -150", out)


class PlanServiceTests(unittest.TestCase):
    def test_worker_process_plans_without_blocking(self):
        """The relay's mode: the search runs in another process; submitting returns at once."""
        import time
        from adas.plan_service import PlanService, plan_line_job
        from sim.hw_worlds import doorway
        svc = PlanService("process").start()
        try:
            pts = world_points(doorway()[0])
            t0 = time.perf_counter()
            job = svc.submit(plan_line_job, P, 1.5, pts, (0.0, 0.0, 0.0), 2.3, 2500, 4000)
            self.assertLess(time.perf_counter() - t0, 0.05)
            while not job.ready():
                time.sleep(0.01)
            path = job.result()
            self.assertIsNotNone(path)
            self.assertGreater(path[-1, 0], 2.8)
        finally:
            svc.shutdown()

    def test_inline_mode_releases_after_the_pi_delay(self):
        from adas.plan_service import PlanService
        svc = PlanService("inline", latency_factor=1e6)
        job = svc.submit(sum, [1, 2])
        self.assertFalse(job.ready(0.05))
        self.assertTrue(job.ready(1e6))
        self.assertEqual(job.result(), 3)


def box_pts(cx, cy, hx, hy, n=30):
    xs, ys = np.linspace(cx - hx, cx + hx, n), np.linspace(cy - hy, cy + hy, n)
    return np.vstack([np.column_stack([xs, np.full(n, cy - hy)]), np.column_stack([xs, np.full(n, cy + hy)]),
                      np.column_stack([np.full(n, cx - hx), ys]), np.column_stack([np.full(n, cx + hx), ys])])


class FallbackTests(unittest.TestCase):
    def test_mppi_steers_round_a_box(self):
        from adas.hybrid_astar import Grid
        from adas.mppi import MPPI
        grid = Grid(box_pts(1.6, 0.0, 0.13, 0.13), -1.0, 5.0, -2.0, 2.0)
        m = MPPI(P)
        x = y = th = 0.0
        closest = 9.0
        for _ in range(160):
            k, ok = m.step(grid, (x, y, th), 0.4, x_goal=2.2)
            self.assertTrue(ok)
            th += k * 0.4 * 0.05
            x += 0.4 * math.cos(th) * 0.05
            y += 0.4 * math.sin(th) * 0.05
            closest = min(closest, float((grid.lookup(x + np.cos(th) * m.circle_x, y + np.sin(th) * m.circle_x)
                                          - m.circle_r).min()))
            if x > 2.4:
                break
        self.assertGreater(x, 2.4)
        self.assertGreater(closest, 0.04)

    def test_evasive_falls_back_to_mppi_when_no_plan(self):
        from adas.assists import DrivingAssists
        from adas.plan_service import Job
        a = DrivingAssists(P, TUN.speed_model)
        a.enabled["evasive"] = True
        a.cfg.evade_ttc, a.cfg.evade_min_look = 4.0, 1.6       # trigger early enough for a local swerve
        a.plan_service.submit = lambda *args, **kw: Job(result=None)     # "Hybrid A* found nothing"
        from adas.vehicle_params import steer_to_delta
        world = box_pts(P.front_x + 1.3, 0.0, 0.13, 0.13)
        x = y = th = 0.0
        saw_mppi, closest = False, 9.0
        for _ in range(120):                                   # a simple kinematic car at 0.4 m/s, 6 s
            c, s_ = math.cos(th), math.sin(th)
            dx, dy = world[:, 0] - x, world[:, 1] - y
            pts = np.column_stack([c * dx + s_ * dy, -s_ * dx + c * dy])
            steer, _pwm, _lvl = a.update(0.05, pts, 0.0, 200, 0.4)
            saw_mppi = saw_mppi or "MPPI" in a.info.get("evasive", "")
            k = math.tan(steer_to_delta(steer, P)) / P.wheelbase
            th += k * 0.4 * 0.05
            x += 0.4 * math.cos(th) * 0.05
            y += 0.4 * math.sin(th) * 0.05
            lx = c * dx + s_ * dy
            ly = -s_ * dx + c * dy
            closest = min(closest, float(np.hypot(np.maximum(np.maximum(P.rear_x - lx, 0), lx - P.front_x),
                                                  np.maximum(np.abs(ly) - P.width / 2, 0)).min()))
            if x > P.front_x + 1.3 + 0.4:
                break
        self.assertTrue(saw_mppi)                              # the fallback steered...
        self.assertGreater(closest, 0.0)                       # ...without touching the box...
        self.assertGreater(x, P.front_x + 1.3 + 0.4)           # ...and got past it


class NudgeTests(unittest.TestCase):
    def test_small_correction_instead_of_braking(self):
        from pi.relay_assists import RelayAssists
        a = RelayAssists(TUN)
        a.set("nudge", True)
        g = PathGate(P, TUN.speed_model)
        g.on_scan(box_pts(P.front_x + 0.45, 0.22, 0.15, 0.15), 1)       # clips the left side by ~3 cm
        c = f"{a.centre:.0f}"
        out, d = a.nudge(g, [f"A {c} {c}", "M -150"], 0.47)
        self.assertIsNotNone(d)
        self.assertLess(d, 0.0)                                          # steered right, away from the box
        self.assertNotEqual(out[0], f"A {c} {c}")


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
        g = self.gate(wall(P.front_x + 0.04, -1, P.front_x + 0.04, 1))
        g.decide(0.05, 200, 0.0, 0.6)                        # brakes
        out, _ = g.decide(0.05, 200, 0.0, 0.0)               # driver still pushing, no room even to creep: hold
        self.assertEqual(out, 0)
        g.decide(0.05, 0, 0.0, 0.0)                          # driver lets go -> released
        self.assertIsNone(g.latch)

    def test_stopped_with_room_to_creep_goes_on_slowly(self):
        """After braking, stopped with the commanded path clear at a creep: released onto it at creep speed only
        (user, 28 Sep: a path that is safe slower must not stay held)."""
        from pi.path_gate import CREEP_V
        g = self.gate(wall(P.front_x + 0.12, -1, P.front_x + 0.12, 1))
        g.decide(0.05, 200, 0.0, 0.6)                        # brakes
        out, _ = g.decide(0.05, 200, 0.0, 0.0)               # stopped: the path is clear at a creep
        self.assertGreater(out, 0)
        self.assertLessEqual(out, g.model.pwm_for_speed(CREEP_V) + 1)


class IntentTests(unittest.TestCase):
    def test_model_loads_and_scores(self):
        from adas.intent_net import DriverProfile, IntentNet, features
        net = IntentNet(os.path.join(ROOT, "pi", "intent_net.json"))
        self.assertGreater(net.report.get("test_auc", 0), 0.8)
        hist = [87.0] * 80
        p = []
        for d in (1.2, 0.25):                                              # stick frozen, throttle held
            pts = wall(P.front_x + d, -1, P.front_x + d, 1)
            f = features(hist, 150, 0.45, pts, P, 87.0, 0.0656, profile=DriverProfile())
            self.assertIsNotNone(f)
            p.append(net.crash_probability(f))
        self.assertGreater(p[1], p[0])                                      # closer = riskier

    def test_frozen_stick_hold_ends_at_the_last_point_to_steer(self):
        """However much the model trusts a driver who is NOT moving the stick, the swerve still starts in time."""
        from adas.assists import DrivingAssists
        a = DrivingAssists(P, TUN.speed_model)
        a.enabled["evasive"] = True
        a.intent_hold, a.intent_attentive, a.intent_k_rate = True, False, 0.0
        v = 0.8
        lps = a._last_point_to_steer(v)
        for gap, expect in ((lps + 0.25, False), (lps - 0.1, True)):
            a._stop_evading()
            a._trigger_for = 0.0
            box = wall(P.front_x + gap + 0.03, -0.15, P.front_x + gap + 0.03, 0.15, 40)
            started = False
            for _ in range(6):                                              # > the 0.15 s confirm time
                a.update(0.05, box, 0.0, 200, v)
                started = started or a.phase is not None
            self.assertEqual(started, expect, f"gap {gap:.2f} m, last point to steer {lps:.2f} m")


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


class StuckRecoveryTests(unittest.TestCase):
    """User, 28 Sep: at full throttle the evasive steer / click-to-go sat still in front of an obstacle although a
    way round existed at a slower speed; reversing legs overshot; the throttle jumped between directions."""

    def test_full_throttle_close_to_a_box_gets_round(self):
        from sim.relay_scenarios import _world, run, steady
        from sim.world import Box
        w = _world()
        w.add(Box(0.80, 0.0, 0.26, 0.26))                       # 0.4 m ahead of the bumper, full throttle from rest
        r = run(w, steady(255), 30, ("evasive",), stop_when=lambda t, x, y, th, v: x > 2.3)
        self.assertFalse(r["collided"])
        self.assertGreater(r["x"], 2.0)                           # round it, not held in front of it

    def test_cusp_switches_to_the_next_leg_after_an_overshoot(self):
        from adas.assists import past_cusp
        P = np.array([[0.0, 0, 0, -1], [-0.2, 0, 0, -1], [-0.4, 0, 0, -1], [-0.4, 0, 0, 1], [-0.2, 0.1, 0.3, 1]])
        self.assertEqual(past_cusp(P, 1, (-0.25, 0.0, 0.0)), 1)   # still reversing towards the cusp
        self.assertEqual(past_cusp(P, 2, (-0.45, 0.0, 0.0)), 3)   # overshot it: drive the forward leg

    def test_throttle_is_smoothed_but_safety_cuts_are_not(self):
        from pi.relay_assists import ThrottleSmoother
        t = ThrottleSmoother()
        self.assertLess(t.step(255, 0.05), 60)                    # no jump to full throttle
        for _ in range(20):
            t.step(255, 0.05)
        seq = [t.step(-150, 0.05) for _ in range(12)]
        self.assertTrue(all(b <= a for a, b in zip(seq, seq[1:])))  # forward -> reverse through zero, no jump
        self.assertIn(0.0, seq)
        self.assertEqual(t.step(200, 0.05, emergency=True), 200)
        self.assertEqual(t.step(0, 0.05, emergency=True), 0)      # a brake / hold cut is immediate
