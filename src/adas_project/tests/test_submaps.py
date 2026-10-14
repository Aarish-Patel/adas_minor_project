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


class Relocalisation(unittest.TestCase):
    """Global localisation against the submaps with no pose guess (TODO P2-2, sim/reloc_eval.py)."""

    @classmethod
    def setUpClass(cls):
        from sim.reloc_eval import Twin, build_map
        cls.tw = Twin("lounge", seed=11)               # an asymmetric room: every place looks different
        cls.g = build_map(cls.tw)

    def poses(self, n, seed):
        from sim.reloc_eval import random_pose
        rng = np.random.default_rng(seed)
        return [random_pose(self.tw, rng) for _ in range(n)]

    def test_one_scan_gives_the_place_within_centimetres(self):
        for x, y, th in self.poses(6, 5):
            fix = self.g.relocalise(self.tw.scan(x, y, th))
            self.assertIsNotNone(fix)
            self.assertLess(math.hypot(fix.pose[0] - x, fix.pose[1] - y), 0.03)
            self.assertLess(abs((fix.pose[2] - th + math.pi) % (2 * math.pi) - math.pi), math.radians(1.5))
            self.assertGreater(fix.margin, self.g.reloc["margin"])

    def test_a_symmetric_room_gives_no_fix_instead_of_a_wrong_one(self):
        from sim.reloc_eval import Twin, build_map, random_pose
        tw = Twin("open", seed=2)                      # an empty rectangle looks the same turned 180 degrees
        g = build_map(tw)
        rng = np.random.default_rng(3)
        for _ in range(4):
            self.assertIsNone(g.relocalise(tw.scan(*random_pose(tw, rng))))

    def test_a_scan_from_somewhere_else_gives_no_fix(self):
        from sim.reloc_eval import Twin, random_pose
        other = Twin("room", seed=4)                   # not the mapped room
        rng = np.random.default_rng(6)
        for _ in range(3):
            self.assertIsNone(self.g.relocalise(other.scan(*random_pose(other, rng))))

    def test_two_scans_must_agree_after_the_motion_between_them(self):
        from adas.submaps import Relocaliser
        x, y, th = self.poses(1, 8)[0]
        rl = Relocaliser(self.g)
        self.assertIsNone(rl.update(0.0, np.zeros(3), self.tw.scan(x, y, th)))                      # one fix is not enough
        x2, y2 = x + 0.1 * math.cos(th), y + 0.1 * math.sin(th)
        fix = rl.update(0.1, np.array([0.1, 0.0, 0.0]), self.tw.scan(x2, y2, th))
        self.assertIsNotNone(fix)
        self.assertLess(math.hypot(fix.pose[0] - x2, fix.pose[1] - y2), 0.03)
        # a second scan that disagrees with the first (the car was moved in between) is not confirmed
        rl2 = Relocaliser(self.g)
        a, b = self.poses(2, 9)
        rl2.update(0.0, np.zeros(3), self.tw.scan(*a))
        self.assertIsNone(rl2.update(0.1, np.array([0.1, 0.0, 0.0]), self.tw.scan(*b)))

    def test_a_saved_map_can_be_reloaded_and_relocalised_in(self):
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "map.npz")
        self.g.save(path)
        g2 = PoseGraphMap.load(path)
        self.assertEqual(len(g2.nodes), len(self.g.nodes))
        self.assertEqual(len(g2.submaps), len(self.g.submaps))
        self.assertTrue(np.allclose(g2.nodes, self.g.nodes, atol=1e-6))
        x, y, th = self.poses(1, 12)[0]
        fix = g2.relocalise(self.tw.scan(x, y, th))
        self.assertIsNotNone(fix)
        self.assertLess(math.hypot(fix.pose[0] - x, fix.pose[1] - y), 0.03)

    def test_rebasing_joins_the_new_stretch_to_the_map(self):
        from adas.submaps import Relocaliser
        g = PoseGraphMap.load(self._saved())
        x, y, th = self.poses(1, 13)[0]
        rl = Relocaliser(g)
        fix = None
        for k in range(2):
            px, py = x + 0.05 * k * math.cos(th), y + 0.05 * k * math.sin(th)
            est = np.array([0.05 * k, 0.0, 0.0])
            fix = rl.update(0.1 * k, est, self.tw.scan(px, py, th))
        g.rebase(fix, est)
        n, e = len(g.nodes), len(g.edges)
        pose = g.add_scan(est, self.tw.scan(px, py, th))
        self.assertLess(math.hypot(pose[0] - px, pose[1] - py), 0.03)
        self.assertEqual(len(g.nodes), n + 1)
        self.assertNotIn((n - 1, n), [(a[0], a[1]) for a in g.edges])          # no odometry edge across the break
        self.assertIn(n, g.breaks)
        self.assertGreater(len(g.edges), e)                                     # tied to the map instead

    def _saved(self):
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "map.npz")
        self.g.save(path)
        return path

    def test_the_service_finds_the_pose_of_a_car_put_down_in_a_mapped_room(self):
        from sim.reloc_eval import kidnap_end_to_end
        r = kidnap_end_to_end("room", seed=3, verbose=False)
        self.assertTrue(r["relocalised"])
        self.assertLessEqual(r["scans_needed"], 3)
        self.assertLess(r["err_cm"], 3.0)
        self.assertLess(r["heading_err_deg"], 1.5)


class ServiceMaps(unittest.TestCase):
    def test_the_service_saves_and_loads_a_map_and_the_relay_class_exposes_it(self):
        import tempfile
        from adas.config import load_tuning
        from pi.relay_assists import RelaySpeed
        from sim.reloc_eval import Twin, build_map, random_pose
        tw = Twin("room", seed=5)
        g = build_map(tw)
        path = os.path.join(tempfile.mkdtemp(), "room.npz")
        g.save(path)
        tun = load_tuning(os.path.join(os.path.dirname(__file__), "..", "pi", "tuning_real_car.json"))
        rs = RelaySpeed(tun.speed_model, tw.p.lidar_x)
        self.assertFalse(rs.lost)
        self.assertTrue(rs.load_map(path))
        self.assertTrue(rs.lost)                                       # a loaded map: the pose is unknown until a fix
        self.assertFalse(rs.load_map(path + ".missing"))               # a missing file changes nothing
        out = os.path.join(tempfile.mkdtemp(), "saved.npz")
        self.assertTrue(rs.save_map(out))
        self.assertTrue(os.path.exists(out))
        rs.slam.stop()

    def test_lost_mode_gives_up_after_the_timeout_and_carries_on(self):
        import time
        from adas.submaps import SlamService
        from sim.reloc_eval import Twin, build_map, random_pose
        tw = Twin("lounge", seed=6)
        svc = SlamService(give_up_s=0.5)
        svc.graph = build_map(tw)
        other = Twin("room", seed=6)
        rng = np.random.default_rng(6)
        svc.relocalise()
        for k in range(4):                                             # scans from a place that is not in the map
            svc.submit(np.array([0.1 * k, 0.0, 0.0]), other.scan(*random_pose(other, rng)), t=0.2 * k)
            time.sleep(0.4)
        end = time.time() + 10
        while svc.lost and time.time() < end:
            time.sleep(0.05)
        self.assertFalse(svc.lost)
        self.assertEqual(svc.relocalisations, 0)                       # nothing was found; mapping goes on from where it was
        svc.stop()


class KidnapEndToEnd(unittest.TestCase):
    def test_a_car_lifted_and_dropped_mid_drive_finds_itself_again(self):
        import time
        from adas.submaps import SlamService
        from sim.reloc_eval import Twin, random_pose
        tw = Twin("room", seed=9)
        svc = SlamService()
        x0, x1, y0, y1 = tw.bbox
        seq = 0

        def feed(px, py, th, est):
            nonlocal seq
            svc.submit(est, tw.scan(px, py, th), t=0.1 * seq)
            seq += 1
            end = time.time() + 20
            while svc.scans_seen < seq and time.time() < end:
                time.sleep(0.002)
        # a serpentine mapping drive, front-end pose = the true pose (the drift is slam_eval's business)
        for k, y in enumerate(np.arange(y0 + 0.4, y1 - 0.3, 0.45)):
            xs = np.arange(x0 + 0.4, x1 - 0.3, 0.1)
            xs = xs if k % 2 == 0 else xs[::-1]
            for x in xs:
                th = 0.0 if k % 2 == 0 else math.pi
                if tw.free(x, y, th):
                    feed(x, y, th, np.array([x, y, th]))
        self.assertFalse(svc.lost)
        est = np.array([xs[-1], y, th])                                   # where the front end thinks the car is
        rng = np.random.default_rng(10)
        px, py, pth = random_pose(tw, rng)                                # ... but it was carried elsewhere
        for k in range(8):
            qx, qy = px + 0.05 * k * math.cos(pth), py + 0.05 * k * math.sin(pth)
            est = est + np.array([0.05 * math.cos(th), 0.05 * math.sin(th), 0.0])
            feed(qx, qy, pth, est)
            if k == 0:
                self.assertTrue(svc.lost or svc.relocalisations > 0)      # the jump was noticed
        end = time.time() + 20
        while svc.lost and time.time() < end:
            time.sleep(0.02)
        self.assertFalse(svc.lost)
        self.assertGreaterEqual(svc.relocalisations, 1)
        c = svc.corrected(est)
        self.assertLess(math.hypot(c[0] - qx, c[1] - qy), 0.05)
        svc.stop()


class KidnapDetector(unittest.TestCase):
    def test_a_lift_and_drop_shows_and_driving_does_not(self):
        from adas.submaps import kidnap_score
        from sim.reloc_eval import Twin, random_pose
        tw = Twin("lounge", seed=1)
        rng = np.random.default_rng(1)
        low = []
        for _ in range(12):
            a, b = random_pose(tw, rng), random_pose(tw, rng)
            low.append(kidnap_score(tw.scan(*a), tw.scan(*b)))
            # 5 cm ahead and 9 degrees of turn between two scans (a fast turn at 0.1 s per scan is ~20 degrees at most)
            nxt = (a[0] + 0.05 * math.cos(a[2]), a[1] + 0.05 * math.sin(a[2]), a[2] + math.radians(9))
            self.assertGreater(kidnap_score(tw.scan(*a), tw.scan(*nxt)), 0.8)
        self.assertGreaterEqual(sum(s < 0.30 for s in low), 9)               # most drops are seen (a few look alike)


if __name__ == "__main__":
    unittest.main()
