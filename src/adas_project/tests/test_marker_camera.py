"""Home-demo bay parking with a front camera (TODO AN1-AN3): camera config, the twin's bay world, the marker camera pipeline, the relay's client
and status, the head-on PnP fix, and one nose-in run through the relay."""
import json
import math
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def write_config(d, **cfg):
    path = os.path.join(d, "camera.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


class CameraConfigTest(unittest.TestCase):
    def test_defaults_and_overrides(self):
        from adas import camera_config
        with tempfile.TemporaryDirectory() as d:
            cfg = camera_config.load(write_config(d, facing="front", marker_size_m=0.06, z=0.14, pitch_deg=9.0))
            self.assertEqual(cfg["marker_id"], 23)                              # not in the file: the default
            cam = camera_config.camera_model(cfg)
            self.assertAlmostEqual(cam.z, 0.14)
            self.assertAlmostEqual(cam.pitch_deg, 9.0)
            self.assertEqual(cam.yaw_deg, 0.0)                                  # a front camera looks ahead
            rear = camera_config.camera_model(camera_config.load(write_config(d, facing="rear")))
            self.assertEqual(rear.yaw_deg, 180.0)

    def test_bad_facing_is_refused(self):
        from adas import camera_config
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                camera_config.load(write_config(d, facing="sideways"))

    def test_the_repo_default_is_the_home_demo_front_camera(self):
        from adas import camera_config
        os.environ.pop("RC_CAMERA_CONFIG", None)
        cfg = camera_config.load()
        self.assertEqual(cfg["facing"], "front")
        self.assertLessEqual(cfg["marker_size_m"], 0.10)                        # a small marker

    def test_bay_world_follows_the_facing(self):
        from sim.parking_worlds import perpendicular
        _, front = perpendicular(23, facing="front")
        _, rear = perpendicular(23, facing="rear")
        self.assertAlmostEqual(abs(front[2]), math.pi)                         # nose in: the car faces the wall (x = -3)
        self.assertLess(front[0], -1.0)
        self.assertEqual(rear[2], 0.0)


class MarkerPoseTest(unittest.TestCase):
    def test_a_marker_seen_exactly_head_on_has_a_pose(self):
        """IPPE returns NaN for an exactly frontal view (nose-in parking looks straight at the marker): the iterative fallback gives it."""
        from adas.markers import marker_corners_3d, pose_from_corners, project
        from adas.park_select import front_camera
        cam, size = front_camera(), 0.08
        c = marker_corners_3d(size)
        n = np.array([-1.0, 0.0, 0.0])
        right = np.cross([0.0, 0.0, 1.0], n)
        for d, y in ((1.5, 0.0), (1.0, 0.0), (0.5, 0.02)):
            pts = np.array([[d, y, 0.10]]) + c[:, :1] * right + c[:, 1:2] * np.array([[0.0, 0.0, 1.0]])
            px, _ = project(cam, pts)
            o = pose_from_corners(px, 23, size, cam)
            self.assertIsNotNone(o, (d, y))
            self.assertAlmostEqual(o.x, d, delta=0.01)
            self.assertAlmostEqual(o.y, y, delta=0.01)
            self.assertAlmostEqual(abs(math.degrees(o.facing)), 180.0, delta=1.0)

    def test_non_finite_corners_give_no_pose(self):
        from adas.markers import pose_from_corners
        from adas.park_select import front_camera
        bad = np.array([[np.nan, 1.0], [2.0, 1.0], [2.0, 3.0], [1.0, 3.0]])
        self.assertIsNone(pose_from_corners(bad, 23, 0.08, front_camera()))


class MarkerServiceTest(unittest.TestCase):
    def test_pipeline_finds_the_marker_in_a_synthetic_frame(self):
        from adas import camera_config
        from adas.vision.sim_camera import SimCamera
        from pi.marker_camera import MarkerPipeline
        with tempfile.TemporaryDirectory() as d:
            cfg = camera_config.load(write_config(d, facing="front"))
            cam = camera_config.camera_model(cfg)
            frame = SimCamera(cam, (cam.width, cam.height), seed=1).render((0.0, 0.0, 0.0), markers=[(23, 1.6, 0.0, cam.z, math.pi, 0.08)])
            pipe = MarkerPipeline(cam, 23, 0.08, "front")
            vis, st = pipe.process(frame, time.time())
            self.assertTrue(st["ok"])
            self.assertEqual(len(st["markers"]), 1)
            m = st["markers"][0]
            self.assertAlmostEqual(m["x"], 1.6, delta=0.06)
            self.assertAlmostEqual(m["y"], 0.0, delta=0.04)
            self.assertEqual(vis.shape, frame.shape)
            other = MarkerPipeline(cam, 7, 0.08, "front")                       # a different id is not "the" marker
            self.assertEqual(other.process(frame, time.time())[1]["markers"], [])

    def test_client_keeps_recent_observations_and_drops_stale_state(self):
        from pi.marker_client import KEEP_S, MarkerClient
        c = MarkerClient(marker_id=23, start=False)
        now = 1000.0
        m = {"id": 23, "x": 1.2, "y": 0.05, "z": 0.1, "facing": math.pi, "dist": 0.95, "bearing": 0.04, "size_px": 30.0, "t": now}
        c.ingest({"ok": True, "t": now, "markers": [m, dict(m, id=7)]}, now)
        self.assertTrue(c.service_ok)
        obs = c(now + 0.2)
        self.assertEqual([o.id for o in obs], [23])                            # only the configured marker
        self.assertAlmostEqual(obs[0].x, 1.2)
        self.assertEqual(c(now + KEEP_S + 0.1), [])                            # dropped after KEEP_S
        c.ingest({"ok": True, "t": now - 5.0, "markers": [m]}, now)            # a service state from the past is not a live camera
        self.assertFalse(c.service_ok)

    def test_relay_reports_whether_the_marker_is_seen(self):
        from pi.marker_client import MarkerClient
        from pi.relay_assists import RelayAssists
        from pi.wifi_drive_safety import TUNING, WIRE_MOTOR_REVERSED
        a = RelayAssists(TUNING, WIRE_MOTOR_REVERSED, plan_mode="inline")
        self.assertFalse(a.camera_status()["service"])                         # no source: no camera
        c = MarkerClient(marker_id=23, start=False)
        a.marker_source, a.camera_facing = c, "front"
        self.assertFalse(a.camera_status()["service"])                         # a source that has not answered
        now = time.time()
        m = {"id": 23, "x": 1.2, "y": 0.05, "z": 0.1, "facing": math.pi, "dist": 0.95, "bearing": 0.04, "size_px": 30.0, "t": now}
        c.ingest({"ok": True, "t": now, "markers": []}, now)
        st = a.camera_status()
        self.assertTrue(st["service"])
        self.assertFalse(st["seen"])
        c.ingest({"ok": True, "t": now, "markers": [m]}, now)
        st = a.camera_status()
        self.assertTrue(st["seen"])
        self.assertAlmostEqual(st["dist"], 0.95, delta=0.01)
        self.assertEqual(st["facing"], "front")


class NoseInThroughTheRelayTest(unittest.TestCase):
    def test_front_camera_bay_run_in_the_twin(self):
        """The whole chain (twin camera -> RelayAssists.park_marker -> plan service -> AutoNav -> brake gate) parks nose in."""
        import sim.parking_validation as V
        name, group, p = [c for c in V.perpendicular_cases() if c[0] == "front_nominal"][0]
        r = V.run_case((name, group, p, 0, False))
        self.assertEqual(r["success"], 1, r)
        self.assertLess(r["pos_err_cm"], 5.0)
        self.assertLess(r["head_err_deg"], 5.0)
        self.assertEqual(int(r.get("collisions") or 0), 0)

    def test_hidden_marker_is_refused_not_guessed(self):
        import sim.parking_validation as V
        name, group, p = [c for c in V.perpendicular_cases() if c[0] == "front_dist3.0"][0]
        r = V.run_case((name, group, p, 0, False))
        self.assertEqual(r["success"], 1, r)                                   # (expect "refuse": it refused)
        self.assertIsNone(r["pos_err_cm"])


class PreflightTest(unittest.TestCase):
    """tools/preflight.py (AN5): the presenter's check says what is ready and what is not."""
    GOOD = {"health": {"state": "normal", "causes": [], "lidar_hz": 11.5, "temp_c": 58.0}, "esp32": {"state": "ok", "port": "/dev/ttyUSB0", "reboots": 1},
            "front_dist": 0.8, "rear_dist": 1.2,
            "camera": {"facing": "front", "marker_id": 23, "service": True, "seen": True, "dist": 1.2, "bearing_deg": 2.0}}

    def levels(self, st, cam=None, panel=True):
        from tools.preflight import evaluate
        return {name: lvl for lvl, name, _ in evaluate(st, cam, panel)}

    def test_a_ready_car(self):
        lv = self.levels(self.GOOD, {"ok": True, "fps": 12, "pipeline_ms": 9})
        self.assertNotIn("FAIL", lv.values())
        self.assertNotIn("WARN", lv.values())

    def test_no_state_is_a_failure(self):
        self.assertEqual(self.levels(None), {"relay": "FAIL"})

    def test_lidar_fault_says_what_to_check(self):
        from tools.preflight import evaluate
        st = dict(self.GOOD, health={"state": "fault", "causes": ["no LiDAR scans"], "lidar_hz": 0.0})
        rows = evaluate(st)
        health = [r for r in rows if r[1] == "health"][0]
        self.assertEqual(health[0], "FAIL")
        self.assertIn("USB cable", health[2])
        self.assertIn("power", health[2])

    def test_marker_not_in_view_and_camera_off_are_warnings_not_failures(self):
        st = dict(self.GOOD, camera=dict(self.GOOD["camera"], seen=False))
        self.assertEqual(self.levels(st)["camera"], "WARN")
        st = dict(self.GOOD, camera={"facing": "front", "marker_id": 23, "service": False, "seen": False})
        self.assertEqual(self.levels(st)["camera"], "WARN")

    def test_esp32_reboots_and_a_dead_esp32(self):
        st = dict(self.GOOD, esp32={"state": "ok", "port": "/dev/ttyUSB0", "reboots": 3})
        self.assertEqual(self.levels(st)["esp32"], "WARN")
        st = dict(self.GOOD, esp32={"state": "silent", "detail": "no PONG"})
        self.assertEqual(self.levels(st)["esp32"], "FAIL")

    def test_a_hot_pi(self):
        st = dict(self.GOOD, health=dict(self.GOOD["health"], temp_c=76.0))
        self.assertEqual(self.levels(st)["pi temperature"], "WARN")


class LidarWatchdogTest(unittest.TestCase):
    def test_relay_exits_when_the_lidar_goes_silent(self):
        """A LiDAR that drops off USB leaves the relay in FAULT for ever; it must exit (non-zero) so that systemd restarts it and the port search
        runs again (TODO AM2 / AN5). Twin relay, LiDAR silent after 3 s: the exit comes ~8 s after health says 'no LiDAR scans'."""
        import subprocess
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env = dict(os.environ, RC_LOG_DIR=tempfile.mkdtemp(), PYTHONUNBUFFERED="1")
        p = subprocess.Popen([sys.executable, os.path.join(root, "tools", "sim_car.py"), "--world", "room", "--lidar-dies-after", "3"], cwd=root,
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            out, _ = p.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
            self.fail("the relay did not exit after the LiDAR went silent: " + out[-600:])
        self.assertEqual(p.returncode, 1, out[-600:])
        self.assertIn("no LiDAR scans for 8 s", out)
        self.assertIn("motor stopped", out)                                   # through the normal shutdown path


if __name__ == "__main__":
    unittest.main()
