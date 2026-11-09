"""The moving-object tracker in its own process (pi/tracker_proc.py, TODO AP1)."""
import math
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def scene(t, lidar_x=0.12):
    """Vehicle-frame points at time t: a wall 2 m ahead and a thin post crossing 1.2 m ahead at 0.4 m/s."""
    ys = np.arange(-1.2, 1.2, 0.02)
    pts = [(2.0, y) for y in ys]
    cx, cy = 1.2, -0.6 + 0.4 * t
    a = np.linspace(0, 2 * math.pi, 9, endpoint=False)
    pts = [p for p in pts if math.hypot(p[0] - cx, p[1] - cy) > 0.07] + [(cx + 0.05 * math.cos(k), cy + 0.05 * math.sin(k)) for k in a]
    return np.array(pts, float)


class SameResultsTest(unittest.TestCase):
    def test_process_gives_the_in_thread_tracks(self):
        from adas.tracking import Tracker
        from adas.vehicle_params import VehicleParams
        from pi.tracker_proc import TrackerProcess
        got = []
        tp = TrackerProcess(VehicleParams(), lambda *r: got.append(r))
        local = Tracker()
        want = []
        try:
            t0 = 1000.0
            for i in range(30):
                t = t0 + 0.1 * i
                xy = scene(0.1 * i)
                tp.submit(xy.copy(), t, 0.0, 0.0, 0.0, 1)                       # (each tracker gets its own copy of the scan)
                want.append([(tr.id, tr.moving, round(tr.pos[0], 3), round(tr.pos[1], 3)) for tr in local.update(xy.copy(), t, 0.0, 0.0)])   # (a snapshot:
                                                                                              # the tracker updates its Track objects in place)
                deadline = time.time() + 15.0                                  # lockstep: the car's relay may drop scans, the test must not
                while len(got) < i + 1 and time.time() < deadline:
                    time.sleep(0.01)
        finally:
            tp.stop()
        self.assertEqual(len(got), 30, f"only {len(got)} results")
        by_t = {round(r[0], 3): r for r in got}
        checked = 0
        for i in range(30):
            r = by_t.get(round(1000.0 + 0.1 * i, 3))
            self.assertIsNotNone(r, i)
            self.assertEqual([(d["id"], d["moving"], d["x"], d["y"]) for d in r[1]], want[i], i)
            checked += 1
        self.assertEqual(checked, 30)
        self.assertTrue(any(d["moving"] for d in got[-1][1]), "the crossing post was never flagged as moving")

    def test_a_dead_worker_is_reported(self):
        from adas.vehicle_params import VehicleParams
        from pi.tracker_proc import TrackerProcess
        tp = TrackerProcess(VehicleParams(), lambda *r: None)
        tp.proc.terminate()
        tp.proc.join(timeout=3.0)
        self.assertFalse(tp.alive())
        self.assertFalse(tp.submit(scene(0), 1.0, 0.0, 0.0, 0.0, 1))           # the caller then tracks in its own thread
        tp.stop()


class FakeLidar:
    """Stands in for the RPLIDAR: `n` scans of scene() at 10 Hz in the relay's (quality, angle deg, distance mm) format."""

    def __init__(self, n, front_offset):
        self.n, self.front_offset = n, front_offset

    def get_info(self):
        return {"driver": "fake"}

    def get_health(self):
        return ("Good", 0)

    def iter_scans(self, max_buf_meas=None, min_len=5):
        from pi.wifi_drive_safety import VP
        for i in range(self.n):
            xy = scene(0.1 * i)
            dx, dy = xy[:, 0] - VP.lidar_x, xy[:, 1]
            ang = (self.front_offset - np.degrees(np.arctan2(dy, dx))) % 360.0           # the sensor turns clockwise
            dist = np.hypot(dx, dy) * 1000.0
            yield [(47, float(a), float(d)) for a, d in zip(ang, dist)]
            time.sleep(0.1)

    def stop(self):
        pass

    def stop_motor(self):
        pass

    def disconnect(self):
        pass


class ClearanceServesTracksTest(unittest.TestCase):
    def run_clearance(self, inline):
        import pi.wifi_drive_safety as R
        old_open, old_env = R.open_lidar, os.environ.get("RC_TRACKER_INLINE")
        R.open_lidar = lambda port, *a, **k: FakeLidar(40, R.FRONT_OFFSET_DEG)
        os.environ["RC_TRACKER_INLINE"] = "1" if inline else "0"
        clr = R.Clearance("fake")
        try:
            clr.start()
            time.sleep(3.5)
            raw = clr.read_raw_tracks()
            info, contact = clr.read_tracks()
            t_mid = (len(raw), len(info), any(d["moving"] for d in info))
            seq = clr.read_seq()
            time.sleep(R.TRACKS_STALE_S + 4.5)                                 # the fake LiDAR ends after ~4 s: the tracks go stale
            stale = (clr.read_raw_tracks(), clr.read_tracks())
            return t_mid, seq, stale, clr
        finally:
            clr.stop()
            R.open_lidar = old_open
            if old_env is None:
                os.environ.pop("RC_TRACKER_INLINE", None)
            else:
                os.environ["RC_TRACKER_INLINE"] = old_env

    def test_tracks_are_served_from_the_process_and_go_stale_when_scans_stop(self):
        (n_raw, n_info, moving), seq, stale, clr = self.run_clearance(inline=False)
        self.assertIsNotNone(clr.tracker_proc)
        self.assertGreater(seq, 20)
        self.assertGreater(n_raw, 0)
        self.assertEqual(n_raw, n_info)
        self.assertTrue(moving, "the crossing post was not flagged as moving")
        self.assertEqual(stale[0], [])                                          # not served late
        self.assertEqual(stale[1][0], [])

    def test_the_in_thread_fallback_still_works(self):
        (n_raw, _, moving), seq, stale, clr = self.run_clearance(inline=True)
        self.assertIsNone(clr.tracker_proc)
        self.assertGreater(n_raw, 0)
        self.assertTrue(moving)


if __name__ == "__main__":
    unittest.main()
