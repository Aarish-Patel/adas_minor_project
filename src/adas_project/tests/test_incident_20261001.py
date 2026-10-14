"""Fixes from the 1 Oct 2026 wall hit (docs/TODO.md section AH):
- the relay acts on the newest driving packet only; queued stale ones are dropped, control commands kept (pi/driver_input.py);
- no LiDAR returns in the direction of travel = unknown, not free: creep, then stop (pi/path_gate.py BLIND_*)."""
import math
import os
import socket
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pi.driver_input import drain, is_driving, latest_driving, small_buffer


class DriverInputTest(unittest.TestCase):
    def test_driving_packets(self):
        self.assertTrue(is_driving(b"A 70.0 80.0\nM -120\n"))
        self.assertTrue(is_driving(b"M 0"))
        self.assertFalse(is_driving(b"PING"))
        self.assertFalse(is_driving(b"GOTO 1.0 0.5"))
        self.assertFalse(is_driving(b"ASSIST evasive ON"))
        self.assertFalse(is_driving(b""))

    def test_only_the_newest_driving_packet_is_kept(self):
        a = ("127.0.0.1", 1)
        batch = [(f"A 75 75\nM {-i}".encode(), a) for i in range(5)] + [(b"PING", a)] + \
                [(f"A 75 75\nM {-i}".encode(), a) for i in range(5, 30)] + [(b"GOTO CANCEL", a)]
        keep, dropped = latest_driving(batch)
        self.assertEqual([d for d, _ in keep], [b"PING", b"A 75 75\nM -29", b"GOTO CANCEL"])
        self.assertEqual(dropped, 29)

    def test_a_queued_socket_is_drained_to_the_newest_command(self):
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        small_buffer(rx)
        rx.bind(("127.0.0.1", 0))
        rx.settimeout(0.05)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i in range(40):                                  # a slow tick: 40 packets (0.8 s at 50 Hz) queue up
            tx.sendto(f"A 75 75\nM {-100 - i}".encode(), rx.getsockname())
        tx.sendto(b"PING", rx.getsockname())
        time.sleep(0.1)
        first = rx.recvfrom(4096)
        keep, dropped = latest_driving([first] + drain(rx))
        self.assertEqual(keep[-1][0], b"PING")
        self.assertEqual(keep[0][0], b"A 75 75\nM -139")   # the newest command, not the oldest
        self.assertEqual(len(keep), 2)
        self.assertEqual(rx.gettimeout(), 0.05)             # the wait for the next tick is unchanged
        rx.close(); tx.close()


class BlindSectorTest(unittest.TestCase):
    def setUp(self):
        from tests.test_planning_safety import P, TUN
        from pi.path_gate import PathGate
        self.P, self.gate = P, PathGate(P, TUN.speed_model)

    def ring(self, r=2.5, skip=None):
        """A room seen all round at r m from the LiDAR, optionally with no returns in the bearing range skip (deg, ccw+)."""
        b = np.arange(-180.0, 180.0, 0.4)
        if skip is not None:
            b = b[~((b >= skip[0]) & (b <= skip[1]))]
        return np.column_stack([self.P.lidar_x + r * np.cos(np.radians(b)), r * np.sin(np.radians(b))])

    def test_full_view_is_not_limited(self):
        from pi.path_gate import sector_cover
        pts = self.ring()
        self.assertEqual(sector_cover(pts, self.P.lidar_x), {1: 1.0, -1: 1.0})
        self.gate.on_scan(pts, 1)
        out, brk = self.gate.decide(0.05, 150, 0.0, 0.4)
        self.assertEqual(out, 150)

    def test_blind_ahead_creeps_then_stops(self):
        pts = self.ring(skip=(-100.0, 100.0))              # the incident: nothing returned across the front
        self.gate.on_scan(pts, 1)
        out, brk = self.gate.decide(0.05, 254, 0.0, 0.2)
        cap = self.gate.model.pwm_for_speed(0.10)
        self.assertLessEqual(out, math.ceil(cap))           # creep
        self.assertGreater(out, 0)
        self.assertIn("blind", self.gate.info)
        t = 0.05
        while t < 1.0:
            out, brk = self.gate.decide(0.05, 254, 0.0, 0.1)
            t += 0.05
        self.assertEqual(out, 0)                             # after 1 s blind: stopped
        self.assertIn("LiDAR blind", self.gate.info.get("action", ""))
        out, _ = self.gate.decide(0.05, -150, 0.0, 0.0)      # reversing away: the rear is seen, allowed
        self.assertEqual(out, -150)
        self.gate.decide(0.05, 0, 0.0, 0.0)                  # the driver lets go: the timer restarts
        self.assertEqual(self.gate.blind_t, 0.0)

    def test_sees_again_releases(self):
        self.gate.on_scan(self.ring(skip=(-100.0, 100.0)), 1)
        for _ in range(25):
            out, _ = self.gate.decide(0.05, 200, 0.0, 0.1)
        self.assertEqual(out, 0)
        self.gate.on_scan(self.ring(), 2)
        out, _ = self.gate.decide(0.05, 200, 0.0, 0.0)
        self.assertEqual(out, 200)

    def test_blind_behind_limits_reverse(self):
        pts = self.ring()
        b = np.degrees(np.arctan2(pts[:, 1], pts[:, 0] - self.P.lidar_x))
        pts = pts[np.abs(b) < 140]                           # nothing returned within 40 deg of straight behind
        self.gate.on_scan(pts, 2)
        out, _ = self.gate.decide(0.05, -200, 0.0, -0.2)
        self.assertGreater(out, -200)
        self.assertLess(out, 0)

    def test_nosing_up_to_a_wall_seen_before_is_not_blind(self):
        from pi.path_gate import BLIND_COVER
        ring = self.ring()
        b = np.degrees(np.arctan2(ring[:, 1], ring[:, 0] - self.P.lidar_x))
        front = np.abs(b) < 60
        wall_x = self.P.lidar_x + 0.45
        near = np.column_stack([np.full(40, wall_x), np.linspace(-0.3, 0.3, 40)])       # a wall 0.45 m ahead of the LiDAR
        self.gate.on_scan(np.vstack([ring[~front], near]), 1)
        self.gate.decide(0.4, 60, 0.0, 0.5)                                              # 0.2 m closer: now 0.25 m, too close to see
        self.gate.on_scan(ring[~front], 2)                                               # nothing returned ahead any more
        self.assertGreaterEqual(self.gate.cover[1], BLIND_COVER)                         # the remembered wall covers it
        self.assertNotIn("blind", self.gate.info)

    def test_a_narrow_gap_is_not_blind(self):
        from pi.path_gate import BLIND_COVER, sector_cover
        cov = sector_cover(self.ring(skip=(-10.0, 10.0)), self.P.lidar_x)   # a 20 deg dark patch straight ahead
        self.assertGreaterEqual(cov[1], BLIND_COVER)


if __name__ == "__main__":
    unittest.main()
