"""Virtual bumper (adas/contact.py, adas/memory.add_bumper): pushing without moving while the gate does not brake = contact with something
the LiDAR cannot see (incident 1 Oct; the twin ROOM's dark box)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.contact import ContactDetector


class ContactTest(unittest.TestCase):
    def run_ticks(self, det, pwm, poses, braking=False, dt=0.05, scan_every=2):
        hits = []
        for i, p in enumerate(poses):
            hits.append(det.update(i * dt, pwm, p, braking, stamp=i // scan_every))
        return hits

    def test_pushing_without_moving_is_a_contact(self):
        det = ContactDetector(pwm_min=40)
        hits = self.run_ticks(det, 120, [(1.0, 0.0, 0.0)] * 20)              # 1 s of throttle, the scan pose does not move
        self.assertIn(1, hits)
        self.assertLess(hits.index(1) * 0.05, 0.8)
        det2 = ContactDetector(pwm_min=40)
        self.assertIn(-1, self.run_ticks(det2, -120, [(1.0, 0.0, 0.0)] * 20))

    def test_driving_is_not_a_contact(self):
        det = ContactDetector(pwm_min=40)
        self.assertNotIn(1, self.run_ticks(det, 120, [(0.01 * i, 0.0, 0.0) for i in range(40)]))   # 0.2 m/s

    def test_braking_dead_band_and_dead_reckoning_are_not_contacts(self):
        self.assertNotIn(1, self.run_ticks(ContactDetector(pwm_min=40), 120, [(1.0, 0.0, 0.0)] * 30, braking=True))
        self.assertNotIn(1, self.run_ticks(ContactDetector(pwm_min=40), 25, [(1.0, 0.0, 0.0)] * 30))        # below the dead-band
        # between scans the pose is dead-reckoned (it creeps with the command), the scans say it does not move: still a contact
        poses = [(1.0 + 0.012 * (i % 2), 0.0, 0.0) for i in range(30)]
        self.assertIn(1, self.run_ticks(ContactDetector(pwm_min=40), 120, poses, scan_every=2))

    def test_bumper_points_stop_the_gate(self):
        from tests.test_planning_safety import P, TUN, wall
        from pi.path_gate import PathGate
        g = PathGate(P, TUN.speed_model)
        room = np.vstack([wall(4.0, -3, 4.0, 3), wall(-4.0, -3, -4.0, 3), wall(-4, 3, 4, 3), wall(-4, -3, 4, -3)])
        g.on_scan(room, 1)
        out, _ = g.decide(0.05, 150, 0.0, 0.0)
        self.assertEqual(out, 150)                                             # nothing seen ahead: free
        g.memory.add_bumper(1)
        g.on_scan(room, 2)                                                     # the dark object still returns nothing
        out, _ = g.decide(0.05, 150, 0.0, 0.0)
        self.assertEqual(out, 0)                                               # the remembered contact holds the car
        out, _ = g.decide(0.05, -150, 0.0, 0.0)
        self.assertEqual(out, -150)                                            # backing away is allowed


class DarkBoxTwinTest(unittest.TestCase):
    """Twin (sim.relay_scenarios.run, the relay's own classes): full throttle straight at a box that returns no LiDAR points."""
    def drive(self, bumper=True):
        from sim.relay_scenarios import run, steady
        from sim.world import Box, World
        from sim.hw_worlds import room_walls
        w = World()
        room_walls(w, -1.0, 4.0, -1.5, 1.5)
        box = Box(1.6, 0.17, 0.30, 0.20)                                       # catches the front-left corner only
        box.dark = True
        w.add(box)
        import pi.path_gate as G
        saved = G.PathGate.decide
        if not bumper:
            G.PathGate.decide = G.PathGate._decide
        try:
            return run(w, steady(120), 8.0, after_contact=True)
        finally:
            G.PathGate.decide = saved

    def test_the_car_stops_pushing_into_an_invisible_box(self):
        on, off = self.drive(True), self.drive(False)
        tr_on, tr_off = on["trace"], off["trace"]
        pushing = lambda tr: sum(1 for r in tr if r[6] and r[6] > 30 and abs(r[4]) < 0.02)     # gate output > 30 PWM, not moving
        self.assertTrue(on["collided"])                                        # it cannot know the box before it touches it
        self.assertLess(pushing(tr_on), 0.5 * pushing(tr_off))                 # ... but it stops pushing into it
        self.assertLessEqual(on["crashes"], off["crashes"])


if __name__ == "__main__":
    unittest.main()
