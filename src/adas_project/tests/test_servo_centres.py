"""Servos with different straight-ahead angles (1 Oct refit: 70 on output 1, 80 on output 2): the relay's one-number steering is
centred on their mean and its own steering lines keep each servo around its own centre (adas/servo.py, pi/relay_assists.py)."""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.config import load_tuning
from adas.servo import output_offsets, steering_centre

TUNING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pi", "tuning_real_car.json")


def refit(tun):
    t = copy.deepcopy(tun)
    s = t.servo
    s.left_channel, s.right_channel = 1, 2
    s.left_right, s.left_center, s.left_left = 13.0, 70.0, 123.0
    s.right_right, s.right_center, s.right_left = 23.0, 80.0, 133.0
    return t


class ServoCentreTest(unittest.TestCase):
    def test_centre_and_offsets(self):
        s = refit(load_tuning(TUNING)).servo
        self.assertEqual(steering_centre(s), 75.0)
        self.assertEqual(output_offsets(s), (-5.0, 5.0))                  # output 1 = left_* (70), output 2 = right_* (80)
        s.left_channel, s.right_channel = 2, 1
        self.assertEqual(output_offsets(s), (5.0, -5.0))

    def test_equal_centres_give_the_old_lines(self):
        from pi.relay_assists import RelayAssists
        a = RelayAssists(load_tuning(TUNING))
        c = a.centre
        self.assertEqual(a.a_line(100), "A 100 100")                     # unchanged format for the twin / equal centres
        self.assertEqual(a.a_line(99.6, ".0f"), "A 100 100")
        self.assertEqual(a.clamp_servo(c - 80), c - 52)
        self.assertEqual(a.clamp_servo(c + 80), c + 58)

    def test_relay_lines_keep_each_servo_on_its_centre(self):
        from pi.relay_assists import RelayAssists
        a = RelayAssists(refit(load_tuning(TUNING)))
        self.assertEqual(a.centre, 75.0)
        self.assertEqual(a.a_line(75), "A 70.0 80.0")                    # straight ahead: both wheels straight
        self.assertEqual(a.a_line(95), "A 90.0 100.0")                   # 20 deg: both servos 20 deg from their centres
        lo, hi = a.clamp_servo(0), a.clamp_servo(200)
        self.assertEqual((lo - a.centre, hi - a.centre), (-52.0, 58.0))  # the hard clamp moves with the centre
        # the steering envelope is symmetric about the true straight-ahead: equal travel both ways at a standstill
        self.assertAlmostEqual(a.limit_servo(0, 0.0) - a.centre, -(a.limit_servo(200, 0.0) - a.centre), places=6)
        # a driver line the envelope has to cut comes back per output, around each servo's centre
        out = a.process(["A 20 30", "M 0"], [(0.0, 3.0)], 1, now=0.0)
        q = next(ln for ln in out if ln.startswith("A ")).split()
        self.assertAlmostEqual(float(q[2]) - float(q[1]), 10.0, places=6)


if __name__ == "__main__":
    unittest.main()
