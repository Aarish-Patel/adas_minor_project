"""Health supervision and degraded modes (pi/health.py)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pi.health import CLEAR_S, LIMP_PWM, HealthMonitor


def drive(h, t0, seconds, scan_hz=10.0, pkt_hz=20.0, loop_ms=15.0, esp="ok", driving=True):
    """Feed the monitor 20 ticks a second of a relay that sees scans at scan_hz and packets at pkt_hz."""
    t, seq, next_scan, next_pkt = t0, h.last_seq or 0, t0, t0
    while t < t0 + seconds:
        if scan_hz and t >= next_scan:
            seq += 1
            next_scan += 1.0 / scan_hz
        pkt = pkt_hz and t >= next_pkt
        if pkt:
            next_pkt += 1.0 / pkt_hz
        h.tick(t, scan_seq=seq, loop_ms=loop_ms if pkt else None, driver_packet=bool(pkt), esp_state=esp,
               driving=driving)
        t += 0.05
    return t


class HealthTest(unittest.TestCase):
    def monitor(self, temp=None):
        path = os.path.join(self.tmp.name, "temp")
        if temp is not None:
            with open(path, "w") as f:
                f.write(str(int(temp * 1000)))
        return HealthMonitor(temp_path=path)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_normal(self):
        h = self.monitor(55)
        drive(h, 0.0, 5.0)
        self.assertEqual(h.state, "normal", h.causes)
        self.assertEqual(h.cap(200), 200)

    def test_lidar_lost_is_a_fault_and_holds_the_motor(self):
        h = self.monitor()
        t = drive(h, 0.0, 3.0)
        drive(h, t, 2.0, scan_hz=0)
        self.assertEqual(h.state, "fault")
        self.assertEqual(h.cap(180), 0.0)

    def test_slow_lidar_limps_then_recovers_after_hysteresis(self):
        h = self.monitor()
        t = drive(h, 0.0, 3.0)
        t = drive(h, t, 3.0, scan_hz=4.0)
        self.assertEqual(h.state, "limp")
        self.assertEqual(h.cap(250), LIMP_PWM)
        self.assertEqual(h.cap(-250), -LIMP_PWM)
        t = drive(h, t, 1.0)                           # healthy again, but not for long enough
        self.assertEqual(h.state, "limp")
        drive(h, t, CLEAR_S + 2.5)
        self.assertEqual(h.state, "normal", h.causes)

    def test_lossy_driver_link_limps_only_while_driving(self):
        h = self.monitor()
        t = drive(h, 0.0, 3.0)
        drive(h, t, 3.0, pkt_hz=8.0)
        self.assertEqual(h.state, "limp")
        h2 = self.monitor()
        t = drive(h2, 0.0, 3.0, driving=False)
        drive(h2, t, 3.0, pkt_hz=8.0, driving=False)
        self.assertEqual(h2.state, "normal")

    def test_slow_loop_hot_pi_and_silent_esp32(self):
        h = self.monitor()
        drive(h, 0.0, 3.0, loop_ms=90.0)
        self.assertIn("control loop slow", " ".join(h.causes))
        h = self.monitor(83)
        drive(h, 0.0, 3.0)
        self.assertEqual(h.state, "limp")
        self.assertIn("hot", " ".join(h.causes))
        h = self.monitor()
        drive(h, 0.0, 3.0, esp="silent")
        self.assertEqual(h.state, "fault")


class UndervoltageTest(unittest.TestCase):
    """The Pi's supply alarm (kernel hwmon rpi_volt in0_lcrit_alarm) is a named LIMP cause, kept for UNDERVOLT_HOLD_S."""

    def make(self, tmp, alarm):
        from pi.health import HealthMonitor
        d = os.path.join(tmp, "hwmon0")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "in0_lcrit_alarm")
        with open(path, "w") as f:
            f.write(alarm)
        return HealthMonitor(temp_path=os.path.join(tmp, "none"), volt_glob=os.path.join(tmp, "hwmon*", "in0_lcrit_alarm")), path

    def test_alarm_names_the_cause_and_clears_after_the_hold(self):
        from pi.health import UNDERVOLT_HOLD_S
        with tempfile.TemporaryDirectory() as tmp:
            h, path = self.make(tmp, "1")
            t = drive(h, 0.0, 4.0)
            self.assertEqual(h.state, "limp")
            self.assertTrue(any("under-voltage" in c for c in h.causes), h.causes)
            with open(path, "w") as f:
                f.write("0")
            t = drive(h, t, UNDERVOLT_HOLD_S + 4.0)
            self.assertEqual(h.state, "normal", h.causes)

    def test_no_alarm_no_cause(self):
        with tempfile.TemporaryDirectory() as tmp:
            h, _ = self.make(tmp, "0")
            drive(h, 0.0, 5.0)
            self.assertEqual(h.state, "normal", h.causes)

    def test_a_machine_without_the_sensor_is_unaffected(self):
        from pi.health import HealthMonitor
        h = HealthMonitor(temp_path="/nonexistent", volt_glob="/nonexistent/*/in0_lcrit_alarm")
        drive(h, 0.0, 5.0)
        self.assertEqual(h.state, "normal", h.causes)


if __name__ == "__main__":
    unittest.main()
