"""Drive logging (pi/drive_log.py) and fitting the simulator to logs (sim/log_fit.py), without the car:
a synthetic log from a simulated car with KNOWN dynamics must be fitted back to those values."""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from pi.drive_log import DriveLog, LoggedSerial, load  # noqa: E402


class FakeSerial:
    def __init__(self):
        self.sent = b""

    def write(self, data):
        self.sent += data
        return len(data)

    def close(self):
        pass


class Recorder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_round_trip(self):
        log = DriveLog("unit", log_dir=self.tmp, enabled=True)
        ser = LoggedSerial(FakeSerial(), log)
        ser.write(b"A 87 87\nM -120\n")
        scan = [(15, 10.5, 1234.0), (15, 200.25, 800.0), (0, 30.0, 0.0)]    # the zero-distance return is dropped
        log.scan(scan)
        log.driver(inp="M -120", pwm_out=-120)
        log.event("stop", why="test")
        ser.close()
        log.close()
        self.assertEqual(ser._ser.sent, b"A 87 87\nM -120\n")             # commands are passed through unchanged
        r = load(log.path)
        self.assertEqual(r["meta"]["source"], "unit")
        self.assertEqual([c[1] for c in r["cmds"]], ["A 87 87", "M -120"])
        t, ang, dist, q = r["scans"][0]
        self.assertEqual(ang, [10.5, 200.25])
        self.assertEqual(dist, [1.234, 0.8])
        self.assertEqual(r["drv"][0]["pwm_out"], -120)
        self.assertEqual(r["ev"][0]["msg"], "stop")

    def test_parked_scans_are_thinned_moving_scans_are_not(self):
        log = DriveLog("unit", log_dir=self.tmp, enabled=True)
        scan = [(15, 0.0, 1000.0)]
        for _ in range(10):
            log.scan(scan)                     # parked: only the first of these is kept (1 per second)
        log.cmd("M -110")                      # motor commanded -> full rate
        for _ in range(10):
            log.scan(scan)
        log.close()
        self.assertEqual(len(load(log.path)["scans"]), 11)

    def test_disabled_writes_nothing(self):
        log = DriveLog("unit", log_dir=self.tmp, enabled=False)
        log.scan([(15, 0.0, 1000.0)])
        log.close()
        self.assertEqual(os.listdir(self.tmp), [])

    def test_truncated_log_still_loads(self):
        log = DriveLog("unit", log_dir=self.tmp, enabled=True)
        log.cmd("M -100")
        log.close()
        import gzip
        with gzip.open(log.path, "at") as f:
            f.write('{"k":"cmd","t":1,"li')           # power cut mid-line
        self.assertEqual(len(load(log.path)["cmds"]), 1)


class FitFromLog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sim.log_fit import fit_logs
        from sim.log_synth import generate
        cls.tmp = tempfile.mkdtemp()
        path = os.path.join(cls.tmp, "synth.jsonl.gz")
        generate(path)
        cls.r = fit_logs([path], out=None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_speed_model_recovered(self):
        sp, truth = self.r["speed"], self.r["truth"]
        self.assertAlmostEqual(sp["v_max"], truth["v_max"], delta=0.06)
        self.assertAlmostEqual(sp["deadband"], truth["deadband"], delta=6)
        self.assertLess(sp["rms_speed_error"], 0.05)

    def test_steering_recovered(self):
        st, truth = self.r["steer"], self.r["truth"]
        self.assertTrue(st["identifiable"])
        self.assertAlmostEqual(st["k_curv_per_deg"], truth["k_curv_per_deg"], delta=0.007)
        self.assertAlmostEqual(st["servo_centre"], truth["servo_centre"], delta=1.0)

    def test_fitted_model_predicts_the_path(self):
        v = self.r["validation"][0]
        self.assertGreater(v["windows"], 20)
        self.assertLess(v["median_pos_err_m"], 0.06)

    def test_synthetic_fit_is_never_used_as_the_real_car(self):
        from sim.real_car import load_fitted
        path = os.path.join(self.tmp, "fit.json")
        json.dump(self.r, open(path, "w"))
        self.assertIsNone(load_fitted(path))


if __name__ == "__main__":
    unittest.main()
