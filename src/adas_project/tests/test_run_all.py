"""The 'Run everything' sequencer and the control panel's Apply logic, against fake scripts and a temporary tuning file
(no car, no Pi)."""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import pi.control_panel as C  # noqa: E402
import pi.run_all as R  # noqa: E402

WRITE_RESULT = '''
import json, os, time
p = {results!r}
try:
    d = json.load(open(p))
except Exception:
    d = {{}}
d[{key!r}] = {{"value": 1, "time": time.strftime("%Y-%m-%d %H:%M:%S")}}
json.dump(d, open(p, "w"))
'''


class RunAll(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "pi"))
        self.saved = (R.ROOT, R.STATE, R.CONTINUE, R.RESULTS, R.REPORT)
        R.ROOT = self.tmp
        R.STATE = os.path.join(self.tmp, "pi", "state.json")
        R.CONTINUE = os.path.join(self.tmp, "pi", "continue.flag")
        R.RESULTS = os.path.join(self.tmp, "pi", "results.json")
        R.REPORT = os.path.join(self.tmp, "pi", "report.json")
        R.state.update(step=0, log=[], results=[], waiting=False, prompt="", status="")

    def tearDown(self):
        R.ROOT, R.STATE, R.CONTINUE, R.RESULTS, R.REPORT = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def script(self, name, body):
        with open(os.path.join(self.tmp, "pi", name), "w") as f:
            f.write(body)
        return "pi/" + name

    def press_continue_when_waiting(self):
        def go():
            for _ in range(200):
                if R.state.get("waiting"):
                    open(R.CONTINUE, "w").close()
                    return
                time.sleep(0.05)
        t = threading.Thread(target=go, daemon=True)
        t.start()
        return t

    def test_happy_path_pauses_and_collects_results(self):
        a = self.script("a.py", WRITE_RESULT.format(results=R.RESULTS, key="lidar"))
        b = self.script("b.py", "print('demo ok')")
        steps = [("wait", "setup", None, None, 0, "put the object in front"),
                 ("run", "lidar", a, "lidar", 20, ""), ("run", "demo", b, None, 20, "")]
        t = self.press_continue_when_waiting()
        self.assertTrue(R.run_steps(steps))
        t.join()
        self.assertEqual([r["ok"] for r in R.state["results"]], [True, True])
        self.assertIn("finished", R.state["status"])

    def test_stops_at_first_failure(self):
        bad = self.script("bad.py", "import sys; print('boom'); sys.exit(3)")
        never = self.script("never.py", "open(%r, 'w').write('ran')" % os.path.join(self.tmp, "ran.txt"))
        steps = [("run", "bad", bad, None, 20, ""), ("run", "never", never, None, 20, "")]
        self.assertFalse(R.run_steps(steps))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "ran.txt")))
        self.assertIn("boom", R.state["results"][0]["message"])

    def test_calibration_without_fresh_result_fails(self):
        quiet = self.script("quiet.py", "print('did nothing')")
        self.assertFalse(R.run_steps([("run", "lidar", quiet, "lidar", 20, "")]))
        self.assertIn("no valid result", R.state["results"][0]["message"])

    def test_timeout_is_a_failure(self):
        slow = self.script("slow.py", "import time; time.sleep(30)")
        self.assertFalse(R.run_steps([("run", "slow", slow, None, 1, "")]))
        self.assertIn("timed out", R.state["results"][0]["message"])


class PanelApply(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.saved = (C.TUNING, C.PARAMS, C.RESULTS)
        C.TUNING = os.path.join(self.tmp, "tuning.json")
        C.PARAMS = os.path.join(self.tmp, "params.json")
        C.RESULTS = os.path.join(self.tmp, "results.json")
        shutil.copy(os.path.join(ROOT, "pi", "tuning_real_car.json"), C.TUNING)
        json.dump({"lidar": {"yaw_offset_deg": 91.7, "time": "x"}, "servo_center": {"servo_center": 86.94, "time": "x"},
                   "speed": {"v_max": 0.9123, "deadband": 47.26, "time": "x"},
                   "turn": {"k_curv_per_deg": 0.07123, "time": "x"}}, open(C.RESULTS, "w"))

    def tearDown(self):
        C.TUNING, C.PARAMS, C.RESULTS = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_apply_each_result(self):
        for kind in ("lidar", "servo_center", "speed", "turn"):
            self.assertNotIn("no such", C.apply_result(kind))
        tun = json.load(open(C.TUNING))
        self.assertEqual(tun["mount"]["yaw_offset_deg"], 91.7)
        self.assertEqual(tun["servo"]["left_center"], 86.9)
        self.assertEqual(tun["servo"]["right_center"], 86.9)
        self.assertEqual(tun["speed_model"], {"v_max": 0.912, "deadband": 47.3})
        self.assertEqual(json.load(open(C.PARAMS))["oa"]["k_curv_per_deg"], 0.0712)
        self.assertEqual(json.load(open(C.PARAMS))["oa"]["servo_straight"], 86.9)
        # every apply left a backup of the previous tuning
        self.assertGreaterEqual(len([f for f in os.listdir(self.tmp) if f.startswith("tuning.json.bak_")]), 1)

    def test_apply_unknown_result_changes_nothing(self):
        before = open(C.TUNING).read()
        self.assertIn("no such", C.apply_result("nothing"))
        self.assertEqual(open(C.TUNING).read(), before)

    def test_safety_settings_reach_tuning_and_ignore_unknown_keys(self):
        C.apply_safety({"margin": 0.12, "latency": 0.2, "evil": 5})
        tun = json.load(open(C.TUNING))
        self.assertEqual(tun["aeb"]["margin"], 0.12)
        self.assertEqual(tun["aeb"]["latency"], 0.2)
        self.assertNotIn("evil", tun["aeb"])


if __name__ == "__main__":
    unittest.main()
