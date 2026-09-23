"""Whole-project checks that need no car: the existing 14-scenario self-test, the control panel's
parameter schema matches what the scripts really accept, and every Python file compiles."""
import glob
import os
import py_compile
import subprocess
import sys
import unittest
from dataclasses import fields

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)


class Selftest(unittest.TestCase):
    def test_sim_selftest_all_ok(self):
        r = subprocess.run([sys.executable, "-m", "sim.selftest"], cwd=ROOT, capture_output=True, text=True, timeout=300)
        self.assertIn("14/14 ok", r.stdout + r.stderr)


class Panel(unittest.TestCase):
    def test_oa_fields_are_all_wired(self):
        import pi.control_panel as C
        src = open(os.path.join(ROOT, "pi", "bypass_run.py"), encoding="utf-8").read()
        keys = [f[0] for f in C.OA_FIELDS]
        self.assertEqual(sorted(keys), sorted(C.OA_DEFAULTS))
        for k in keys:
            self.assertIn(f'"{k}":', src, f"panel field '{k}' is not read by bypass_run.py")
        for k, _, lo, hi, *_ in C.OA_FIELDS:
            self.assertTrue(lo <= C.OA_DEFAULTS[k] <= hi, k)

    def test_oa_defaults_match_controller_constants(self):
        import pi.control_panel as C
        import pi.bypass_core as B
        core = {"min_detect_dist_m": B.MIN_DETECT_DIST_M, "look_m": B.LOOK_M, "half_path": B.HALF_PATH, "clear_m": B.CLEAR_M,
                "min_gap_m": B.MIN_GAP_M, "l_h": B.L_H, "tau_s": B.TAU_S, "max_servo_deg": B.MAX_SERVO_DEG,
                "extra_straight_m": B.EXTRA_STRAIGHT_M}
        for k, v in core.items():
            self.assertAlmostEqual(C.OA_DEFAULTS[k], v, msg=k)

    def test_safety_fields_exist_in_aeb_config(self):
        import pi.control_panel as C
        from adas.aeb import AEBConfig
        names = {f.name for f in fields(AEBConfig)}
        for f in C.SAFETY_FIELDS:
            self.assertIn(f[0], names)

    def test_every_mode_script_exists(self):
        import pi.control_panel as C
        for m, d in C.MODES.items():
            if d["script"]:
                self.assertTrue(os.path.exists(os.path.join(ROOT, d["script"])), m)


class Compiles(unittest.TestCase):
    def test_all_python_compiles(self):
        for f in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
            if "__pycache__" in f:
                continue
            py_compile.compile(f, doraise=True)


if __name__ == "__main__":
    unittest.main()
