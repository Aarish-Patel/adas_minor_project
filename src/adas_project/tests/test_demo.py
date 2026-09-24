"""The demo acceptance suite (sim/demo_tests.py) must pass on the real-car profile."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim import demo_tests as D  # noqa: E402


class Demo(unittest.TestCase):
    def test_each_acceptance_test_passes(self):
        for t in (D.test_braking, D.test_rollout, D.test_failsafe, D.test_follow, D.test_bypass):
            r = t()
            self.assertTrue(r["pass"], f"{r['name']}: {r['detail']}")


if __name__ == "__main__":
    unittest.main()
