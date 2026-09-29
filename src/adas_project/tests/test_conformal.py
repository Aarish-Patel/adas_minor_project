"""Conformal risk control (adas/conformal.py): the guarantee holds on synthetic data, and the threshold behaves sensibly."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.conformal import conformal_risk_threshold, late_warning_loss


class ConformalTest(unittest.TestCase):
    def test_expected_risk_is_below_alpha(self):
        # each drive's 'danger level' u ~ U(0,1); it is warned late at threshold lam iff lam > u -> loss = 1{lam > u}
        rng = np.random.default_rng(1)
        lams = np.round(np.arange(0.0, 1.001, 0.01), 2)
        alpha, n, trials, tot = 0.15, 60, 400, 0.0
        for _ in range(trials):
            cal = rng.random(n)
            lam = conformal_risk_threshold((lams[None, :] > cal[:, None]).astype(float), lams, alpha)
            test = rng.random(2000)
            tot += float((lam > test).mean())
        self.assertLessEqual(tot / trials, alpha + 0.01)

    def test_none_when_too_few_drives(self):
        lams = np.array([0.1, 0.5, 0.9])
        L = np.ones((3, 3))
        self.assertIsNone(conformal_risk_threshold(L, lams, 0.1))

    def test_late_warning_loss(self):
        dt = 0.05
        # a drive whose risk is 0.8 for the last 2 s: warned early enough for lambda <= 0.8
        r1 = np.concatenate([np.full(20, 0.05), np.full(40, 0.8)])
        # a drive whose risk only rises in the last 0.5 s: late for any lambda above its floor
        r2 = np.concatenate([np.full(50, 0.05), np.full(10, 0.9)])
        L = late_warning_loss([r1, r2], [len(r1) - 1, len(r2) - 1], np.array([0.1, 0.5, 0.85]), dt)
        self.assertEqual(L[0].tolist(), [0, 0, 1])
        self.assertEqual(L[1].tolist(), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
