"""Conformal risk control of the driver-intent warning threshold (adas/conformal.py) on the twin's held-out drives.

    python -m sim.conformal_intent

Calibration drives: half of the crash drives of data/intent/test.npz (the nominal twin). Tests: the other half (same
distribution: the guarantee should hold) and data/intent/test_wide.npz (cars drawn from 1.6x wider ranges: shifted, the
guarantee is not promised - the point is to see how far it degrades). For each target alpha the chosen threshold, the
achieved late-warning rate and the price (fraction of safe drives that get a false alarm >= 0.25 s) are reported.
Writes models/conformal_intent.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from adas.conformal import conformal_risk_threshold, late_warning_loss
from adas.intent_net import IntentNet
from sim.train_intent_torch import DT, flat_all, load

LAMBDAS = np.round(np.arange(0.05, 0.96, 0.01), 2)


def risks(d, net):
    """Per-tick risk of every drive: sigmoid(logit) floored by the physics floor (what the car shows)."""
    rows = np.arange(len(d["y"]))
    X, F = flat_all(d, rows)
    raw = np.array([net._trees_raw(x.astype(np.float64)) for x in X]) / net.temperature
    return np.maximum(1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30))), F)


def per_run(d, p):
    crash, safe = [], []
    for r in range(len(d["run_family"])):
        m = d["run"] == r
        pr, yr = p[m], d["y"][m]
        if d["run_crashed"][r]:
            crash.append(pr)
        elif yr.sum() == 0:
            safe.append(pr)
    return crash, safe


def false_alarm_rate(safe, lam):
    hit = 0
    for pr in safe:
        run = np.convolve((pr >= lam).astype(int), np.ones(5, int), "valid") if len(pr) >= 5 else np.zeros(1)
        hit += int((run >= 5).any())
    return hit / max(len(safe), 1)


def main():
    net = IntentNet(os.path.join(HERE, "..", "models", "intent_v3.json"))
    sets = {}
    for name in ("test", "test_wide"):
        d = load(name)
        p = risks(d, net)
        crash, safe = per_run(d, p)
        sets[name] = (crash, safe)
        print(f"{name}: {len(crash)} crash drives, {len(safe)} safe drives")
    crash, safe = sets["test"]
    wide_crash, wide_safe = sets["test_wide"]

    def loss(runs):
        return late_warning_loss(runs, [len(r) - 1 for r in runs], LAMBDAS, DT)

    L_all, L_wide = loss(crash), loss(wide_crash)
    # the guarantee is over the random calibration draw: repeat it 300 times and report the mean and the spread
    rng = np.random.default_rng(0)
    out = {"lambdas": LAMBDAS.tolist(), "splits": 300, "rows": []}
    print(f"\n{len(crash)} crash drives split 300 times into calibration / held-out halves; loss = crash warned < 1 s "
          f"before contact\n")
    print(f"{'alpha':>6} {'lambda (median)':>16} | {'late, held-out (mean, 5-95 %)':>34} {'late, wide (shifted)':>21} | "
          f"{'false alarms, safe drives':>26}")
    for alpha in (0.10, 0.15, 0.20, 0.30):
        lams, held, wide = [], [], []
        for _ in range(300):
            order = rng.permutation(len(crash))
            cal, tst = order[:len(crash) // 2], order[len(crash) // 2:]
            lam = conformal_risk_threshold(L_all[cal], LAMBDAS, alpha)
            if lam is None:
                continue
            j = int(np.where(LAMBDAS == lam)[0][0])
            lams.append(lam)
            held.append(L_all[tst, j].mean())
            wide.append(L_wide[:, j].mean())
        if not lams:
            print(f"{alpha:6.2f}  n/a (too few calibration drives)")
            continue
        lam = float(np.median(lams))
        fa = false_alarm_rate(safe, lam)
        fa_w = false_alarm_rate(wide_safe, lam)
        lo, hi = np.percentile(held, [5, 95])
        print(f"{alpha:6.2f} {lam:16.2f} | {np.mean(held):15.3f}   ({lo:.3f} - {hi:.3f}) {np.mean(wide):21.3f} | "
              f"{fa:12.3f} (wide {fa_w:.3f})")
        out["rows"].append({"alpha": alpha, "lambda_median": lam, "late_held_out_mean": float(np.mean(held)),
                            "late_held_out_p5": float(lo), "late_held_out_p95": float(hi),
                            "late_wide_mean": float(np.mean(wide)), "false_alarm_safe": fa, "false_alarm_safe_wide": fa_w})
    # the floor: drives with less than 1 s of history before contact can never be warned 1 s early
    out["unwarnable_1s_fraction"] = float(np.mean(L_all[:, 0] == 1.0))
    print(f"\ncrash drives that cannot be warned 1 s early by anyone (the danger appears < 1 s before contact): "
          f"{100 * out['unwarnable_1s_fraction']:.1f} %")
    json.dump(out, open(os.path.join(HERE, "..", "models", "conformal_intent.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
