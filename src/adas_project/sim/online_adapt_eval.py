"""Prequential evaluation of on-the-go recalibration of the crash predictor (adas/online_calibration.py, TODO W6).

    python -m sim.online_adapt_eval      # writes models/online_adapt.json

A 'session' is one driver on one car: the twin's held-out drives of one driver style, in random order, played tick by tick. At every
tick the model's risk is recorded (static) together with the recalibrated one (online, using ONLY feedback the car could observe:
safety events, 2 s after the prediction), then the calibrator is updated - so every number is a genuine prediction made before its
outcome was known. Two conditions: the nominal twin the model was trained for, and cars from 1.6x wider ranges (unseen).
Reported over the second half of each session: expected calibration error (10 bins), Brier score, log-loss.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def ece(p, y, bins=10):
    e, n = 0.0, len(p)
    for k in range(bins):
        m = (p >= k / bins) & (p < (k + 1) / bins + (k == bins - 1))
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def metrics(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"ece": ece(p, y), "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}


def session(d, p_all, runs, cal_kw=None, seed=0):
    from adas.online_calibration import OnlineCalibrator
    rng = np.random.default_rng(seed)
    order = rng.permutation(runs)
    cal = OnlineCalibrator(**(cal_kw or {}))
    static, adapted, ys = [], [], []
    clock = 0.0
    events = []
    for r in order:
        rows = np.where(d["run"] == r)[0]
        rows = rows[np.argsort(d["tick"][rows])]
        for k, i in enumerate(rows):
            t = clock + d["tick"][i] * 0.05
            while events and events[0] <= t:                     # a safety event the car has now experienced
                cal.event(events.pop(0))
            p = float(np.clip(p_all[i], 1e-5, 1 - 1e-5))
            x = math.log(p / (1 - p))
            static.append(p)
            adapted.append(cal.probability(x))
            ys.append(float(d["y"][i]))
            if d["y"][i] and d["tte"][i] < 0.06:                 # the contact / near miss happens now
                events.append(t + float(d["tte"][i]))
            if k % 5 == 0:
                cal.observe(t, x)
        clock += (d["tick"][rows[-1]] + 1) * 0.05 + 1.0
    n = len(ys)
    h = n // 2
    y = np.array(ys)
    return metrics(np.array(static)[h:], y[h:]), metrics(np.array(adapted)[h:], y[h:]), cal


def main():
    from adas.intent_net import IntentNet
    from sim.conformal_intent import risks
    from sim.train_intent_torch import load
    net = IntentNet(os.path.join(HERE, "..", "models", "intent_v3.json"))
    out = {}
    for name in ("test", "test_wide"):
        d = load(name)
        p = risks(d, net)
        styles = sorted(set(d["run_style"].tolist()))
        rows = []
        print(f"\n{name}: {len(d['run_family'])} drives")
        print(f"{'driver style':>14} {'runs':>5} | {'static ECE / Brier / logloss':>32} | {'online ECE / Brier / logloss':>32}")
        for sty in styles:
            runs = np.where(d["run_style"] == sty)[0]
            if len(runs) < 12:
                continue
            res = [session(d, p, runs, seed=s) for s in range(5)]
            st = {k: float(np.mean([r[0][k] for r in res])) for k in res[0][0]}
            ad = {k: float(np.mean([r[1][k] for r in res])) for k in res[0][1]}
            rows.append({"style": sty, "runs": int(len(runs)), "static": st, "online": ad})
            print(f"{sty:>14} {len(runs):5d} | {st['ece']:9.3f} {st['brier']:9.3f} {st['logloss']:9.3f}    | {ad['ece']:9.3f} {ad['brier']:9.3f} {ad['logloss']:9.3f}", flush=True)
        mean = lambda key, which: float(np.mean([r[which][key] for r in rows]))
        out[name] = {"styles": rows, "mean_static": {k: mean(k, "static") for k in ("ece", "brier", "logloss")},
                     "mean_online": {k: mean(k, "online") for k in ("ece", "brier", "logloss")}}
        print(f"{'mean':>14}       | {out[name]['mean_static']['ece']:9.3f} {out[name]['mean_static']['brier']:9.3f} {out[name]['mean_static']['logloss']:9.3f}    | "
              f"{out[name]['mean_online']['ece']:9.3f} {out[name]['mean_online']['brier']:9.3f} {out[name]['mean_online']['logloss']:9.3f}")
    json.dump(out, open(os.path.join(HERE, "..", "models", "online_adapt.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
