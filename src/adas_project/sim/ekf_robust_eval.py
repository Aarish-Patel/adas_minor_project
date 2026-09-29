"""Robust adaptive speed EKF vs the plain one (TODO N3): the same twin drive, with a fraction of the scan-matching
measurements replaced by bad ones that still report a small covariance.

    python -m sim.ekf_robust_eval          # writes models/ekf_robust.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from sim.odometry_eval import twin_run


def main():
    out = {}
    print("bad measurements that a covariance gate cannot see: a moderate speed error (mag) with a small covariance")
    print(f"{'mag (m/s)':>10} {'bad':>6} | {'plain EKF v / w RMSE':>24} | {'robust EKF v / w RMSE':>24}")
    for mag in ((0.05, 0.10), (0.10, 0.20), (0.10, 0.35)):
        for corrupt in (0.0, 0.10, 0.30):
            rows = {}
            for robust in (False, True):
                v, w = [], []
                for seed in (3, 4, 5):
                    res, _a, crashes = twin_run(seed=seed, robust=robust, corrupt=corrupt, mag=mag)
                    v.append(res["ekf"]["v"]["rmse"] * 100)
                    w.append(res["ekf"]["w"]["rmse"] * 57.2958)
                rows[robust] = (float(np.mean(v)), float(np.mean(w)))
            print(f"{mag[0]:.2f}-{mag[1]:.2f} {100 * corrupt:5.0f}% | {rows[False][0]:9.1f} cm/s {rows[False][1]:6.1f} deg/s | "
                  f"{rows[True][0]:9.1f} cm/s {rows[True][1]:6.1f} deg/s", flush=True)
            out[f"{mag}/{corrupt}"] = {"plain": rows[False], "robust": rows[True]}
    json.dump(out, open(os.path.join(HERE, "..", "models", "ekf_robust.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
