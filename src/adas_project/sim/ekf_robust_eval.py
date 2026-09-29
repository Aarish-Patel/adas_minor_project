"""Robust adaptive speed EKF vs the plain one (TODO N3): the same twin drive, with a fraction of the scan-matching
measurements replaced by bad ones that still report a small covariance.

    python -m sim.ekf_robust_eval          # writes models/ekf_robust.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from sim.odometry_eval import twin_run


def main():
    out = {}
    print(f"{'bad measurements':>17} {'battery':>8} | {'plain EKF v / w RMSE':>24} | {'robust EKF v / w RMSE':>24}")
    for corrupt in (0.0, 0.05, 0.15, 0.30):
        for battery in (1.0, 0.8):
            rows = {}
            for robust in (False, True):
                res, _a, crashes = twin_run(seed=3, battery=battery, robust=robust, corrupt=corrupt)
                rows[robust] = (res["ekf"]["v"]["rmse"] * 100, res["ekf"]["w"]["rmse"] * 57.2958)
            print(f"{100 * corrupt:15.0f} % {battery:8.1f} | {rows[False][0]:9.1f} cm/s {rows[False][1]:6.1f} deg/s | "
                  f"{rows[True][0]:9.1f} cm/s {rows[True][1]:6.1f} deg/s")
            out[f"{corrupt}/{battery}"] = {"plain": rows[False], "robust": rows[True]}
    json.dump(out, open(os.path.join(HERE, "..", "models", "ekf_robust.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
