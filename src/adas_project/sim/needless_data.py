"""Dataset of takeover DECISIONS for the 'would this takeover be needless?' model (TODO W5).

    python -m sim.needless_data [seeds] [first_seed]        # writes data/needless/decisions.npz

Plain ADAS drives the Monte Carlo drives; at every takeover onset (evasive steer / steering correction) the car's own feature vector
(the v3 window features: stick, throttle, speed, free distance on five arcs, time to contact, stopping distances, reaction
overdue ...) is recorded together with the ground truth from the counterfactual: would the same driver, left alone for 2 s, have
come within 2 cm of anything (needed) or not (needless)? These are exactly the decisions the intent-aware system has to get right.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

STYLES = ("lapsing", "aggressive", "late", "good", "distracted")


def _job(a):
    os.environ["RC_SHADOW"] = "1"
    from sim import relay_mc
    r = relay_mc.run((a[0], "adas", a[1]))
    return [dict(row, seed=a[0], style=a[1]) for row in r["shadow"]]


def main(n_seeds=200, first=100):
    jobs = [(s, st) for s in range(first, first + n_seeds) for st in STYLES]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, out in enumerate(ex.map(_job, jobs, chunksize=2)):
            rows += out
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} drives, {len(rows)} takeovers", flush=True)
    X = np.array([r["x"] for r in rows], np.float32)
    y = np.array([r["needed"] for r in rows], bool)
    os.makedirs(os.path.join(HERE, "..", "data", "needless"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "..", "data", "needless", f"decisions_{first}.npz"), X=X, y=y,
                        p_v3=np.array([np.nan if r["p_v3"] is None else r["p_v3"] for r in rows], np.float32),
                        seed=np.array([r["seed"] for r in rows]), style=np.array([r["style"] for r in rows]),
                        lapsed=np.array([r["lapsed"] for r in rows]), ratio=np.array([np.nan if r["ratio"] is None else r["ratio"] for r in rows], np.float32),
                        t=np.array([r["t"] for r in rows], np.float32))
    print(f"{len(rows)} takeovers from {len(jobs)} drives: {int(y.sum())} needed ({100 * y.mean():.1f} %), {int((~y).sum())} needless")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200, int(sys.argv[2]) if len(sys.argv) > 2 else 100)
