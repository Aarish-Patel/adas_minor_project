"""Time Hybrid A* on recorded planning jobs (pickled argument tuples of adas.plan_service.plan_line_job), on this
machine, with the car's budgets.   python3 tools/plan_bench.py jobs.pkl"""
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)


def main(path):
    from adas.plan_service import BACKOFF_BUDGET_S, EVASIVE_BUDGET_S, plan_line_job
    jobs = [tuple(np.asarray(x, float) if isinstance(x, list) else x for x in a) for a in pickle.load(open(path, "rb"))]
    for label, b1, b2 in (("no budget", None, None), ("car budgets", EVASIVE_BUDGET_S, BACKOFF_BUDGET_S)):
        ts, found = [], 0
        for a in jobs:
            t0 = time.perf_counter()
            r = plan_line_job(*a[:7], budget_s=b1, reverse_budget_s=b2)
            ts.append((time.perf_counter() - t0) * 1000)
            found += r is not None
        t = np.array(ts)
        print(f"{label:12s}: {len(jobs)} searches, found {found}; median {np.median(t):.0f} ms, 95th "
              f"{np.percentile(t, 95):.0f} ms, max {t.max():.0f} ms")


if __name__ == "__main__":
    main(sys.argv[1])
