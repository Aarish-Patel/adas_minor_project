"""Nose-in bay parking with the front camera (TODO AN1): runs the `front_*` cases of sim/parking_validation.py.

    python tools/front_bay_validation.py            # the deterministic cases
    python tools/front_bay_validation.py mc 6       # Monte Carlo: 6 seeds per case (randomised start, camera and LiDAR noise)

Prints one line per case and the totals; the numbers go to docs/TODO.md AN1.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)


def main():
    import sim.parking_validation as V
    mc = len(sys.argv) > 1 and sys.argv[1] == "mc"
    seeds = int(sys.argv[2]) if mc and len(sys.argv) > 2 else 1
    cases = [c for c in V.perpendicular_cases() if c[0].startswith("front_")]
    jobs = [(n, g, p, 100 + s if mc else 0, mc) for n, g, p in cases for s in range(seeds)]
    rows = V.run_all(jobs, 8, "front")
    by = {}
    for r in rows:
        by.setdefault(r["name"], []).append(r)
    ok = total = 0
    for name, rs in by.items():
        n_ok = sum(int(r["success"]) for r in rs)
        ok, total = ok + n_ok, total + len(rs)
        pos = [r["pos_err_cm"] for r in rs if r["pos_err_cm"] is not None]
        head = [r["head_err_deg"] for r in rs if r["head_err_deg"] is not None]
        fails = sorted({str(r.get("failure")) for r in rs if not r["success"]})
        print(f"{name:22s} {n_ok}/{len(rs)}  pos {max(pos) if pos else '-':>5} cm max  head {max(head) if head else '-':>5} deg max  "
              f"{'; '.join(fails)}")
    park = [r for r in rows if r["pos_err_cm"] is not None and not str(r["name"]).endswith(("dist3.0", "occluder_full"))]
    near = [r for r in park if r["pos_err_cm"] <= 6.0 and r["head_err_deg"] <= 10.0]
    print(f"total {ok}/{total} within the strict 5 cm / 5 deg  collisions {sum(int(r.get('collisions') or 0) for r in rows)}")
    if park:
        pos = np.array([r["pos_err_cm"] for r in park])
        head = np.array([r["head_err_deg"] for r in park])
        print(f"parked {len(park)} runs: {len(near)} within 6 cm / 10 deg ({100 * len(near) / len(park):.0f} %); position median {np.median(pos):.1f} cm, "
              f"95th {np.percentile(pos, 95):.1f} cm; heading median {np.median(head):.1f} deg, 95th {np.percentile(head, 95):.1f} deg")


if __name__ == "__main__":
    main()
