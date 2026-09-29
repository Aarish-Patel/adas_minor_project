"""Monte Carlo test of marker-guided parking.

    python -m sim.parking_eval [runs]

Random starting positions and headings in front of the bay; measures how often the
car parks without touching anything, and how accurately (lateral offset from the
bay's centre line, yaw error).
"""

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .intent_data import MODEL_DIR
from .library import parking
from .simulator import Simulator
from .world import Marker

GOOD_LATERAL = 0.03        # m
GOOD_YAW = math.radians(6)


def run_one(seed):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-0.2, 1.6), rng.uniform(-1.2, 1.2)
    th = math.radians(rng.uniform(-35, 35))
    world, _ = parking()
    sim = Simulator(world, (x, y, th), adas_on="active", seed=seed)
    sim.adas.parking.start()
    marker = next(o for o in world.objects if isinstance(o, Marker))
    while sim.t < 60 and not sim.car.collided:
        sim.step(0.0, 0.0)
        if sim.adas.parking.state in ("done", "failed"):
            break
    for _ in range(80):
        sim.step(0.0, 0.0)
    lat, yaw = sim.adas.parking.error(sim.car.x, sim.car.y, sim.car.theta, marker)
    state = sim.adas.parking.state
    ok = state == "done" and not sim.car.collided and abs(lat) < GOOD_LATERAL and abs(yaw) < GOOD_YAW
    return {"seed": seed, "start": [round(x, 2), round(y, 2), round(math.degrees(th))], "state": state,
            "crash": bool(sim.car.collided), "lateral_cm": round(lat * 100, 2), "yaw_deg": round(math.degrees(yaw), 2),
            "time": round(sim.t, 1), "ok": bool(ok)}


def main(runs=60):
    with ProcessPoolExecutor() as pool:
        res = list(pool.map(run_one, range(runs), chunksize=1))
    ok = sum(r["ok"] for r in res)
    crash = sum(r["crash"] for r in res)
    lat = np.array([abs(r["lateral_cm"]) for r in res if r["state"] == "done"])
    yaw = np.array([abs(r["yaw_deg"]) for r in res if r["state"] == "done"])
    tm = np.array([r["time"] for r in res if r["state"] == "done"])
    summary = {"runs": runs, "success": ok, "crashes": crash, "failed": sum(r["state"] == "failed" for r in res),
               "median_lateral_cm": float(np.median(lat)) if len(lat) else None,
               "p95_lateral_cm": float(np.percentile(lat, 95)) if len(lat) else None,
               "median_yaw_deg": float(np.median(yaw)) if len(yaw) else None,
               "p95_yaw_deg": float(np.percentile(yaw, 95)) if len(yaw) else None,
               "median_time_s": float(np.median(tm)) if len(tm) else None, "runs_detail": res}
    print(f"{runs} random starts: {ok} parked correctly ({100 * ok / runs:.0f}%), {crash} crashes, "
          f"{summary['failed']} gave up")
    if len(lat):
        print(f"lateral error  median {summary['median_lateral_cm']:.1f} cm, 95th {summary['p95_lateral_cm']:.1f} cm")
        print(f"yaw error      median {summary['median_yaw_deg']:.1f} deg, 95th {summary['p95_yaw_deg']:.1f} deg")
        print(f"time           median {summary['median_time_s']:.0f} s")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "parking_eval.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
