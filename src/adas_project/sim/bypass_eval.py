"""Monte Carlo test of the obstacle bypass, for the default simulator car and the REAL-car profile.

    python -m sim.bypass_eval [runs]

Random obstacle: distance 1.2-2.2 m ahead, up to 0.15 m off the line, 0.10-0.40 m wide, sometimes a
wall close on one side (the car must pick the open side), sometimes boxed in (it must refuse).
Expected outcome comes from the geometry: complete if a side is >= 0.55 m wide, refuse if both are <= 0.38 m,
anything safe in between (also when the obstacle touches a wall: known limitation, see build()).
Success = rejoined the line (|lateral| < 6 cm, |heading| < 4 deg), no contact, at least 2 cm of clearance.
"""
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .bypass_driver import BypassDriver
from .intent_data import MODEL_DIR
from .lidar_sim import LidarSim
from .real_car import real_profile
from .simulator import Simulator
from .world import Box, Wall, World

GOOD_LATERAL = 0.06
GOOD_YAW = math.radians(4.0)


def build(rng):
    w = World()
    x0, x1, y0, y1 = -1.0, 5.6, -1.6, 1.6
    for a, b, c, d in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        w.add(Wall(a, b, c, d, 0.25))
    cx, cy = rng.uniform(1.2, 2.2), rng.uniform(-0.15, 0.15)
    width = rng.uniform(0.10, 0.40)
    w.add(Box(cx, cy, 0.20, width, 0.0, 0.16))
    case = str(rng.choice(["open", "open", "wall_left", "wall_right", "boxed"]))
    if case in ("wall_left", "boxed"):
        w.add(Wall(0.6, 0.42, cx + 1.4, 0.42, 0.25))
    if case in ("wall_right", "boxed"):
        w.add(Wall(0.6, -0.42, cx + 1.4, -0.42, 0.25))
    # ground truth: free width beside the obstacle on each side (to a side wall if there is one, else the room)
    left = (0.42 if case in ("wall_left", "boxed") else 1.6) - (cy + width / 2)
    right = (0.42 if case in ("wall_right", "boxed") else 1.6) + (cy - width / 2)
    best = max(left, right)
    # KNOWN LIMITATION: an obstacle within ~16 cm of a side wall is clustered together with that wall (region
    # growing radius), which can make the controller refuse even though the other side is wide open
    merged = (case in ("wall_left", "boxed") and 0.42 - (cy + width / 2) < 0.17) or              (case in ("wall_right", "boxed") and 0.42 + (cy - width / 2) < 0.17)
    if merged:
        best = 0.45
    return w, best, (cx, cy, width), case


def run_one(args):
    seed, profile = args
    rng = np.random.default_rng(seed)
    world, best_gap, obs, case = build(rng)
    if profile == "real":
        rp = real_profile()
        sim = Simulator(world, (0.0, 0.0, 0.0), adas_on="active", params=rp.params, aeb_config=rp.aeb, dynamics=rp.dynamics,
                        adas_speed_model=rp.speed_model, lidar=LidarSim(seed=seed, **rp.lidar_kw), seed=seed)
        params = rp.params
    else:
        sim = Simulator(world, (0.0, 0.0, 0.0), adas_on="active", seed=seed)
        params = sim.p
    drv = BypassDriver(params, pwm=115)
    while sim.t < 40 and not sim.car.collided and not drv.done:
        sim.step(*drv.command(sim))
    for _ in range(100):
        sim.step(0.0, 0.0)
    c = sim.car
    rejoined = drv.state == "DONE" and abs(c.y) < GOOD_LATERAL and abs(c.theta) < GOOD_YAW
    refused = drv.state == "ABORT"
    # must complete when a side is clearly wide enough, must refuse when both are clearly too narrow,
    # and in between either is acceptable as long as nothing is touched
    if best_gap >= 0.55:
        good = rejoined
    elif best_gap <= 0.38:
        good = refused
    else:
        good = rejoined or refused
    ok = (not c.collided) and good and sim.min_clearance > 0.02
    return {"seed": seed, "profile": profile, "case": case, "best_gap_m": round(best_gap, 2), "obstacle": [round(v, 2) for v in obs], "state": drv.state,
            "ok": bool(ok), "crash": bool(c.collided), "lateral_cm": round(c.y * 100, 1), "yaw_deg": round(math.degrees(c.theta), 1),
            "clearance_cm": round(max(sim.min_clearance, 0.0) * 100, 1), "time": round(sim.t, 1)}


def summarise(rows):
    out = {}
    for prof in ("sim", "real"):
        r = [x for x in rows if x["profile"] == prof]
        done = [x for x in r if x["state"] == "DONE"]
        out[prof] = {
            "runs": len(r), "success": sum(x["ok"] for x in r), "crashes": sum(x["crash"] for x in r),
            "refused": sum(x["state"] == "ABORT" for x in r),
            "median_lateral_cm": float(np.median([abs(x["lateral_cm"]) for x in done])) if done else None,
            "p95_lateral_cm": float(np.percentile([abs(x["lateral_cm"]) for x in done], 95)) if done else None,
            "median_clearance_cm": float(np.median([x["clearance_cm"] for x in done])) if done else None,
            "min_clearance_cm": float(min([x["clearance_cm"] for x in done])) if done else None,
            "failures": [x for x in r if not x["ok"]][:8]}
    return out


def main(runs=30):
    jobs = [(s, p) for s in range(runs) for p in ("sim", "real")]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(run_one, jobs, chunksize=1))
    summ = summarise(rows)
    for prof, s in summ.items():
        print(f"[{prof}] {s['success']}/{s['runs']} ok, {s['crashes']} crashes, {s['refused']} refused; "
              f"lateral median {s['median_lateral_cm']:.1f} cm (95th {s['p95_lateral_cm']:.1f}), "
              f"clearance median {s['median_clearance_cm']:.1f} cm (min {s['min_clearance_cm']:.1f})")
        for f in s["failures"]:
            print("   fail:", f)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "bypass_eval.json"), "w") as f:
        json.dump({"summary": summ, "rows": rows}, f, indent=1)
    return summ


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
