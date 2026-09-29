"""'Return to start' on the relay code in the twin: drive out along a curve, then send HOME and let the car drive back on its own.

    python -m sim.home_eval          # writes models/home.json

With and without loop closure (adas/submaps.py). Reported: where the car ends up relative to where it truly started (position and
heading), from the twin's ground truth.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def drive_out_and_home(world_name="lounge", slam=True, seed=0, out_s=18.0, t_max=90.0):
    from pi.relay_assists import home_goal
    from sim.hw_worlds import WORLDS
    from sim.relay_scenarios import run
    world, start = WORLDS[world_name]()
    st = {"homing": False, "finished": False, "ok": None, "pose_at_home": None}

    def hook(t, assist, pts, seq):
        if seq == 1 and slam:
            assist.speed.enable_slam()
        if t >= out_s and not st["homing"]:
            st["homing"] = True
            time_ = getattr(assist.speed, "slam", None)
            st["pose_at_home"] = assist.speed.pose_corrected
            hx, hy, hh = home_goal(assist.speed.pose_corrected if slam else assist.speed.pose)
            st["ok"] = assist.goto(hx, hy, points=pts, heading_deg=hh)
        elif st["homing"] and not assist.nav.active:
            st["finished"] = True

    def operator(t, x, y, th, v):
        if t < out_s:                                   # out: a wide loop, throttle on
            stick = 0.0 if t < 2.0 else (0.5 if t < 6.0 else (0.0 if t < 8.5 else 0.5))
            return stick, 150.0
        holding = st["homing"] and st["ok"] and not st["finished"]
        return 0.0, (150.0 if holding else 0.0)

    r = run(world, operator, t_max, start=start, seed=seed, hook=hook,
            stop_when=lambda t, x, y, th, v: (st["finished"] and abs(v) < 0.02) or st["ok"] is False)
    sx, sy, sth = start
    err = math.hypot(r["x"] - sx, r["y"] - sy)
    herr = abs(math.degrees((r["th"] - sth + math.pi) % (2 * math.pi) - math.pi))
    tr = np.array([(row[1], row[2]) for row in r["trace"]])
    far = float(np.max(np.hypot(tr[:, 0] - sx, tr[:, 1] - sy))) if len(tr) else 0.0
    loops = r["assist"].speed.slam.loops if getattr(r["assist"].speed, "slam", None) else 0
    return {"slam": slam, "err_cm": 100 * err, "heading_err_deg": herr, "farthest_m": far, "t": round(r["t_end"], 1),
            "crashed": bool(r["collided"]), "why": r["assist"].nav.msg, "loops": loops}


def _job(a):
    return drive_out_and_home(world_name=a[0], slam=a[1], seed=a[2])


def main(seeds=6):
    from concurrent.futures import ProcessPoolExecutor
    res = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for world in ("lounge", "open"):
            for slam in (False, True):
                rows = list(ex.map(_job, [(world, slam, sd) for sd in range(seeds)]))
                key = f"{world} {'with loop closure' if slam else 'without'}"
                res[key] = {"err_cm_mean": float(np.mean([r["err_cm"] for r in rows])), "err_cm_max": float(np.max([r["err_cm"] for r in rows])),
                            "heading_deg_mean": float(np.mean([r["heading_err_deg"] for r in rows])),
                            "heading_deg_max": float(np.max([r["heading_err_deg"] for r in rows])),
                            "crashes": int(sum(r["crashed"] for r in rows)), "loops_mean": float(np.mean([r["loops"] for r in rows])), "runs": len(rows)}
                r_ = res[key]
                print(f"{key:28s} end error mean {r_['err_cm_mean']:5.1f} cm (max {r_['err_cm_max']:5.1f}), heading mean {r_['heading_deg_mean']:4.1f} deg "
                      f"(max {r_['heading_deg_max']:4.1f}), {r_['loops_mean']:.0f} loop closures, crashes {r_['crashes']}/{r_['runs']}", flush=True)
    json.dump(res, open(os.path.join(HERE, "..", "models", "home.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
