"""Evasive-steer evaluation on the relay code in the twin (TODO AR, 14 Nov 2026): the driver holds the throttle and does not steer, a box
stands in the lane. The evasive steer is forward-only (user, 10 Nov): it drives round the box if it can and stops in front of it if not.

For box distances, lateral offsets, throttles and seeds: did the car get past (x beyond the box + 0.8 m) without contact, did it stop in
front of it, did it touch anything, how close did it come, how far did it roll backwards, how long did it take.
Writes models/evasive_eval.json.        python -m sim.evasive_eval [seeds]
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def one(dist, off, pwm, seed, assists=("evasive",)):
    from sim.relay_scenarios import _world, run, steady
    from sim.world import Box
    w = _world()
    w.add(Box(dist, off, 0.26, 0.26))
    r = run(w, steady(pwm), 25, assists, seed=seed, stop_when=lambda t, x, y, th, v: x > dist + 0.8)
    best, back = -1e9, 0.0
    for row in r["trace"]:
        best = max(best, row[1])
        back = max(back, best - row[1])
    passed = r["x"] > dist + 0.8 and not r["collided"]
    return {"dist": dist, "off": off, "pwm": pwm, "seed": seed, "passed": bool(passed), "collided": bool(r["collided"]),
            "min_clear_cm": None if r["min_clear"] is None else round(r["min_clear"] * 100, 1), "back_cm": round(back * 100, 1),
            "x_end": round(r["x"], 2), "t_s": round(float(r["t_end"]), 1), "peak_v": round(max(abs(q[4]) for q in r["trace"]), 2)}


def main(seeds=2):
    rows = [one(d, o, p, s) for d in (1.0, 1.3, 1.7, 2.2, 3.0) for o in (0.0, 0.08, -0.08) for p in (130, 180, 255) for s in range(seeds)]
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "models", "evasive_eval.json"), "w"), indent=1)
    n = len(rows)
    print(f"{n} runs: passed {sum(r['passed'] for r in rows)}, contacts {sum(r['collided'] for r in rows)}, "
          f"rolled back more than 4 cm {sum(r['back_cm'] > 4 for r in rows)}")
    for d in sorted({r["dist"] for r in rows}):
        sel = [r for r in rows if r["dist"] == d]
        cl = [r["min_clear_cm"] for r in sel if r["min_clear_cm"] is not None]
        print(f"  box {d:.1f} m ahead: passed {sum(r['passed'] for r in sel)}/{len(sel)}, contacts {sum(r['collided'] for r in sel)}, "
              f"closest {min(cl) if cl else None} cm, mean time of the passes "
              f"{sum(r['t_s'] for r in sel if r['passed']) / max(1, sum(r['passed'] for r in sel)):.1f} s")
    for p in sorted({r["pwm"] for r in rows}):
        sel = [r for r in rows if r["pwm"] == p]
        print(f"  throttle {p}: passed {sum(r['passed'] for r in sel)}/{len(sel)}, contacts {sum(r['collided'] for r in sel)}")
    return rows


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
