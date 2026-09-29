"""Click-to-go autonomy on the car's own relay code (pi/relay_assists.py + pi/path_gate.py) in the digital twin.

The operator holds the throttle (the dead-man switch) and the car drives itself to a goal picked in the vehicle
frame. Run:  python -m sim.autonav_eval   -> prints each case, writes reports/autonav.png
"""
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# (world, goal in the start vehicle frame, what it shows)
CASES = [
    ("doorway", (3.4, 0.0), "round the box, then through the 72 cm doorway"),
    ("doorway", (1.2, 0.9), "a goal off to the left"),
    ("room", (2.0, -0.6), "across the room past the furniture"),
    ("open", (0.3, 1.2), "a sharp turn to a point beside the car"),
    ("gap", (3.2, -0.7), "through the 52 cm gap between two boxes"),
    ("corridor", (-0.6, 0.0), "behind the car in an 80 cm corridor (reverse)"),
]


def drive(world_name, goal, t_max=40.0, seed=0, heading_deg=None, reverse_first=False):
    """One click-to-go on the relay code (sim/relay_scenarios.run): the goal is sent once two scans are in, the
    operator holds the throttle and lets go once the car reports it has finished."""
    from sim.hw_worlds import WORLDS
    from sim.relay_scenarios import run
    world, start = WORLDS[world_name]()
    sx, sy, sth = start
    goal_w = (sx + goal[0] * math.cos(sth) - goal[1] * math.sin(sth), sy + goal[0] * math.sin(sth) + goal[1] * math.cos(sth))
    st = {"started": False, "finished": False, "ok_plan": None, "plan_ms": None, "planned": None}

    def hook(t, assist, pts, seq):
        if not st["started"] and seq >= 2:
            st["started"] = True
            t0 = time.perf_counter()
            st["ok_plan"] = assist.goto(*goal, points=pts, heading_deg=heading_deg, reverse_first=reverse_first)
            st["plan_ms"] = (time.perf_counter() - t0) * 1000
            st["planned"] = None if assist.nav.path is None else assist.nav.path.copy()
        elif st["started"] and not assist.nav.active:
            st["finished"] = True

    def operator(t, x, y, th, v):
        holding = st["started"] and st["ok_plan"] and not st["finished"]
        return 0.0, (150.0 if holding else 0.0)

    r = run(world, operator, t_max, start=start, seed=seed, hook=hook,
            stop_when=lambda t, x, y, th, v: (st["finished"] and abs(v) < 0.02) or st["ok_plan"] is False)
    nav = r["assist"].nav
    if st["ok_plan"] is False:
        return {"ok": False, "why": nav.msg, "trace": [], "goal": goal_w, "world": world}
    err = math.hypot(r["x"] - goal_w[0], r["y"] - goal_w[1])
    planned = st["planned"]
    if planned is not None:                      # the planned path in the world frame, for the figure
        c, s = math.cos(sth), math.sin(sth)
        planned = np.column_stack([sx + c * planned[:, 0] - s * planned[:, 1], sy + s * planned[:, 0] + c * planned[:, 1]])
    head_err = None if heading_deg is None else         abs(math.degrees((r["th"] - sth - math.radians(heading_deg) + math.pi) % (2 * math.pi) - math.pi))
    reversed_m = 0.0 if planned is None else float(sum(
        math.hypot(*(planned[i + 1] - planned[i])) for i in range(len(planned) - 1) if st["planned"][i + 1, 3] < 0))
    return {"ok": nav.msg.startswith("arrived") and err < 0.2 and not r["collided"], "why": nav.msg,
            "heading_err_deg": head_err, "reversed_m": reversed_m,
            "t": round(r["t_end"], 1), "err": err, "plan_ms": st["plan_ms"], "replans": nav.replans,
            "crashed": r["collided"], "min_clear": r["min_clear"],
            "gate_ticks": sum(1 for row in r["trace"] if row[9] is not None),
            "trace": [(row[1], row[2]) for row in r["trace"]], "planned": planned, "goal": goal_w, "world": world}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = []
    for wname, goal, what in CASES:
        r = drive(wname, goal)
        res.append((wname, goal, what, r))
        extra = f"{r['t']} s, ends {r['err'] * 100:.0f} cm from the goal, min clearance {r['min_clear'] * 100:.0f} cm, " \
                f"brake-gate ticks {r['gate_ticks']}, plan {r['plan_ms']:.0f} ms, re-plans {r['replans']}" if "t" in r else ""
        print(f"{wname:8s} goal {goal}: {'OK ' if r['ok'] else 'FAIL'} {r['why']:32s} {extra}  ({what})")
    fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 4.2))
    for ax, (wname, goal, what, r) in zip(axes, res):
        for x1, y1, x2, y2 in r["world"].segments():
            ax.plot([x1, x2], [y1, y2], color="0.25", lw=1.5)
        if r.get("planned") is not None:
            ax.plot(r["planned"][:, 0], r["planned"][:, 1], "--", color="tab:blue", lw=1, label="Hybrid A* plan")
        if r["trace"]:
            tr = np.array(r["trace"])
            ax.plot(tr[:, 0], tr[:, 1], color="tab:green" if r["ok"] else "tab:red", lw=2, label="driven (twin)")
        ax.plot(*r["goal"], marker="*", ms=14, color="tab:orange", label="goal")
        ax.set_title(f"{wname}: {what}\n{'arrived' if r['ok'] else r['why']}", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "..", "reports"), exist_ok=True)
    out = os.path.join(HERE, "..", "reports", "autonav.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
