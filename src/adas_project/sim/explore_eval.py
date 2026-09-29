"""Exploration on the car's relay code in the digital twin: the car maps a room by itself.

    python -m sim.explore_eval [world ...]       # writes reports/explore.png and models/explore.json

The operator only holds the throttle (the dead-man switch); the Explorer picks frontiers and sends click-to-go goals. Reported:
explored share of the true free floor (cells the 20 cm car body could stand on), time, distance driven, crashes.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def true_free_cells(world, grid, body=0.15):
    """Boolean grid of cells the car could stand on (clearance of a point >= body from every wall / obstacle), inside the room."""
    from sim.hw_worlds import WORLDS
    segs = np.asarray(world.segments(), float).reshape(-1, 4)
    xs, ys = segs[:, [0, 2]], segs[:, [1, 3]]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    free = np.zeros((grid.n, grid.n), bool)
    step = 3
    for i in range(0, grid.n, step):
        for j in range(0, grid.n, step):
            x, y = grid.world(i, j)
            if x0 + body < x < x1 - body and y0 + body < y < y1 - body and world.clearance(x, y, 0.0, _P) >= body - 0.10:
                free[i:i + step, j:j + step] = True
    return free


_P = None
_EX = None


def explore(world_name, t_max=150.0, seed=0):
    global _P
    from adas.explore import Explorer
    from sim.hw_worlds import WORLDS
    from sim.relay_scenarios import run
    world, start = WORLDS[world_name]()
    ex = None
    st = {"finished": False, "started": False}

    def hook(t, assist, pts, seq):
        global _P
        _P = assist.p
        global _EX
        if seq >= 2 and not st["started"]:
            st["started"] = True
            assist.explore(True)                     # the relay's own exploration (RelayAssists.explore / _explore_tick)
            _EX = assist.explorer
        if st["started"] and assist.explorer is None:
            st["finished"] = True

    def operator(t, x, y, th, v):
        return 0.0, (0.0 if st["finished"] else 150.0)

    r = run(world, operator, t_max, start=start, seed=seed, hook=hook,
            stop_when=lambda t, x, y, th, v: st["finished"] and abs(v) < 0.02)
    ex = _EX
    free = true_free_cells(world, ex.grid)
    known = ex.grid.state() == 1
    cov = float(np.count_nonzero(known & free)) / max(1, np.count_nonzero(free))
    tr = np.array([(row[1], row[2]) for row in r["trace"]])
    dist = float(np.sum(np.hypot(*np.diff(tr, axis=0).T))) if len(tr) > 1 else 0.0
    return {"world": world_name, "coverage": cov, "t": round(r["t_end"], 1), "distance_m": round(dist, 1),
            "crashed": bool(r["collided"]), "min_clear_cm": round(r["min_clear"] * 100, 1), "goals": ex.goals_sent,
            "finished": ex.done, "msg": ex.msg, "grid": ex.grid, "trace": tr, "true_free": free, "world_obj": world}


def main(names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = []
    for n in names:
        r = explore(n)
        res.append(r)
        print(f"{n:9s} explored {100 * r['coverage']:5.1f} % of the free floor in {r['t']} s, {r['distance_m']} m driven, "
              f"{r['goals']} frontier goals, crashed={r['crashed']}, min clearance {r['min_clear_cm']} cm  ({r['msg']})", flush=True)
    fig, axes = plt.subplots(1, len(res), figsize=(5 * len(res), 4.6))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, res):
        g = r["grid"]
        st = g.state().T
        img = np.zeros(st.shape + (3,))
        img[st == 0] = (0.55, 0.58, 0.64)
        img[st == 1] = (0.93, 0.95, 0.98)
        img[st == 2] = (0.13, 0.16, 0.22)
        ax.imshow(img, origin="lower", extent=(-g.half, g.half, -g.half, g.half))
        tr = r["trace"]
        ax.plot(tr[:, 0], tr[:, 1], color="#1E6BFF", lw=1.5)
        segs = np.asarray(r["world_obj"].segments(), float).reshape(-1, 4)
        lo, hi = segs[:, [0, 2]].min() - 0.8, segs[:, [0, 2]].max() + 0.8
        ylo, yhi = segs[:, [1, 3]].min() - 0.8, segs[:, [1, 3]].max() + 0.8
        ax.set_xlim(lo, hi)
        ax.set_ylim(ylo, yhi)
        ax.set_title(f"{r['world']}: {100 * r['coverage']:.0f} % explored, {r['t']} s", fontsize=9)
        ax.set_aspect("equal")
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "..", "reports"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "..", "reports", "explore.png"), dpi=130)
    json.dump([{k: v for k, v in r.items() if k in ("world", "coverage", "t", "distance_m", "crashed", "min_clear_cm", "goals", "finished", "msg")}
               for r in res], open(os.path.join(HERE, "..", "models", "explore.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["open", "room"])
