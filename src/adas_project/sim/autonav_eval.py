"""Click-to-go autonomy on the car's own relay code (pi/relay_assists.py + pi/path_gate.py) in the digital twin.

The operator holds the throttle (the dead-man switch) and the car drives itself to a goal picked in the vehicle
frame. Run:  python -m sim.autonav_eval         -> the six classic goals, writes reports/autonav.png
             python -m sim.autonav_eval suite [n] -> the scenario suite (forward preference, dead end, obstacles appearing, a walker,
                                                    goal in an obstacle) over n seeds with behaviour metrics -> models/autonav_suite.json
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


# ------------------------------------------------------------------------------------------------ scenario suite (TODO AB3)
# Click-to-go as the user drives it: the goal is clicked once, the operator holds the throttle, the car must get there - forward
# whenever it can, reversing only when it has to, re-planning round what appears, never oscillating or stuck. Behaviour metrics
# come from the drive itself (the twin's true trace), not from the planner's own report.

def _world_open_dead_end():
    """A 0.7 m wide dead end the car stands in, nose to the closed end; the way out is behind it."""
    from sim.world import Wall, World
    from sim.hw_worlds import _finish
    w = World()
    w.add_room(-3.0, 3.0, -2.0, 2.0)
    w.add(Wall(-0.8, -0.35, 1.2, -0.35))
    w.add(Wall(-0.8, 0.35, 1.2, 0.35))
    w.add(Wall(1.2, -0.35, 1.2, 0.35))
    return _finish(w), (0.0, 0.0, 0.0)


def _world_open():
    from sim.hw_worlds import WORLDS
    return WORLDS["open"]()


def _world_hall():
    """An empty 9 x 4 m hall (x -2 .. 7): room for things to appear in front of the car on a long drive."""
    from sim.world import World
    from sim.hw_worlds import _finish
    w = World()
    w.add_room(-2.0, 7.0, -2.0, 2.0)
    return _finish(w), (0.0, 0.0, 0.0)


def _box_on_path(at_t, x, y, l=0.30, w=0.30):
    """An event: a box appears at (x, y) (world frame) at time at_t - a chair pushed in front of the car."""
    def ev(t, world, state):
        if t >= at_t and not state.get("box_" + str(x)):
            from sim.world import Box
            world.add(Box(x, y, l, w, 0.0, height=0.3, style="box", colour="#b45309"))
            state["box_" + str(x)] = True
    return ev


def _walker(x, y, vx, vy, from_t):
    """An event: a pedestrian starts walking across at from_t."""
    def ev(t, world, state):
        if t >= from_t and not state.get("walker"):
            from sim.world import MovingCircle
            world.add(MovingCircle(x, y, vx, vy, radius=0.08))
            state["walker"] = True
    return ev


# (name, world factory, goal (start vehicle frame), heading deg or None, events, expectation)
SUITE = [
    ("straight_open", _world_open, (3.0, 0.0), None, [], "forward only"),
    ("behind_open", _world_open, (-1.6, 0.0), None, [], "turns round forward (a U-turn / 3-point), not a long reverse"),
    ("side_sharp", _world_open, (0.3, 1.2), None, [], "sharp turn to a point beside the car"),
    ("corridor_ahead", lambda: __import__("sim.hw_worlds", fromlist=["WORLDS"]).WORLDS["corridor"](), (4.5, 0.0), None, [], "forward"),
    ("corridor_behind", lambda: __import__("sim.hw_worlds", fromlist=["WORLDS"]).WORLDS["corridor"](), (-0.6, 0.0), None, [],
     "reverse is required (no room to turn)"),
    ("dead_end_out", _world_open_dead_end, (-2.0, 0.0), None, [], "reverse out of the dead end, then forward"),
    ("doorway", lambda: __import__("sim.hw_worlds", fromlist=["WORLDS"]).WORLDS["doorway"](), (3.4, 0.0), None, [],
     "round the box, then through the 72 cm doorway"),
    ("gap52", lambda: __import__("sim.hw_worlds", fromlist=["WORLDS"]).WORLDS["gap"](), (3.2, -0.7), None, [], "through a 52 cm gap"),
    ("room_across", lambda: __import__("sim.hw_worlds", fromlist=["WORLDS"]).WORLDS["room"](), (2.0, -0.6), None, [],
     "across the room past furniture"),
    ("box_appears", _world_hall, (5.0, 0.0), None, [_box_on_path(3.0, 2.3, 0.0)], "re-plans round a box pushed onto the path"),
    ("boxes_appear", _world_hall, (5.5, 0.0), None, [_box_on_path(2.0, 1.6, 0.0), _box_on_path(5.0, 3.0, 0.25), _box_on_path(8.0, 4.4, -0.25)],
     "several re-plans in a row"),
    ("walker_crosses", _world_hall, (5.0, 0.0), None, [_walker(2.2, -1.2, 0.0, 0.5, 2.5)], "yields to a crossing walker, then goes on"),
    ("goal_in_box", lambda: _world_with_box((2.5, 0.0)), (2.5, 0.0), None, [], "goal inside an obstacle: stops next to it, no dithering"),
]


def _world_with_box(c):
    from sim.world import Box
    from sim.hw_worlds import _finish, WORLDS
    w, st = WORLDS["open"]()
    w.add(Box(c[0], c[1], 0.4, 0.4, 0.0, height=0.3, style="box", colour="#b45309"))
    return _finish(w), st


def _behaviour(trace):
    """From the true trace: metres driven forward / in reverse, gear changes, steering reversals per metre, stop-and-go count."""
    tr = np.array([(r[0], r[1], r[2], r[4], r[8] if r[8] is not None else 0.0) for r in trace], float)
    if len(tr) < 3:
        return {}
    ds = np.hypot(np.diff(tr[:, 1]), np.diff(tr[:, 2]))
    v = tr[1:, 3]
    fwd, rev = float(ds[v > 0.02].sum()), float(ds[v < -0.02].sum())
    sgn = np.sign(np.where(np.abs(v) > 0.03, v, 0.0))
    nz = sgn[sgn != 0]
    gears = int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0
    steer = tr[1:, 4]
    big = steer[np.abs(steer) > 0.10]                                   # stick -1..1: a real steer, not noise
    flips = int((np.diff(np.sign(big)) != 0).sum()) if len(big) > 1 else 0
    moving = np.abs(v) > 0.05
    stops = int(((~moving[1:]) & moving[:-1]).sum())                 # moving -> stopped transitions
    dist = fwd + rev
    return {"fwd_m": round(fwd, 2), "rev_m": round(rev, 2), "gears": gears, "steer_flips_per_m": round(flips / max(dist, 0.1), 2),
            "stops": max(0, stops - gears), "dist_m": round(dist, 2)}


def run_scenario(item, seed=0, t_max=60.0):
    name, make, goal, heading, events, expect = item
    from sim.relay_scenarios import run
    world, start = make()
    sx, sy, sth = start
    goal_w = (sx + goal[0] * math.cos(sth) - goal[1] * math.sin(sth), sy + goal[0] * math.sin(sth) + goal[1] * math.cos(sth))
    st = {"started": False, "finished": False, "ok_plan": None}
    evstate = {}

    def hook(t, assist, pts, seq):
        for ev in events:
            ev(t, world, evstate)
        if not st["started"] and seq >= 2:
            st["started"] = True
            st["ok_plan"] = assist.goto(*goal, points=pts, heading_deg=heading)
            st["t0"] = t
        elif st["started"] and not assist.nav.active:
            st["finished"] = True

    def operator(t, x, y, th, v):
        return 0.0, (150.0 if st["started"] and not st["finished"] else 0.0)

    r = run(world, operator, t_max, start=start, seed=seed, hook=hook,
            stop_when=lambda t, x, y, th, v: st["finished"] and abs(v) < 0.02)
    nav = r["assist"].nav
    err = math.hypot(r["x"] - goal_w[0], r["y"] - goal_w[1])
    out = {"name": name, "seed": seed, "msg": nav.msg, "err_cm": round(err * 100, 1), "collided": bool(r["collided"]),
           "min_clear_cm": round(r["min_clear"] * 100, 1), "t_s": round(r["t_end"] - st.get("t0", 0.0), 1), "replans": nav.replans,
           "finished": st["finished"], "expect": expect}
    out.update(_behaviour(r["trace"]))
    ok = not r["collided"] and st["finished"]
    if name == "goal_in_box":
        ok = ok and err < 0.70 and nav.msg.startswith("arrived")   # as near as it fits: box half 0.2 + nose 0.28 + <= 0.2 m clearance
    else:
        ok = ok and nav.msg.startswith("arrived") and err < 0.12
    if name in ("straight_open", "corridor_ahead", "doorway", "room_across", "gap52"):
        ok = ok and out.get("rev_m", 0) < 0.05              # forward only where forward is possible
    if name == "behind_open":
        ok = ok and out.get("rev_m", 0) < 0.6               # turns round forward; at most a short reverse of a 3-point turn
    ok = ok and out.get("steer_flips_per_m", 0) < 1.0 and out.get("stops", 0) <= (3 if events or name == "goal_in_box" else 1)   # (goal in a box: stop, re-plan, creep nearer)
    out["ok"] = bool(ok)
    return out


def suite(seeds=(0,), workers=8):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(i, s) for i in range(len(SUITE)) for s in seeds]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_suite_job, jobs))
    n_ok = sum(r["ok"] for r in res)
    for r in res:
        print(("  " if r["ok"] else "XX"), {k: r.get(k) for k in ("name", "seed", "msg", "err_cm", "collided", "t_s", "replans", "fwd_m", "rev_m",
                                                                  "gears", "steer_flips_per_m", "stops", "min_clear_cm")})
    print(f"{n_ok}/{len(res)} click-to-go scenarios passed")
    import json
    os.makedirs(os.path.join(HERE, "..", "models"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "..", "models", "autonav_suite.json"), "w"), indent=1)
    return res


def _suite_job(job):
    i, s = job
    return run_scenario(SUITE[i], seed=s)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "suite":         # python -m sim.autonav_eval suite [seeds]
        suite(seeds=tuple(range(int(sys.argv[2]) if len(sys.argv) > 2 else 1)))
    else:
        main()
