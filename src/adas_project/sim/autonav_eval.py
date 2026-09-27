"""Click-to-go autonomy on the car's own relay code (pi/relay_assists.py + pi/path_gate.py) in the digital twin.

The operator holds the throttle (the dead-man switch) and the car drives itself to a goal picked in the vehicle
frame. Run:  python -m sim.autonav_eval   -> prints each case, writes reports/autonav.png
"""
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DT, SCAN_DT = 0.05, 0.1

# (world, goal in the start vehicle frame, what it shows)
CASES = [
    ("doorway", (3.4, 0.0), "round the box, then through the 72 cm doorway"),
    ("doorway", (1.2, 0.9), "a goal off to the left"),
    ("room", (2.0, -0.6), "across the room past the furniture"),
    ("open", (0.3, 1.2), "a sharp turn to a point beside the car"),
    ("gap", (3.2, -0.7), "through the 52 cm gap between two boxes"),
    ("corridor", (-0.6, 0.0), "behind the car in an 80 cm corridor (reverse)"),
]


def drive(world_name, goal, t_max=40.0, seed=0):
    from adas.aeb import SpeedEstimator
    from adas.config import load_tuning
    from pi.path_gate import PathGate
    from pi.relay_assists import K_CURV_PER_SERVO_DEG as K, RelayAssists, apply_car_model
    from sim.hw_sim import SimLidar, VirtualCar
    from sim.hw_worlds import WORLDS

    tun = load_tuning(os.path.join(HERE, "..", "pi", "tuning_real_car.json"))
    apply_car_model(tun)
    world, start = WORLDS[world_name]()
    assist = RelayAssists(tun)
    assist.nav.threaded = False                  # deterministic here; on the car the planner runs in a thread
    p = assist.p
    car = VirtualCar(world, p, start, threaded=False)
    car.last_cmd_t = 0.0
    lidar = SimLidar(car, tun.mount.yaw_offset_deg, n=720, seed=seed)
    gate = PathGate(p, tun.speed_model)
    vest = SpeedEstimator(tun.speed_model)
    sx, sy, sth = start
    goal_w = (sx + goal[0] * math.cos(sth) - goal[1] * math.sin(sth), sy + goal[0] * math.sin(sth) + goal[1] * math.cos(sth))
    t, next_scan, seq, pts = 0.0, 0.0, 0, []
    trace, started, planned = [], False, None
    brakes, min_clear = 0, 9.0
    while t < t_max:
        x, y, th, v, *_rest, crashed = car.pose()
        if crashed:
            break
        if t >= next_scan:
            next_scan += SCAN_DT
            ox, oy = x + p.lidar_x * math.cos(th), y + p.lidar_x * math.sin(th)
            best, _ = lidar._raycast(ox, oy, th)
            r = best + lidar.rng.normal(0, 0.008, len(best))
            ok = np.isfinite(best) & (r >= 0.2) & (r < 12) & (lidar.rng.random(len(best)) > 0.04)
            cw = (-np.degrees(lidar.ccw)) % 360
            cw = np.where(cw > 180, cw - 360, cw)
            pts = [(round(float(a), 1), round(float(d), 3)) for a, d, o in zip(cw, r, ok) if o]
            seq += 1
            gate.on_scan(RelayAssists.points_vehicle_frame(pts, p.lidar_x), seq)
        lines = [f"A {assist.centre:.1f} {assist.centre:.1f}", f"M {-150}"]     # operator: stick centred, throttle held
        if not started and seq >= 2:
            started = True
            import time
            t_plan = time.perf_counter()
            ok_plan = assist.goto(*goal, points=pts)
            plan_ms = (time.perf_counter() - t_plan) * 1000
            if not ok_plan:
                return {"ok": False, "why": assist.nav.msg, "trace": [], "goal": goal_w, "world": world}
            planned = assist.nav.path.copy()
        out = assist.process(lines, pts, seq, now=t) if started else [f"A {assist.centre:.1f} {assist.centre:.1f}", "M 0"]
        servo_out, phys = assist.centre, 0.0
        for ln in out:
            q = ln.split()
            if q[0] == "A":
                servo_out = (float(q[1]) + float(q[2])) / 2
            elif q[0] == "M":
                phys = -float(q[1])
        if started and not assist.nav.active:
            phys = 0.0                             # finished: the operator lets go
        delta = math.atan(-K * (servo_out - assist.centre) * p.wheelbase)
        g_phys, g_brake = gate.decide(DT, phys, delta, vest.v, 0.0, None)
        brakes += int(abs(g_phys - phys) > 1.0)
        vest.update(DT, g_phys)
        car.command(f"A {servo_out:.1f} {servo_out:.1f}", now=t)
        car.command(f"M {-int(g_phys)}", now=t)
        t += DT
        car.step_to(t)
        min_clear = min(min_clear, world.clearance(x, y, th, p))
        trace.append((x, y))
        if started and not assist.nav.active and abs(v) < 0.02:
            break
    x, y, th, *_ = car.pose()
    err = math.hypot(x - goal_w[0], y - goal_w[1])
    # the planned path in the world frame, for the figure
    if planned is not None:
        c, s = math.cos(sth), math.sin(sth)
        planned = np.column_stack([sx + c * planned[:, 0] - s * planned[:, 1], sy + s * planned[:, 0] + c * planned[:, 1]])
    return {"ok": assist.nav.msg == "arrived" and err < 0.2 and car.crash_count == 0, "why": assist.nav.msg,
            "t": round(t, 1), "err": err, "plan_ms": plan_ms, "replans": assist.nav.replans, "crashed": car.crash_count > 0, "min_clear": min_clear,
            "gate_ticks": brakes, "trace": trace, "planned": planned, "goal": goal_w, "world": world}


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
