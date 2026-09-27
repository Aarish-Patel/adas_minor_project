"""Monte Carlo of the CAR'S OWN decision code (pi/relay_assists.py + pi/path_gate.py + the Hybrid A* evasive steer)
on the digital twin (sim/hw_sim.py physics, simulated LiDAR), in simulated time.

    python -m sim.relay_mc [runs]     -> reports/monte_carlo_relay.png, models/relay_mc.json

Every scenario is driven three times with the same room, the same driver and the same attention lapses:
  off          no ADAS (what the driver alone does)
  adas         path-predicted safety filter + evasive steer, driver's current stick only
  adas+intent  the same, plus the driver's steering trend (where they will steer in 0.4 s) - intent-aware
Metrics: crashes, minimum clearance, interventions, false positives (an intervention while the driver's own
command, held for 2 s, would NOT have hit anything - ground truth), time to reach the goal.
"""
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

DT = 0.05
SCAN_DT = 0.1
T_MAX = 30.0
VARIANTS = ("off", "adas", "adas+intent")


# ------------------------------------------------------------------ scenarios
def scenario(seed):
    from sim.hw_worlds import _finish, room_walls
    from sim.world import Box, Wall, World
    rng = np.random.default_rng(seed)
    L, W = rng.uniform(5.0, 6.5), rng.uniform(3.0, 4.0)
    w = World()
    room_walls(w, -0.8, L, -W / 2, W / 2)
    boxes = []
    if rng.random() < 0.5:                                  # a dividing wall with a doorway
        xd, gw, gy = rng.uniform(2.3, L - 1.6), rng.uniform(0.55, 0.9), rng.uniform(-0.6, 0.6)
        w.add(Wall(xd, -W / 2, xd, gy - gw / 2))
        w.add(Wall(xd, gy + gw / 2, xd, W / 2))
        boxes.append((xd, 0, 0.02, 9))                      # (for placement checks only)
    for _ in range(rng.integers(3, 7)):
        for _try in range(20):
            cx, cy = rng.uniform(1.0, L - 1.2), rng.uniform(-W / 2 + 0.4, W / 2 - 0.4)
            sz = rng.uniform(0.15, 0.40)
            if all(abs(cx - bx) > (sz + bs) / 2 + 0.45 or abs(cy - by) > (sz + bs) / 2 + 0.45 for bx, by, bs, _ in boxes):
                w.add(Box(cx, cy, sz, rng.uniform(0.15, 0.40), rng.uniform(-0.4, 0.4)))
                boxes.append((cx, cy, sz, 0))
                break
    goal = (L - 0.6, rng.uniform(-W / 2 + 0.5, W / 2 - 0.5))
    return _finish(w), goal, rng


class HumanDriver:
    """Heads for the goal, steers round what it sees (ground truth) every 0.3 s, and has attention lapses:
    1-2 s where it keeps doing whatever it was doing (the hazards the ADAS exists for)."""

    def __init__(self, world, goal, rng, params, k_curv, centre):
        self.w, self.goal, self.p = world, goal, params
        self.k, self.c = k_curv, centre
        self.cruise = rng.uniform(120, 200)
        self.lapses = []
        t = rng.exponential(5.0)
        while t < T_MAX:
            d = rng.uniform(1.0, 2.0)
            self.lapses.append((t, t + d))
            t += d + rng.exponential(5.0)
        self.servo, self.pwm, self.next_decide = centre, 0.0, 0.3
        self.stuck_for, self.recover_until, self.recover_servo = 0.0, -1.0, centre

    def lapsed(self, t):
        return any(a <= t < b for a, b in self.lapses)

    def command(self, t, pose, v=0.0):
        # held at an obstacle (by the ADAS or by contact): a person lets go, backs up a little, then re-steers
        if t < self.recover_until:
            return (self.recover_servo, 0.0) if self.recover_until - t > 0.9 else (self.recover_servo, -140.0)
        if self.pwm > 0 and abs(v) < 0.03 and not self.lapsed(t):
            self.stuck_for += DT
        else:
            self.stuck_for = 0.0
        if self.stuck_for > 1.0:
            self.stuck_for = 0.0
            self.recover_until = t + 1.3
            self.recover_servo = self.c                          # back straight away from what the nose is on
            self.next_decide = t + 1.3
            return self.recover_servo, 0.0
        if t < self.next_decide or self.lapsed(t):
            return self.servo, self.pwm
        self.next_decide = t + 0.3
        x, y, th = pose
        gx, gy = self.goal
        best, best_cost = 0.0, 1e9
        for cand in np.radians(np.arange(-60, 61, 10)):
            h = th + cand
            clear = min(self._ray(x, y, h), 1.5)
            want = math.atan2(gy - y, gx - x)
            err = abs((h - want + math.pi) % (2 * math.pi) - math.pi)
            cost = err + 3.0 * max(0.0, 0.9 - clear)
            if cost < best_cost:
                best, best_cost = cand, cost
        kappa_left = best / 0.6                                  # turn toward it over ~0.6 m
        self.servo = float(np.clip(self.c - kappa_left / self.k, 50, 125))
        self.pwm = self.cruise
        return self.servo, self.pwm

    def _ray(self, x, y, h):
        best = 9.0
        dx, dy = math.cos(h), math.sin(h)
        for (x1, y1, x2, y2) in self.w.segments():
            ex, ey = x2 - x1, y2 - y1
            den = dx * ey - dy * ex
            if abs(den) < 1e-9:
                continue
            tt = ((x1 - x) * ey - (y1 - y) * ex) / den
            u = ((x1 - x) * dy - (y1 - y) * dx) / den
            if tt > 0 and 0 <= u <= 1:
                best = min(best, tt)
        return best - 0.1                                        # roughly: from the front of the car


def would_hit(world, params, pose, v, servo, k, c, horizon=2.0, direction=1):
    """Ground truth: does holding this command (in its direction of travel) for `horizon` s hit anything?"""
    x, y, th = pose
    kap = -k * (servo - c)
    v = direction * max(abs(v), 0.15)
    for _ in range(int(horizon / 0.05)):
        th += kap * v * 0.05
        x += v * math.cos(th) * 0.05
        y += v * math.sin(th) * 0.05
        if world.clearance(x, y, th, params) <= 0.0:
            return True
    return False


# ------------------------------------------------------------------ one run
def run(args):
    seed, variant = args
    from adas.aeb import SpeedEstimator
    from adas.config import load_tuning
    from pi.path_gate import PathGate
    from pi.relay_assists import K_CURV_PER_SERVO_DEG as K, RelayAssists
    from sim.hw_sim import SimLidar, VirtualCar

    tun = load_tuning(os.path.join(HERE, "..", "pi", "tuning_real_car.json"))
    world, goal, rng = scenario(seed)
    assist = RelayAssists(tun)
    p = assist.p
    car = VirtualCar(world, p, (0.0, 0.0, 0.0), threaded=False)
    car.last_cmd_t = 0.0
    lidar = SimLidar(car, tun.mount.yaw_offset_deg, n=720, seed=seed)
    driver = HumanDriver(world, goal, np.random.default_rng(seed + 1000), p, K, assist.centre)
    gate = PathGate(p, tun.speed_model)
    vest = SpeedEstimator(tun.speed_model)
    if variant != "off":
        assist.set("evasive", True)
    t, next_scan, seq = 0.0, 0.0, 0
    pts = []
    servo_hist = []
    trace, events = [], []
    interventions = fp = 0
    was_intervening = False
    min_clear, reached = 9.0, None
    while t < T_MAX:
        x, y, th, v, servo_now, pwm_now, crashed = car.pose()
        if crashed:
            break
        if math.hypot(x - goal[0], y - goal[1]) < 0.35:
            reached = t
            break
        if t >= next_scan:
            next_scan += SCAN_DT
            ox, oy = x + p.lidar_x * math.cos(th), y + p.lidar_x * math.sin(th)
            best, dark = lidar._raycast(ox, oy, th)
            r = best + lidar.rng.normal(0, 0.008, len(best))
            ok = np.isfinite(best) & (r >= 0.2) & (r < 12) & (lidar.rng.random(len(best)) > 0.04)
            cw = (-np.degrees(lidar.ccw)) % 360
            cw = np.where(cw > 180, cw - 360, cw)
            pts = [(round(float(a), 1), round(float(d), 3)) for a, d, o in zip(cw, r, ok) if o]
            seq += 1
            gate.on_scan(RelayAssists.points_vehicle_frame(pts, p.lidar_x), seq)
        d_servo, d_pwm = driver.command(t, (x, y, th), v)
        servo_hist = (servo_hist + [d_servo])[-7:]
        lines = [f"A {d_servo:.1f} {d_servo:.1f}", f"M {-int(d_pwm)}"]
        servo_out, phys = d_servo, d_pwm
        intervening = False
        if variant != "off":
            out = assist.process(lines, pts, seq, now=t)
            for ln in out:
                q = ln.split()
                if q[0] == "A":
                    servo_out = (float(q[1]) + float(q[2])) / 2
                elif q[0] == "M":
                    phys = -float(q[1])
            delta = math.atan(-K * (servo_out - assist.centre) * p.wheelbase)
            d_int = None
            if variant == "adas+intent" and len(servo_hist) >= 7:
                rate = (servo_hist[-1] - servo_hist[0]) / (DT * 6)
                d_int = math.atan(-K * (servo_out + rate * 0.4 - assist.centre) * p.wheelbase)
            g_phys, _ = gate.decide(DT, phys, delta, vest.v, 0.0, d_int)
            intervening = assist.assists.evading or abs(g_phys - phys) > 1.0 or abs(servo_out - d_servo) > 1.0
            phys = g_phys
        else:
            gate.memory.advance(DT, vest.v, 0.0)
        vest.update(DT, phys)
        if intervening and not was_intervening:
            interventions += 1
            if not would_hit(world, p, (x, y, th), v, d_servo, K, assist.centre, direction=1 if d_pwm >= 0 else -1):
                fp += 1
                events.append((x, y, "fp"))
            else:
                events.append((x, y, "ok"))
        was_intervening = intervening
        car.command(f"A {servo_out:.1f} {servo_out:.1f}", now=t)
        car.command(f"M {-int(phys)}", now=t)
        t += DT
        car.step_to(t)
        min_clear = min(min_clear, world.clearance(x, y, th, p))
        trace.append((round(x, 3), round(y, 3)))
    x, y, th, v, *_ = car.pose()
    crashed = car.crash_count > 0
    return {"seed": seed, "variant": variant, "crashed": crashed, "reached": reached, "min_clear": float(min_clear),
            "interventions": interventions, "false_positives": fp, "lapses": len(driver.lapses),
            "trace": trace, "events": events, "goal": goal}


def main(runs=24):
    jobs = [(s, v) for s in range(runs) for v in VARIANTS]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run, jobs, chunksize=1))
    summ = {}
    for v in VARIANTS:
        r = [x for x in res if x["variant"] == v]
        n = len(r)
        summ[v] = {"runs": n, "crashes": sum(x["crashed"] for x in r), "reached_goal": sum(x["reached"] is not None for x in r),
                   "interventions": sum(x["interventions"] for x in r), "false_positives": sum(x["false_positives"] for x in r),
                   "median_time_to_goal_s": float(np.median([x["reached"] for x in r if x["reached"]] or [np.nan])),
                   "min_clearance_cm": float(min(x["min_clear"] for x in r) * 100)}
    for v, s in summ.items():
        print(f"{v:12s} crashes {s['crashes']:2d}/{s['runs']}  goal {s['reached_goal']:2d}  interventions {s['interventions']:3d}  "
              f"false positives {s['false_positives']:3d}  median time {s['median_time_to_goal_s']:.1f} s  min clearance {s['min_clearance_cm']:.0f} cm")
    json.dump({"summary": summ, "runs": [{k: v for k, v in x.items() if k != "trace"} for x in res]},
              open(os.path.join(HERE, "..", "models", "relay_mc.json"), "w"), indent=1, default=str)
    figure(res, summ)
    return summ


def figure(res, summ, panels=12):
    from sim.report import style
    plt = style()
    cols = {"off": "#ef4444", "adas": "#60a5fa", "adas+intent": "#2dd4bf"}
    seeds = sorted({x["seed"] for x in res})[:panels]
    fig = plt.figure(figsize=(16, 11))
    for k, s in enumerate(seeds):
        ax = fig.add_subplot(3, 5, k + 1)
        world, goal, _ = scenario(s)
        for x1, y1, x2, y2 in world.segments():
            ax.plot([x1, x2], [y1, y2], color="#94a3b8", lw=1)
        for v in VARIANTS:
            r = next(x for x in res if x["seed"] == s and x["variant"] == v)
            tr = np.array(r["trace"]) if r["trace"] else np.zeros((1, 2))
            ax.plot(tr[:, 0], tr[:, 1], color=cols[v], lw=1.6 if v != "off" else 1.0)
            if r["crashed"]:
                ax.plot(tr[-1, 0], tr[-1, 1], "x", color=cols[v], ms=10, mew=3)
        ax.plot(*goal, "*", color="#facc15", ms=10)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"scenario {s}", fontsize=9)
    ax = fig.add_subplot(3, 5, 14)
    names = list(VARIANTS)
    ax.bar(names, [summ[v]["crashes"] for v in names], color=[cols[v] for v in names]); ax.set_title("crashes")
    ax = fig.add_subplot(3, 5, 15)
    ax.bar(names, [summ[v]["false_positives"] for v in names], color=[cols[v] for v in names]); ax.set_title("needless interventions")
    fig.suptitle("Monte Carlo on the car's own decision code (digital twin): red = no ADAS, blue = ADAS, "
                 "teal = ADAS + driver intent;  x = crash, * = goal", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "..", "reports", "monte_carlo_relay.png"), dpi=120)
    plt.close(fig)
    print("wrote reports/monte_carlo_relay.png")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
