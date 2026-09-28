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
VARIANTS = ("off", "brake-only", "adas", "adas+intent")
TRUST_THRESHOLD = float(os.environ.get("RC_TRUST", "0.5"))   # hold the swerve when P(driver crashes) is below this
FOS_TRUSTED = os.environ.get("RC_FOS_TRUSTED")                 # sweep the gate's soft-cap FOS for trusted drivers


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


# driver styles: (throttle range, look-ahead m, turn length m, stick rate deg/s, lapses: (mean gap s, duration s),
# slow-down distance m: below this much free way on the chosen heading the driver eases off, as people do)
STYLES = {
    "lapsing": ((120, 200), (1.3, 1.3), 0.6, 90.0, (5.0, (1.0, 2.0)), 0.9),       # careful, but attention lapses
    "late": ((120, 200), (0.7, 0.9), 0.4, 180.0, None, 0.55),                    # reacts close in, steers decisively
    "good": ((120, 200), (1.3, 1.5), 0.6, 90.0, None, 1.0),                      # avoids early and smoothly
    "distracted": ((120, 200), (1.3, 1.3), 0.6, 90.0, (3.0, (1.5, 3.0)), 0.9),   # frequent, long lapses
    "aggressive": ((200, 250), (0.8, 1.0), 0.4, 180.0, None, 0.45),              # fast and late, but competent
}
STOP_GAP = 0.22    # m of free way (bumper to obstacle) at which a driver stops and backs up to re-steer
DEFAULT_STYLES = ("lapsing", "late", "good", "distracted", "aggressive")


class HumanDriver:
    """Heads for the goal, steers round what it sees (ground truth) every 0.2 s, and - depending on the style - has
    attention lapses where it keeps doing whatever it was doing (the hazards the ADAS exists for)."""

    STEER_RATE = 90.0          # servo degrees per second: people move the stick smoothly, not in jumps

    def __init__(self, world, goal, rng, params, k_curv, centre, style="lapsing"):
        """style: see STYLES. 'late' / 'aggressive' never lapse but only react close in and then steer round -
        correct driving that a naive ADAS mistakes for a threat; 'good' is the driver who is already avoiding."""
        self.w, self.goal, self.p = world, goal, params
        self.k, self.c = k_curv, centre
        self.style = style
        thr, look, self.turn_len, self.steer_rate, lapse, self.slow_at = STYLES[style]
        self.cruise = rng.uniform(*thr)
        self.look = rng.uniform(*look) if look[0] != look[1] else look[0]
        self.lapses = []
        if lapse is not None:
            gap, dur = lapse
            t = rng.exponential(gap)
            while t < T_MAX:
                d = rng.uniform(*dur)
                self.lapses.append((t, t + d))
                t += d + rng.exponential(gap)
        self.servo, self.pwm, self.next_decide = centre, 0.0, 0.3
        self.target = centre
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
        if self.lapsed(t):
            return self.servo, self.pwm
        if t < self.next_decide:
            return self._slew(), self.pwm
        self.next_decide = t + 0.2
        x, y, th = pose
        gx, gy = self.goal
        best, best_cost, best_clear = 0.0, 1e9, 9.0
        for cand in np.radians(np.arange(-60, 61, 10)):
            h = th + cand
            clear = min(self._ray(x, y, h), self.look + 0.3)
            want = math.atan2(gy - y, gx - x)
            err = abs((h - want + math.pi) % (2 * math.pi) - math.pi)
            cost = err + 3.0 * max(0.0, self.look - clear)
            if cost < best_cost:
                best, best_cost, best_clear = cand, cost, clear
        kappa_left = best / self.turn_len                        # turn toward it
        self.target = float(np.clip(self.c - kappa_left / self.k, 50, 125))
        # speed like a person: ease off when even the best way is short of room; stop and back up when blocked
        ahead = min(best_clear, self._ray(x, y, th))
        if ahead < STOP_GAP and v > -0.05:
            self.recover_until = t + 1.3
            self.recover_servo = self.c
            self.next_decide = t + 1.3
            self.pwm = 0.0
            return self._slew(), 0.0
        if ahead < self.slow_at:
            frac = (ahead - STOP_GAP) / max(self.slow_at - STOP_GAP, 1e-3)
            self.pwm = max(70.0, self.cruise * (0.35 + 0.65 * frac))
        else:
            self.pwm = self.cruise
        return self._slew(), self.pwm

    def _slew(self):
        step = self.steer_rate * DT
        self.servo += max(-step, min(step, self.target - self.servo))
        return self.servo

    def clone(self):
        import copy
        return copy.copy(self)

    def _ray(self, x, y, h):
        """Free distance for the whole body width: three parallel rays (centre and both sides)."""
        nx, ny = -math.sin(h), math.cos(h)
        return min(self._ray1(x + o * nx, y + o * ny, h) for o in (-0.12, 0.0, 0.12))

    def _ray1(self, x, y, h):
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
        return best - self.p.front_x                             # the ray starts at the rear axle


NEAR_MISS_M = 0.02    # body-to-obstacle clearance counted as a near miss (surrogate safety measure)


def counterfactual(car, driver, t0, horizon=2.0):
    """Ground truth for "was this intervention needed?": fork the simulation now, let the SAME driver carry on
    with no ADAS for `horizon` seconds (their own reactions, their own lapses). Returns "crash", "near-miss" (the
    body came within NEAR_MISS_M - ADAS evaluations count near-crashes as justified interventions too) or None."""
    from sim.hw_sim import VirtualCar
    c2 = VirtualCar(car.world, car.p, (car.x, car.y, car.th), car.reversed, threaded=False)
    c2.v, c2.servo, c2.servo_cmd, c2.pwm = car.v, car.servo, car.servo_cmd, car.pwm
    c2.queue, c2.clock, c2.last_cmd_t = list(car.queue), car.clock, car.last_cmd_t
    d2 = driver.clone()
    t, closest = t0, 9.0
    while t < t0 + horizon:
        x, y, th, v, *_ = c2.pose()
        s, u = d2.command(t, (x, y, th), v)
        c2.command(f"A {s:.1f} {s:.1f}", now=t)
        c2.command(f"M {-int(u)}", now=t)
        t += DT
        c2.step_to(t)
        if c2.crash_count:
            return "crash"
        closest = min(closest, car.world.clearance(c2.x, c2.y, c2.th, car.p))
    return "near-miss" if closest < NEAR_MISS_M else None


def counterfactual_crash(car, driver, t0, horizon=2.0):
    return counterfactual(car, driver, t0, horizon) == "crash"


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
    seed, variant, style = args if len(args) == 3 else (*args, "lapsing")
    from adas.config import load_tuning
    from pi.path_gate import PathGate
    from pi.relay_assists import K_CURV_PER_SERVO_DEG as K, RelayAssists
    from sim.hw_sim import SimLidar, VirtualCar

    tun = load_tuning(os.path.join(HERE, "..", "pi", "tuning_real_car.json"))
    import pi.path_gate
    if FOS_TRUSTED:
        pi.path_gate.FOS_TRUSTED = float(FOS_TRUSTED)
    if os.environ.get("RC_K_WINDOW"):
        pi.path_gate.K_WINDOW_S = float(os.environ["RC_K_WINDOW"])
    from pi.relay_assists import apply_car_model
    apply_car_model(tun)                         # the same fitted model the relay uses
    world, goal, rng = scenario(seed)
    from sim.hw_sim import PI_COMPUTE_FACTOR
    assist = RelayAssists(tun, plan_latency=PI_COMPUTE_FACTOR)   # plans arrive as late as they would on the Pi
    if os.environ.get("RC_EVADE_TTC"):             # sweep the evasive trigger: "<normal>,<attentive>" seconds
        a_n, a_a = (float(s) for s in os.environ["RC_EVADE_TTC"].split(","))
        assist.assists.cfg.evade_ttc, assist.assists.cfg.evade_ttc_attentive = a_n, a_a
    p = assist.p
    car = VirtualCar(world, p, (0.0, 0.0, 0.0), threaded=False)
    car.last_cmd_t = 0.0
    lidar = SimLidar(car, tun.mount.yaw_offset_deg, n=720, seed=seed)
    driver = HumanDriver(world, goal, np.random.default_rng(seed + 1000), p, K, assist.centre, style)
    rint = None
    if variant == "adas+intent":                 # the relay's own intent code (pi/relay_assists.RelayIntent)
        from pi.relay_assists import RelayIntent
        rint = RelayIntent(assist, os.environ.get("RC_INTENT_PATH") or os.path.join(HERE, "..", "models", "intent_net.json"),
                           trust=TRUST_THRESHOLD)
    vpts = np.empty((0, 2))
    gate = PathGate(p, tun.speed_model)
    from pi.relay_assists import ThrottleSmoother
    smoother = ThrottleSmoother()
    from pi.relay_assists import RelaySpeed
    vest = RelaySpeed(tun.speed_model, p.lidar_x)       # the relay's speed: throttle model + LiDAR EKF
    assist.speed = vest
    if variant in ("adas", "adas+intent"):
        assist.set("evasive", True)
        if os.environ.get("RC_NUDGE") == "1":        # opt-in: more (short) corrections, less time overridden
            assist.set("nudge", True)
    t, next_scan, seq = 0.0, 0.0, 0
    pts = []
    trace, events = [], []
    interventions = fp = 0
    kind = None
    p_crash = None
    was_intervening = False
    ep_fp = False
    burden = {f"{a}{b}": 0.0 for a in ("needless", "needed") for b in ("_s", "_steer_s", "_throttle_s")}
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
            vpts = RelayAssists.points_vehicle_frame(pts, p.lidar_x)
            gate.on_scan(vpts, seq)
            vest.on_scan(pts, seq, t)
        d_servo, d_pwm = driver.command(t, (x, y, th), v)
        lines = [f"A {d_servo:.1f} {d_servo:.1f}", f"M {-int(d_pwm)}"]
        servo_out, phys = d_servo, d_pwm
        intervening = False
        if variant != "off":
            p_crash = None
            if rint is not None:
                # learned intent decides whether this driver will handle the threat (personal reaction profile,
                # stick history, LiDAR free distances); the stopping-distance brake below is never suppressed
                rint.update(t, d_servo, d_pwm, vest.v, pts)
                p_crash = rint.p_crash
            lines, _nd = assist.nudge(gate, lines, vest.v_gate(1))  # steering correction first (as in the relay)
            out = assist.process(lines, pts, seq, now=t)
            for ln in out:
                q = ln.split()
                if q[0] == "A":
                    servo_out = (float(q[1]) + float(q[2])) / 2
                elif q[0] == "M":
                    phys = -float(q[1])
            delta = math.atan(-K * (servo_out - assist.centre) * p.wheelbase)
            # intent decides whether to take over the STEERING (evasive hold above); braking stays pure physics
            g_phys, g_brake = gate.decide(DT, phys, delta, vest.v_gate((phys > 0) - (phys < 0)), 0.0,
                                    intent_k_rate=None if rint is None else rint.gate_k_rate,
                                    trusted=rint is not None and rint.gate_trust, leg=assist.planned_leg())
            act = str(gate.info.get("action") or "")
            g_phys = smoother.step(g_phys, DT, emergency=bool(g_brake) or act.startswith(("holding", "stopped")))  # as the relay
            kind = "evasive" if assist.assists.evading else ("gate:" + str(gate.info.get("action")) if abs(g_phys - phys) > 1.0 else
                                                                 ("steer" if abs(servo_out - d_servo) > 1.0 else None))
            intervening = kind is not None
            phys = g_phys
        else:
            gate.memory.advance(DT, vest.v, 0.0)
        vest.command(t, DT, phys, servo_out)
        if intervening and not was_intervening:
            interventions += 1
            # how deep inside its stopping envelope was the car on the DRIVER's own path: free distance / physical
            # stopping distance (1.0 = the last moment braking could still stop it)
            from pi.path_gate import BASE_M, DECEL, REACTION_S
            d_delta = math.atan(-K * (d_servo - assist.centre) * p.wheelbase)
            f_drv, _ = gate.free_distance(d_delta, 1 if d_pwm >= 0 else -1, slop=False)
            stop = BASE_M + abs(v) * REACTION_S + v * v / (2 * DECEL)
            ratio = round(f_drv / stop, 2) if math.isfinite(f_drv) else None
            why = counterfactual(car, driver, t)
            if why is None:
                fp += 1
                events.append((x, y, "fp", kind, None if p_crash is None else round(p_crash, 3), driver.lapsed(t),
                               round(rint.profile.reaction_distance, 2) if rint is not None else None, ratio))
            else:
                events.append((x, y, "ok", kind, ratio, why))
            ep_fp = events[-1][2] == "fp"
        if intervening:
            # how much the driver felt it: seconds overridden, seconds the wheel was taken, share of throttle removed
            steer_taken = abs(servo_out - d_servo) > 1.0
            cut = max(0.0, (d_pwm - phys) / d_pwm) if d_pwm > 0 else (1.0 if phys != d_pwm else 0.0)
            key = "needless" if ep_fp else "needed"
            burden[key + "_s"] += DT
            burden[key + "_steer_s"] += DT * steer_taken
            burden[key + "_throttle_s"] += DT * min(1.0, cut)
        was_intervening = intervening
        car.command(f"A {servo_out:.1f} {servo_out:.1f}", now=t)
        car.command(f"M {-int(phys)}", now=t)
        t += DT
        car.step_to(t)
        min_clear = min(min_clear, world.clearance(x, y, th, p))
        trace.append((round(x, 3), round(y, 3)))
    x, y, th, v, *_ = car.pose()
    crashed = car.crash_count > 0
    return {"seed": seed, "variant": variant, "style": style, "crashed": crashed, "reached": reached, "min_clear": float(min_clear),
            "interventions": interventions, "false_positives": fp, "lapses": len(driver.lapses),
            "trace": trace, "events": events, "goal": goal, "burden": {k: round(v, 2) for k, v in burden.items()}}


def main(runs=24, styles=None):
    styles = styles or tuple(os.environ.get("RC_STYLES", ",".join(DEFAULT_STYLES)).split(","))
    jobs = [(sd, v, st) for sd in range(runs) for v in VARIANTS for st in styles]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run, jobs, chunksize=1))
    summ = {}
    for st in tuple(styles) + ("all",):
        for v in VARIANTS:
            r = [x for x in res if x["variant"] == v and (st == "all" or x["style"] == st)]
            summ[f"{st}/{v}"] = {"runs": len(r), "crashes": sum(x["crashed"] for x in r),
                                 "reached_goal": sum(x["reached"] is not None for x in r),
                                 "interventions": sum(x["interventions"] for x in r),
                                 "needless": sum(x["false_positives"] for x in r),
                                 "needless_takeovers": sum(1 for x in r for e in x["events"] if e[2] == "fp" and e[3] in ("evasive", "steer")),
                                 "needless_brakes": sum(1 for x in r for e in x["events"] if e[2] == "fp" and str(e[3]).startswith("gate:") and "limited" not in str(e[3])),
                                 "needless_limits": sum(1 for x in r for e in x["events"] if e[2] == "fp" and e[3] == "gate:limited"),
                                 "median_time_s": float(np.median([x["reached"] for x in r if x["reached"]] or [np.nan])),
                                 "min_clearance_cm": float(min(x["min_clear"] for x in r) * 100)}
    print(f"{'driver / system':24s} crashes  goal  interv.  needless: takeover brake limit   median time")
    for k, s_ in summ.items():
        print(f"{k:24s} {s_['crashes']:3d}/{s_['runs']:<3d} {s_['reached_goal']:4d}  {s_['interventions']:7d}  "
              f"{s_['needless_takeovers']:17d} {s_['needless_brakes']:5d} {s_['needless_limits']:5d}   {s_['median_time_s']:6.1f} s")
    json.dump({"summary": summ, "runs": [{k: v for k, v in x.items() if k != "trace"} for x in res]},
              open(os.path.join(HERE, "..", "models", "relay_mc.json"), "w"), indent=1, default=str)
    figure([x for x in res if x["style"] == "lapsing"], {v: summ[f"all/{v}"] for v in VARIANTS})
    return summ


LIVE_DIR = os.path.join(HERE, "..", "models", "mc_live")


def summarise(res, variants=VARIANTS):
    out = {}
    for v in variants:
        r = [x for x in res if x["variant"] == v]
        if not r:
            continue
        ev = [e for x in r for e in x["events"]]
        out[v] = {"runs": len(r), "crashes": sum(x["crashed"] for x in r),
                  "reached_goal": sum(x["reached"] is not None for x in r),
                  "interventions": sum(x["interventions"] for x in r),
                  "needless": sum(x["false_positives"] for x in r),
                  "needless_takeovers": sum(1 for e in ev if e[2] == "fp" and e[3] in ("evasive", "steer")),
                  "needless_brakes": sum(1 for e in ev if e[2] == "fp" and str(e[3]).startswith("gate:")
                                         and "limited" not in str(e[3])),
                  "needless_limits": sum(1 for e in ev if e[2] == "fp" and e[3] == "gate:limited"),
                  "overridden_needlessly_s": round(sum(x["burden"]["needless_s"] for x in r), 1),
                  "wheel_taken_needlessly_s": round(sum(x["burden"]["needless_steer_s"] for x in r), 1),
                  "median_time_s": float(np.median([x["reached"] for x in r if x["reached"]] or [np.nan])),
                  "min_clearance_cm": float(min(x["min_clear"] for x in r) * 100)}
    return out


def paired_tests(res, a="adas", b="adas+intent"):
    """One-sided Wilcoxon signed-rank tests on paired runs (same room, driver and lapses): is b better than a?"""
    from scipy.stats import wilcoxon
    ra = {(x["seed"], x["style"]): x for x in res if x["variant"] == a}
    rb = {(x["seed"], x["style"]): x for x in res if x["variant"] == b}
    keys = sorted(set(ra) & set(rb))
    out = {"pairs": len(keys)}
    for name, f in (("needless interventions", lambda x: x["false_positives"]),
                    ("needless takeovers", lambda x: sum(1 for e in x["events"] if e[2] == "fp" and e[3] in ("evasive", "steer"))),
                    ("overridden needlessly (s)", lambda x: x["burden"]["needless_s"]),
                    ("wheel taken needlessly (s)", lambda x: x["burden"]["needless_steer_s"])):
        xa = np.array([f(ra[k]) for k in keys], float)
        xb = np.array([f(rb[k]) for k in keys], float)
        diff = xa - xb
        p = wilcoxon(xa, xb, alternative="greater").pvalue if len(keys) and np.any(np.abs(diff) > 1e-9) else 1.0
        out[name] = {a: float(xa.sum()), b: float(xb.sum()), "better": int((diff > 0).sum()),
                     "worse": int((diff < 0).sum()), "p": float(p)}
    return out


def live(runs, variants, styles, intent_path=None):
    """Monte Carlo for the GUI (gui/lab_tabs.py): every finished run is appended to models/mc_live/runs.jsonl (with
    its trace for the replay), a status file tracks progress, and the summary + paired tests are written at the end."""
    from concurrent.futures import as_completed
    if intent_path:
        os.environ["RC_INTENT_PATH"] = intent_path
    os.makedirs(LIVE_DIR, exist_ok=True)
    runs_path, status_path = os.path.join(LIVE_DIR, "runs.jsonl"), os.path.join(LIVE_DIR, "status.json")
    open(runs_path, "w").close()
    jobs = [(sd, v, st) for sd in range(runs) for v in variants for st in styles]

    def status(**kw):
        json.dump(dict(kw, total=len(jobs), variants=list(variants), styles=list(styles)), open(status_path + ".tmp", "w"))
        os.replace(status_path + ".tmp", status_path)
    status(state="running", done=0)
    res = []
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            res.append(r)
            with open(runs_path, "a") as fh:
                fh.write(json.dumps(r, default=str) + "\n")
            status(state="running", done=len(res), summary=summarise(res, variants))
    tests = paired_tests(res) if "adas" in variants and "adas+intent" in variants else {}
    status(state="done", done=len(res), summary=summarise(res, variants), tests=tests)
    return res


def figure(res, summ, panels=12):
    from sim.report import style
    plt = style()
    cols = {"off": "#ef4444", "brake-only": "#fbbf24", "adas": "#60a5fa", "adas+intent": "#2dd4bf"}
    seeds = sorted({x["seed"] for x in res})[:panels]
    fig = plt.figure(figsize=(16, 11))
    for k, s in enumerate(seeds):
        ax = fig.add_subplot(3, 5, k + 1)
        world, goal, _ = scenario(s)
        for x1, y1, x2, y2 in world.segments():
            ax.plot([x1, x2], [y1, y2], color="#94a3b8", lw=1)
        for v in ("off", "adas", "adas+intent"):
            r = next(x for x in res if x["seed"] == s and x["variant"] == v and x.get("style", "lapsing") == "lapsing")
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
    ax.bar(names, [summ[v]["needless"] for v in names], color=[cols[v] for v in names]); ax.set_title("needless interventions (counterfactual)")
    fig.suptitle("Monte Carlo on the car's own decision code (digital twin): red = no ADAS, blue = ADAS, "
                 "teal = ADAS + driver intent;  x = crash, * = goal", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "..", "reports", "monte_carlo_relay.png"), dpi=120)
    plt.close(fig)
    print("wrote reports/monte_carlo_relay.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        # python -m sim.relay_mc --live <runs> <variants,...> <styles,...> [intent model path]
        live(int(sys.argv[2]), sys.argv[3].split(","), sys.argv[4].split(","), sys.argv[5] if len(sys.argv) > 5 else None)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
