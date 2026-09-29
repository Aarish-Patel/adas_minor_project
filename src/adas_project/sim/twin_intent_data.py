"""Training / test data for the driver-intent (crash-risk) model from the digital twin, with domain randomisation.

    python -m sim.twin_intent_data [train_runs] [test_runs]  -> data/intent/{train,test}.npz

Why this exists (TODO M1): the v2 model was trained only on moments within 1.2 m of an obstacle, on simulated drivers
who slow down like careful people, with one fixed car and a clean sensor. On the car it is asked about every tick -
far away, full throttle, reversing, a vibrating LiDAR - so it was mostly answering questions it had never seen.

Digital-twin training (RESEARCH.md section 7):
  - The twin's physics is the car model fitted from real drive logs (sim/hw_sim.load_car_model: system
    identification); every run draws the car and the sensor from a range AROUND that fit (domain randomisation,
    Tobin et al. 2017; dynamics randomisation, Peng et al. 2018): top speed, dead-band, motor lag, coast and brake
    deceleration, command delay, steering gain, the servo's true centre (the relay assumes the nominal one), LiDAR
    range noise, dropout, yaw-offset error, vibration jitter, and the speed estimate's lag, noise and scale error.
  - Many situation families, not one driver type: random rooms with seven driver styles, and scripted approaches
    (straight at walls and boxes, alongside a wall, on a curve, reversing, through gaps) whose reaction - none,
    brake, coast, steer away, stop - happens at a random distance, from far too late to comfortably early.
  - Label for every tick (not only threat moments): did this driver, with no ADAS, touch something or pass within
    2 cm while moving, within the next 2 s? The run itself is the counterfactual (no ADAS acts on it).
Each tick stores the v3 tick vector the relay computes (adas/intent_net.tick_vector) from the noisy sensing, so the
models learn from exactly what the car sees; test runs also store the v2 features so the old model can be audited on
the same moments.
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
OUT_DIR = os.path.join(ROOT, "data", "intent")

DT = 0.05
SCAN_DT = 0.1
HORIZON = 2.0
NEAR_MISS_M = 0.02
FAMILIES = ("rooms", "straight", "parallel", "curve", "reverse", "gap")
FAMILY_WEIGHTS = (0.30, 0.25, 0.12, 0.13, 0.10, 0.10)
ROOM_STYLES = ("lapsing", "late", "good", "distracted", "aggressive", "reckless", "keyboard")
EXTRA_STYLES = {       # added to sim.relay_mc.STYLES for data only (the Monte Carlo keeps its five)
    "reckless": ((200, 255), (0.5, 0.7), 0.35, 240.0, (6.0, (0.5, 1.5)), 0.30),     # full throttle, reacts late
    "keyboard": ((150, 255), (0.8, 1.0), 0.40, 1000.0, (8.0, (0.5, 1.5)), 0.60),    # bang-bang stick and throttle
}


# ---------------------------------------------------------------- domain randomisation
def randomise(rng, level=1.0):
    """One draw of the car + sensor around the fitted twin. level 0 = the nominal twin, 1 = the full ranges."""
    def u(a, b):
        mid = (a + b) / 2
        return mid + (rng.uniform(a, b) - mid) * level
    dr = {
        "v_max_x": u(0.85, 1.15), "deadband": u(6.0, 18.0), "tau_motor": u(0.02, 0.08),
        "coast_decel": u(2.5, 5.0), "brake_decel": u(5.0, 10.0), "delay_s": u(0.06, 0.20),
        "k_curv_x": u(0.85, 1.15), "servo_offset": u(-3.0, 3.0),
        "noise": u(0.004, 0.02), "dropout": u(0.0, 0.12), "yaw_err_deg": u(-2.0, 2.0),
        "jitter_deg": u(0.0, 1.2), "v_lag_s": u(0.0, 0.10), "v_noise": u(0.01, 0.05), "v_scale": u(0.9, 1.1),
    }
    for k in ("tau_motor", "delay_s", "noise", "dropout", "jitter_deg", "v_lag_s", "v_noise"):
        dr[k] = max(0.0, dr[k])            # level > 1 widens the ranges; physical quantities stay >= 0
    for k in ("coast_decel", "brake_decel"):
        dr[k] = max(1.0, dr[k])
    return dr


def apply_car(car, dr):
    m = car.m
    m["v_max"] *= dr["v_max_x"]
    m["k_curv_per_deg"] *= dr["k_curv_x"]
    m["servo_centre"] += dr["servo_offset"]
    for k in ("deadband", "tau_motor", "coast_decel", "brake_decel", "delay_s"):
        m[k] = dr[k]
    car.servo = car.servo_cmd = m["servo_centre"]


class Sensor:
    """The LiDAR as the relay sees it (vehicle-frame points), with this run's noise, dropout, yaw error, jitter."""

    def __init__(self, car, dr, rng, p, n=720):
        from sim.hw_sim import SimLidar
        self.lidar = SimLidar(car, 0.0, n=n, seed=int(rng.integers(1 << 30)))
        self.car, self.dr, self.rng, self.p = car, dr, rng, p

    def points(self):
        from pi.relay_assists import RelayAssists
        x, y, th, *_ = self.car.pose()
        p, dr, rng = self.p, self.dr, self.rng
        ox, oy = x + p.lidar_x * math.cos(th), y + p.lidar_x * math.sin(th)
        best, dark = self.lidar._raycast(ox, oy, th)
        fin = np.where(np.isfinite(best), best, 0.0)
        r = best + rng.normal(0, dr["noise"] + 0.004 * fin, len(best))
        ok = np.isfinite(best) & (r >= 0.2) & (r < 12) & ~dark & (rng.random(len(best)) > dr["dropout"])
        cw = (-np.degrees(self.lidar.ccw)) % 360
        cw = np.where(cw > 180, cw - 360, cw)
        cw = cw + dr["yaw_err_deg"] + rng.normal(0, dr["jitter_deg"]) + rng.normal(0, 0.3 * dr["jitter_deg"], len(cw))
        return RelayAssists.points_vehicle_frame([(a, d) for a, d, o in zip(cw, r, ok) if o], p.lidar_x)


class SpeedEstimate:
    """The relay's speed estimate: lagged, noisy, scaled."""

    def __init__(self, dr, rng):
        self.dr, self.rng, self.buf = dr, rng, []

    def __call__(self, t, v):
        self.buf.append((t, v))
        while len(self.buf) > 2 and self.buf[1][0] <= t - self.dr["v_lag_s"]:
            self.buf.pop(0)
        return self.buf[0][1] * self.dr["v_scale"] + self.rng.normal(0, self.dr["v_noise"])


# ---------------------------------------------------------------- scripted drivers
def ray(world, x, y, h, front_x, half=0.12):
    """Ground-truth free distance ahead of the bumper for the body width (three parallel rays)."""
    best = 9.0
    dx, dy = math.cos(h), math.sin(h)
    seg = world.segments()
    for o in (-half, 0.0, half):
        px, py = x - o * math.sin(h), y + o * math.cos(h)
        ex, ey = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
        den = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = ((seg[:, 0] - px) * ey - (seg[:, 1] - py) * ex) / den
            uu = ((seg[:, 0] - px) * dy - (seg[:, 1] - py) * dx) / den
        hit = (np.abs(den) > 1e-9) & (tt > 0) & (uu >= 0) & (uu <= 1)
        if hit.any():
            best = min(best, float(tt[hit].min()))
    return best - front_x


class ScriptedDriver:
    """Drives a plan: throttle `pwm` (negative = reverse) on steering `servo`, optionally with stick noise, then -
    when the ground-truth free way in the direction of travel falls below `d_react` (or the side gap below it, for
    'parallel') - reacts: 'none' (keeps going: an attention lapse), 'brake' (reverse pulse, then stop), 'coast'
    (throttle off), 'steer' (turns `steer_deg` at `rate` deg/s, throttle kept or eased), 'stop' (stops, waits, backs
    up), 'aim' (steers toward a target point, e.g. a gap's centre)."""

    def __init__(self, world, rng, p, centre, plan):
        self.w, self.rng, self.p, self.c = world, rng, p, centre
        self.plan = plan
        self.servo, self.pwm = centre + plan.get("servo", 0.0), 0.0
        self.reacted_at = None
        self.noise = 0.0

    def lapsed(self, t):
        return self.plan["behaviour"] == "none"

    def _free(self, pose, direction):
        x, y, th = pose
        if direction > 0:
            return ray(self.w, x, y, th, self.p.front_x)
        return ray(self.w, x, y, th + math.pi, -self.p.rear_x)

    def command(self, t, pose, v=0.0):
        pl = self.plan
        target_pwm = pl["pwm"] * min(1.0, t / max(pl.get("ramp", 0.3), 1e-3))
        if pl.get("stick_noise", 0.0) > 0:
            self.noise += (-self.noise * 2.0 + self.rng.normal(0, pl["stick_noise"]) * 3.0) * DT
        direction = 1 if pl["pwm"] >= 0 else -1
        if self.reacted_at is None:
            if pl["family"] == "parallel":
                x, y, th = pose
                side = min(ray(self.w, x, y, th + math.pi / 2, self.p.width / 2),
                           ray(self.w, x, y, th - math.pi / 2, self.p.width / 2))
                trig = side < pl["d_react"] or self._free(pose, 1) < pl["d_react"]
            else:
                trig = self._free(pose, direction) < pl["d_react"]
            if trig and pl["behaviour"] != "none":
                self.reacted_at = t
            base = self.c + pl.get("servo", 0.0) + self.noise
            self.servo = base
            self.pwm = target_pwm
            return self.servo, self.pwm
        dt_r = t - self.reacted_at
        b = pl["behaviour"]
        if b == "brake":
            self.pwm = -direction * pl.get("brake_pwm", 180.0) if dt_r < pl.get("brake_s", 0.35) else 0.0
        elif b == "coast":
            self.pwm = 0.0
        elif b == "stop":
            self.pwm = 0.0 if dt_r < 1.0 else (-direction * 140.0 if dt_r < 1.8 else 0.0)
        elif b in ("steer", "aim"):
            if b == "aim":
                x, y, th = pose
                gx, gy = pl["aim"]
                err = (math.atan2(gy - y, gx - x) - th + math.pi) % (2 * math.pi) - math.pi
                goal = self.c - math.degrees(err) * pl.get("aim_gain", 1.5)
            else:
                goal = self.c + pl.get("servo", 0.0) + pl["steer_deg"]
            goal = max(self.c - 55.0, min(self.c + 55.0, goal))
            step = pl["rate"] * DT
            self.servo += max(-step, min(step, goal - self.servo))
            self.pwm = target_pwm * pl.get("ease", 1.0)
        return self.servo, self.pwm


# ---------------------------------------------------------------- scenes
def _room(rng, x0=-1.5, x1=None, half=None):
    from sim.hw_worlds import room_walls
    from sim.world import World
    w = World()
    x1 = x1 if x1 is not None else rng.uniform(4.0, 6.0)
    half = half if half is not None else rng.uniform(1.5, 2.5)
    room_walls(w, x0, x1, -half, half)
    return w


def scene(family, rng, centre):
    """-> (world, start pose, plan) for a scripted family."""
    from sim.hw_worlds import _finish
    from sim.world import Box, Wall
    behaviours = {"straight": (("none", .3), ("brake", .2), ("coast", .15), ("steer", .25), ("stop", .1)),
                  "parallel": (("none", .45), ("steer", .45), ("brake", .1)),
                  "curve": (("none", .35), ("steer", .35), ("brake", .15), ("coast", .15)),
                  "reverse": (("none", .35), ("brake", .3), ("coast", .25), ("steer", .1)),
                  "gap": (("none", .3), ("aim", .45), ("brake", .15), ("coast", .1))}[family]
    names, probs = zip(*behaviours)
    b = str(rng.choice(names, p=np.array(probs) / sum(probs)))
    plan = {"family": family, "behaviour": b, "pwm": rng.uniform(40, 255), "ramp": rng.uniform(0.1, 1.0),
            "d_react": rng.uniform(0.05, 2.0), "steer_deg": rng.choice([-1, 1]) * rng.uniform(15, 55),
            "rate": rng.uniform(90, 500), "ease": rng.choice([1.0, rng.uniform(0.4, 1.0)]),
            "brake_pwm": rng.uniform(80, 255), "brake_s": rng.uniform(0.15, 0.6),
            "stick_noise": rng.choice([0.0, rng.uniform(0.5, 4.0)])}
    th0 = math.radians(rng.uniform(-10, 10))
    if family == "straight":
        d0 = rng.uniform(0.6, 4.0)
        w = _room(rng, x1=d0 + rng.uniform(1.5, 3.0))
        if rng.random() < 0.5:
            w.add(Wall(d0 + 0.25, -3, d0 + 0.25, 3))
        else:
            w.add(Box(d0 + 0.25, rng.uniform(-0.25, 0.25), rng.uniform(0.15, 0.5), rng.uniform(0.15, 0.5),
                      rng.uniform(-0.5, 0.5)))
        start = (0.0, 0.0, th0)
    elif family == "parallel":
        off = rng.uniform(0.10, 0.6) + 0.1
        side = rng.choice([-1, 1])
        w = _room(rng, x1=rng.uniform(5.0, 7.0), half=2.5)
        w.add(Wall(-1.0, side * off, 8.0, side * off))
        plan["d_react"] = rng.uniform(0.01, 0.4)
        plan["steer_deg"] = side * rng.uniform(8, 30)                   # away from the wall (+ servo = right)
        start = (0.0, 0.0, math.radians(side * rng.uniform(0, 12)))
    elif family == "curve":
        w = _room(rng)
        for _ in range(rng.integers(1, 5)):
            w.add(Box(rng.uniform(0.6, 3.5), rng.uniform(-1.3, 1.3), rng.uniform(0.15, 0.45),
                      rng.uniform(0.15, 0.45), rng.uniform(-0.5, 0.5)))
        plan["servo"] = rng.choice([-1, 1]) * rng.uniform(8, 50)
        plan["steer_deg"] = -plan["servo"] * rng.uniform(0.5, 1.0)       # straighten out
        start = (0.0, 0.0, th0)
    elif family == "reverse":
        d0 = rng.uniform(0.3, 2.5)
        w = _room(rng, x0=-(d0 + 0.2 + rng.uniform(0.0, 1.0)))
        if rng.random() < 0.5:
            w.add(Wall(-d0 - 0.2, -3, -d0 - 0.2, 3))
        else:
            w.add(Box(-d0 - 0.35, rng.uniform(-0.25, 0.25), rng.uniform(0.15, 0.5), rng.uniform(0.15, 0.5)))
        plan["pwm"] = -rng.uniform(60, 255)
        plan["servo"] = rng.choice([0.0, rng.uniform(-30, 30)])
        start = (0.0, 0.0, th0)
    else:                                                                 # gap: a doorway in a dividing wall
        xd, gw, gy = rng.uniform(1.2, 3.5), rng.uniform(0.35, 0.9), rng.uniform(-0.3, 0.3)
        w = _room(rng, x1=xd + rng.uniform(1.5, 3.0))
        w.add(Wall(xd, -3, xd, gy - gw / 2))
        w.add(Wall(xd, gy + gw / 2, xd, 3))
        plan["aim"] = (xd + 0.5, gy)
        plan["aim_gain"] = rng.uniform(0.8, 2.5)
        plan["d_react"] = rng.uniform(0.3, 2.5)
        start = (0.0, rng.uniform(-0.35, 0.35), th0)
    return _finish(w), start, plan


# ---------------------------------------------------------------- one run
def run(args):
    seed, family, level, with_v2 = args
    from adas.config import load_tuning
    from adas.intent_net import DriverProfile, features, free_now, tick_vector, Z_FREE_NOW, HORIZON3
    from pi.relay_assists import K_CURV_PER_SERVO_DEG as K, apply_car_model, car_params
    from sim.hw_sim import VirtualCar
    import sim.relay_mc as mc

    for k, v in EXTRA_STYLES.items():
        mc.STYLES.setdefault(k, v)
    rng = np.random.default_rng(seed)
    tun = load_tuning(os.path.join(ROOT, "pi", "tuning_real_car.json"))
    apply_car_model(tun)
    p = car_params(tun.mount)
    centre = float(tun.servo.left_center)
    dr = randomise(rng, level)
    style = None
    if family == "rooms":
        world, goal, _ = mc.scenario(int(rng.integers(1 << 30)))
        start = (0.0, 0.0, 0.0)
        style = str(rng.choice(ROOM_STYLES))
        driver = mc.HumanDriver(world, goal, np.random.default_rng(int(rng.integers(1 << 30))), p, K, centre, style)
        t_max = 25.0
    else:
        world, start, plan = scene(family, rng, centre)
        goal = None
        driver = ScriptedDriver(world, rng, p, centre, plan)
        style = plan["behaviour"]
        t_max = 9.0
    car = VirtualCar(world, p, start, threaded=False)
    car.last_cmd_t = 0.0
    apply_car(car, dr)
    sensor, vest = Sensor(car, dr, rng, p), SpeedEstimate(dr, rng)
    prof, prof2 = DriverProfile(), DriverProfile()
    t, next_scan, pts = 0.0, 0.0, np.empty((0, 2))
    hist, pwm_hist = [], []
    Z, V2, LAP, VT, CLR, MOV, POSE = [], [], [], [], [], [], []
    still_for = 0.0
    while t < t_max:
        x, y, th, v, _, _, crashed = car.pose()
        if crashed:
            break
        if goal is not None and math.hypot(x - goal[0], y - goal[1]) < 0.35:
            break
        if t >= next_scan:
            next_scan += SCAN_DT
            pts = sensor.points()
        s, u = driver.command(t, (x, y, th), v)
        ve = vest(t, v)
        hist = (hist + [s])[-80:]
        pwm_hist = (pwm_hist + [u])[-40:]
        z = tick_vector(s, u, ve, pts, p, centre, K, react=prof.reaction_distance)
        prof.update(hist, z[Z_FREE_NOW] * HORIZON3)
        Z.append(z)
        if with_v2:                                    # the v2 model as the relay ran it, with its own profile
            f = features(hist, u, ve, pts, p, centre, K, profile=prof2, pwm_hist=pwm_hist)
            if f is not None:
                prof2.update(hist, free_now(f))
            V2.append(f if f is not None else np.full(23, np.nan))
        LAP.append(driver.lapsed(t))
        VT.append(v)
        POSE.append((x, y, th))
        CLR.append(world.clearance(x, y, th, p))
        car.command(f"A {s:.1f} {s:.1f}", now=t)
        car.command(f"M {-int(u)}", now=t)
        t += DT
        car.step_to(t)
        # scripted runs end once the car has been standing for a second after its reaction
        still_for = still_for + DT if abs(v) < 0.02 and t > 1.0 else 0.0
        if family != "rooms" and still_for > 1.5:
            break
    n = len(Z)
    crash_tick = n if car.crash_count else None
    VT, CLR = np.array(VT), np.array(CLR)
    event = (CLR < NEAR_MISS_M) & (np.abs(VT) > 0.05)
    ev_idx = list(np.flatnonzero(event)) + ([crash_tick] if crash_tick is not None else [])
    tte = np.full(n, np.inf)
    nxt = np.inf
    ev_set = set(ev_idx)
    for i in range(n, -1, -1):
        if i in ev_set:
            nxt = i
        if i < n:
            tte[i] = (nxt - i) * DT if np.isfinite(nxt) else np.inf
    y = (tte <= HORIZON).astype(np.int8)
    out = {"Z": np.array(Z, np.float32).reshape(n, -1), "y": y, "tte": tte.astype(np.float32),
           "lapsed": np.array(LAP, bool), "v_true": VT.astype(np.float32), "clear": CLR.astype(np.float32),
           "family": family, "style": style, "crashed": bool(car.crash_count), "seed": seed,
           "dr": np.array([dr[k] for k in sorted(dr)], np.float32), "dr_dict": dr,
           "pose": np.array(POSE, np.float32).reshape(n, 3), "segments": world.segments().tolist()}
    if with_v2:
        out["V2"] = np.array(V2, np.float32).reshape(n, -1)
    return out


def jobs(n_runs, seed0, level, with_v2, rng_seed):
    rng = np.random.default_rng(rng_seed)
    fam = rng.choice(FAMILIES, size=n_runs, p=np.array(FAMILY_WEIGHTS) / sum(FAMILY_WEIGHTS))
    return [(seed0 + i, str(f), level, with_v2) for i, f in enumerate(fam)]


def build(n_runs, seed0, level, with_v2, path, rng_seed, workers=None):
    """Runs the jobs in parallel and saves one npz: the tick arrays concatenated, with run ids and run metadata."""
    import json
    import time
    from concurrent.futures import as_completed
    js = jobs(n_runs, seed0, level, with_v2, rng_seed)
    status_path = os.path.join(ROOT, "models", "intent_training", "data_status.json")
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    name = os.path.basename(path).replace(".npz", "")
    runs = [None] * len(js)
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(run, j): i for i, j in enumerate(js)}
        for k, f in enumerate(as_completed(futs)):
            runs[futs[f]] = f.result()
            if k % 20 == 0 or k == len(js) - 1:
                json.dump({"set": name, "done": k + 1, "total": len(js), "level": level, "t": time.time()},
                          open(status_path + ".tmp", "w"))
                os.replace(status_path + ".tmp", status_path)
    runs = [r for r in runs if len(r["y"])]
    keys = ["Z", "y", "tte", "lapsed", "v_true", "clear"] + (["V2"] if with_v2 else [])
    cat = {k: np.concatenate([r[k] for r in runs]) for k in keys}
    cat["run"] = np.concatenate([np.full(len(r["y"]), i, np.int32) for i, r in enumerate(runs)])
    cat["tick"] = np.concatenate([np.arange(len(r["y"]), dtype=np.int32) for r in runs])
    cat["run_family"] = np.array([r["family"] for r in runs])
    cat["run_style"] = np.array([r["style"] for r in runs])
    cat["run_crashed"] = np.array([r["crashed"] for r in runs])
    cat["run_seed"] = np.array([r["seed"] for r in runs])
    cat["run_dr"] = np.stack([r["dr"] for r in runs])
    cat["dr_keys"] = np.array(sorted(randomise(np.random.default_rng(0))))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **cat)
    return cat


def main(train_runs=3000, test_runs=600, level=1.0):
    import time
    t0 = time.time()
    tr = build(train_runs, 10_000, level, False, os.path.join(OUT_DIR, "train.npz"), 1)
    print(f"train: {len(tr['y'])} ticks from {len(tr['run_family'])} runs, positives {tr['y'].mean():.3f}, "
          f"{time.time() - t0:.0f} s", flush=True)
    te = build(test_runs, 900_000, 1.0, True, os.path.join(OUT_DIR, "test.npz"), 2)
    print(f"test:  {len(te['y'])} ticks from {len(te['run_family'])} runs, positives {te['y'].mean():.3f}, "
          f"{time.time() - t0:.0f} s", flush=True)


def showcase(n=12, path=None, level=1.0):
    """A few twin drives with everything the GUI's training lab animates (gui/lab_tabs.py): the car's pose, its
    randomised parameters, the room, the label (contact within 2 s) and the v3 model's risk at every tick."""
    import json
    from adas.intent_net import IntentNet, window_of
    net_path = os.path.join(ROOT, "models", "intent_v3.json")
    net = IntentNet(net_path) if os.path.exists(net_path) else None
    rng = np.random.default_rng(77)
    fams = ["straight", "rooms", "curve", "parallel", "gap", "reverse"]
    out = []
    i = 0
    while len(out) < n and i < n * 6:
        fam = fams[i % len(fams)]
        r = run((990_000 + i, fam, level, False))
        i += 1
        if len(r["y"]) < 40:
            continue
        risk = []
        if net is not None:
            Z = r["Z"].astype(np.float64)
            risk = [round(net.risk(window_of(list(Z[max(0, k - 31):k + 1]))), 3) for k in range(len(Z))]
        out.append({"family": fam, "style": r["style"], "crashed": r["crashed"],
                    "pose": np.round(r["pose"], 3).tolist(), "y": r["y"].astype(int).tolist(), "risk": risk,
                    "v": np.round(r["v_true"], 3).tolist(), "segments": np.round(r["segments"], 3).tolist(),
                    "dr": {k: round(float(v), 3) for k, v in r["dr_dict"].items()}})
    path = path or os.path.join(ROOT, "models", "intent_training", "showcase.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"))
    return out


def ablation(train_runs=1500, test_runs=400):
    """Data for the randomisation ablation (RESEARCH.md section 7, step 6): a training set from the NOMINAL twin
    only (level 0), and two test sets - the nominal twin, and cars/sensors drawn from 1.6x wider ranges than any
    training set saw (a stand-in for the unknown real car)."""
    import time
    t0 = time.time()
    for name, n, seed0, level, rs in (("train_nominal", train_runs, 20_000, 0.0, 3),
                                       ("test_nominal", test_runs, 800_000, 0.0, 4),
                                       ("test_wide", test_runs, 700_000, 1.6, 5)):
        d = build(n, seed0, level, name.startswith("test"), os.path.join(OUT_DIR, name + ".npz"), rs)
        print(f"{name}: {len(d['y'])} ticks, {len(d['run_family'])} runs, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "showcase":
        print(len(showcase()), "showcase drives")
    elif len(sys.argv) > 1 and sys.argv[1] == "ablation":
        ablation(*(int(a) for a in sys.argv[2:4]))
    else:
        main(*(int(a) for a in sys.argv[1:3]), *(float(a) for a in sys.argv[3:4]))
