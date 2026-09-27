"""Driving-assist scenarios on the CAR'S OWN decision code (pi/relay_assists.py + pi/path_gate.py, as in the relay)
driving the digital twin (sim/hw_sim.py VirtualCar + the simulated LiDAR), each with a pass/fail criterion.

    python -m sim.relay_scenarios          prints the table, writes models/relay_scenarios.json

Replaces sim/assist_eval.py (which ran the older laptop pipeline) for checking what the car will actually do.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DT, SCAN_DT = 0.05, 0.1


def _tuning():
    from adas.config import load_tuning
    from pi.relay_assists import apply_car_model
    tun = load_tuning(os.path.join(ROOT, "pi", "tuning_real_car.json"))
    apply_car_model(tun)
    return tun


def run(world, driver, seconds, assists=(), start=(0.0, 0.0, 0.0), seed=0, stop_when=None):
    """driver(t, x, y, th, v) -> (stick -1..1, + = left, physical PWM, + = forward). Returns a record with a per-tick
    trace: t, x, y, th, v, pwm_in, pwm_out, stick_in, stick_out, gate action, evading."""
    from adas.aeb import SpeedEstimator
    from pi.path_gate import PathGate
    from pi.relay_assists import K_CURV_PER_SERVO_DEG as K, RelayAssists, RelayIntent
    from sim.hw_sim import SimLidar, VirtualCar
    tun = _tuning()
    assist = RelayAssists(tun)
    rint = RelayIntent(assist)                   # always running, as in the relay
    for a in assists:
        assist.set(a, True)
    p = assist.p
    car = VirtualCar(world, p, start, threaded=False)
    car.last_cmd_t = 0.0
    lidar = SimLidar(car, tun.mount.yaw_offset_deg, n=720, seed=seed)
    gate = PathGate(p, tun.speed_model)
    vest = SpeedEstimator(tun.speed_model)
    t, next_scan, seq, pts = 0.0, 0.0, 0, []
    rec = {"trace": [], "infos": set(), "max_level": 0, "min_clear": 9.0}
    while t < seconds:
        x, y, th, v, *_r, crashed = car.pose()
        if crashed:
            break
        if stop_when is not None and stop_when(t, x, y, th, v):
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
        stick, pwm = driver(t, x, y, th, v)
        servo = assist.stick_to_servo(stick)
        lines = [f"A {servo:.1f} {servo:.1f}", f"M {-int(pwm)}"]
        rint.update(t, servo, float(pwm), vest.v, pts)
        out = assist.process(lines, pts, seq, now=t)
        servo_out, phys = servo, float(pwm)
        for ln in out:
            q = ln.split()
            if q[0] == "A":
                servo_out = (float(q[1]) + float(q[2])) / 2
            elif q[0] == "M":
                phys = -float(q[1])
        delta = math.atan(-K * (servo_out - assist.centre) * p.wheelbase)
        g_phys, g_brake = gate.decide(DT, phys, delta, vest.v, 0.0, None)
        vest.update(DT, g_phys)
        rec["infos"].update(f"{k}: {s}" for k, s in assist.info.items())
        rec["max_level"] = max(rec["max_level"], assist.level)
        rec["trace"].append((t, x, y, th, v, pwm, g_phys, stick, assist.servo_to_stick(servo_out),
                             gate.info.get("action"), assist.assists.evading))
        car.command(f"A {servo_out:.1f} {servo_out:.1f}", now=t)
        car.command(f"M {-int(g_phys)}", now=t)
        t += DT
        car.step_to(t)
        rec["min_clear"] = min(rec["min_clear"], world.clearance(x, y, th, p))
    x, y, th, v, *_ = car.pose()
    rec.update(collided=car.crash_count > 0, x=x, y=y, th=th, v=v, evading_end=assist.assists.evading, p=p)
    return rec


def room(w, x0=-1.0, x1=8.0, y0=-1.6, y1=1.6):
    from sim.world import Wall
    for a, b, c, d in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        w.add(Wall(a, b, c, d))
    return w


def steady(pwm, stick=0.0, t0=0.3):
    return lambda t, x, y, th, v: (stick, pwm if t > t0 else 0.0)


def _world():
    from sim.world import World
    return room(World())


# ------------------------------------------------------------------ scenarios
def evasive_box():
    from sim.world import Box
    w = _world()
    w.add(Box(2.2, 0.0, 0.22, 0.22))
    on = run(w, steady(150), 12, ("evasive",))
    w = _world()
    w.add(Box(2.2, 0.0, 0.22, 0.22))
    off = run(w, steady(150), 12, ())
    ok = (not on["collided"]) and on["x"] > 2.8 and on["min_clear"] > 0.03 and not on["evading_end"] and \
        (not off["collided"]) and off["x"] < 2.0
    return ok, (f"with evasive: around the box (x {on['x']:.1f} m), closest {on['min_clear'] * 100:.0f} cm, handed back; "
                f"without: braked to a stop at x {off['x']:.2f} m")


def throttle_cuts(rec):
    """Ticks where an evasive manoeuvre was running but nothing reached the motor while the driver held the throttle."""
    tr = rec["trace"]
    return [(round(r[0], 2), r[9]) for r in tr if r[10] and r[5] > 0 and r[6] <= 0]


def doorway_full_throttle():
    from sim.hw_worlds import doorway
    w, start = doorway()
    r = run(w, steady(255), 14, ("evasive",), start=start, stop_when=lambda t, x, y, th, v: x > 3.6)
    cuts = throttle_cuts(r)
    ok = (not r["collided"]) and r["x"] > 3.0 and not cuts
    return ok, (f"full throttle at the box in front of the doorway: {'through the door' if r['x'] > 3.0 else f'stopped at x {r[chr(120)]:.2f}'}"
                f", closest {r['min_clear'] * 100:.0f} cm, throttle cut mid-manoeuvre on {len(cuts)} ticks"
                + (f" (first at {cuts[0][0]} s, gate: {cuts[0][1]})" if cuts else ""))


def evasive_driver_already_avoiding():
    from sim.world import Box
    w = _world()
    w.add(Box(2.2, 0.0, 0.22, 0.22))
    drv = lambda t, x, y, th, v: (0.35 if 0.8 < t < 2.2 else (-0.2 if 2.2 <= t < 3.4 else 0.0), 150 if t > 0.3 else 0.0)
    # until the car is past the box (afterwards this scripted driver freezes the stick and drifts at the side wall,
    # which is a real lapse the evasive steer rightly catches)
    r = run(w, drv, 10, ("evasive",), stop_when=lambda t, x, y, th, v: x > 2.6)
    ev = sum(1 for row in r["trace"] if row[10])
    ok = (not r["collided"]) and ev == 0 and r["x"] > 2.6
    return ok, f"driver steers around the box themselves: evasive took over for {ev} ticks, no contact"


def wall_full_speed(pwm=255):
    """B11: straight at a wall at full throttle, braking only - where does it stop?"""
    from sim.world import Wall
    w = _world()
    w.add(Wall(4.0, -1.6, 4.0, 1.6))
    r = run(w, steady(pwm), 10, ())
    gap = 4.0 - (r["x"] + r["p"].front_x)
    ok = (not r["collided"]) and 0.02 < gap < 0.35
    return ok, f"full throttle at a wall 4 m ahead: stopped with {gap * 100:.0f} cm to spare"


def corridor(assist):
    from sim.world import Wall, World
    w = World()
    w.add(Wall(-1.0, 0.36, 7.0, 0.36))
    w.add(Wall(-1.0, -0.36, 7.0, -0.36))
    w.add(Wall(7.0, -0.36, 7.0, 0.36))
    return run(w, steady(140, 0.06), 12, ("centring",) if assist else ())


def centring():
    on, off = corridor(True), corridor(False)
    max_y = max(abs(r[2]) for r in on["trace"])
    ok = (not on["collided"]) and max_y < 0.08 and on["x"] > 5.0 and (off["collided"] or off["x"] < on["x"] - 1.0)
    return ok, (f"steering drifts left in a 72 cm corridor: with centring within {max_y * 100:.0f} cm of the centre for "
                f"{on['x']:.1f} m; without: {'hits the wall' if off['collided'] else f'stopped at x {off[chr(120)]:.1f} m'}")


def centring_yields():
    from sim.world import Wall
    w = _world()
    w.add(Wall(-1.0, 0.45, 2.0, 0.45))
    w.add(Wall(-1.0, -0.45, 2.0, -0.45))
    turned = [False]

    def drv(t, x, y, th, v):                     # turns left as the front reaches the corridor's end (x = 2.0)
        turned[0] = turned[0] or x + 0.28 > 1.95
        return (0.6 if turned[0] else 0.0), (130 if t > 0.3 else 0.0)
    r = run(w, drv, 9.0, ("centring",), stop_when=lambda t, x, y, th, v: y > 0.6)
    ok = (not r["collided"]) and r["y"] > 0.5
    return ok, f"driver turns left out of the corridor's end on purpose: centring lets go (reaches {r['y']:.2f} m to the left)"


def limiter():
    on = run(_world(), steady(255, 1.0), 6, ("limiter",))
    off = run(_world(), steady(255, 1.0), 6, ())
    from pi.relay_assists import K_CURV_PER_SERVO_DEG as K

    def lat(rec):
        return max((row[4] ** 2 * abs(math.tan(__import__("adas.vehicle_params", fromlist=["x"]).steer_to_delta(row[8], rec["p"])) / rec["p"].wheelbase)
                    for row in rec["trace"] if row[0] > 2.0), default=0.0)
    a_on, a_off = lat(on), lat(off)
    ok = a_on <= 1.2 * 1.15 and a_off > a_on * 1.2
    return ok, f"full throttle at full lock: lateral acceleration {a_on:.2f} m/s^2 with the limiter, {a_off:.2f} without"


def gap(width_m, assists=("narrow",)):
    from sim.world import Box
    w = _world()
    half = width_m / 2
    w.add(Box(2.4, half + 0.15, 0.3, 0.3))
    w.add(Box(2.4, -half - 0.15, 0.3, 0.3))
    return run(w, steady(150), 10, assists)


def narrow_wont_fit():
    r = gap(0.19)
    ok = (not r["collided"]) and r["x"] < 2.3
    return ok, f"19 cm gap for a 20 cm car: stopped before it (x {r['x']:.2f} m), no contact, warning level {r['max_level']}"


def narrow_tight_fits():
    r = gap(0.32)
    ok = (not r["collided"]) and r["x"] > 3.0
    return ok, f"32 cm gap: passed through (x {r['x']:.1f} m), closest {r['min_clear'] * 100:.0f} cm"


def proximity():
    from sim.world import Box
    w = _world()
    w.add(Box(0.1, 0.30, 0.6, 0.18))
    drv = lambda t, x, y, th, v: (0.5 if t > 1.0 else 0.0, 0.0)
    r = run(w, drv, 2.0, ("proximity",))
    ok = r["max_level"] >= 2 and any("steering toward" in i for i in r["infos"])
    return ok, "object beside the parked car: side alert, raised to a warning when the driver steers toward it"


def passing_beside_a_wall():
    """No needless interruption: driving parallel to a wall 15 cm off the side must not be slowed."""
    from sim.world import Wall
    w = _world()
    w.add(Wall(-1.0, 0.25, 6.0, 0.25))
    r = run(w, steady(180), 5, ("evasive", "centring", "limiter", "narrow", "proximity"))
    slowed = sum(1 for row in r["trace"] if row[0] > 0.5 and row[6] < row[5] - 1)
    ok = (not r["collided"]) and slowed == 0
    return ok, f"parallel to a wall 15 cm off the side, all assists on: throttle reduced on {slowed} ticks"


SCENARIOS = [("Evasive steer around a block", evasive_box),
             ("Doorway at full throttle: no throttle cut mid-manoeuvre (B4)", doorway_full_throttle),
             ("Evasive stays out when the driver is already avoiding", evasive_driver_already_avoiding),
             ("Full speed at a wall: stops close, not early (B11)", wall_full_speed),
             ("Corridor centring", centring),
             ("Centring yields to a deliberate turn", centring_yields),
             ("Speed-vs-steering limiter", limiter),
             ("Narrow gap: won't fit", narrow_wont_fit),
             ("Narrow gap: tight but fits", narrow_tight_fits),
             ("Side proximity alert", proximity),
             ("No needless slowing beside a wall", passing_beside_a_wall)]


def main(names=None):
    out = []
    for name, fn in SCENARIOS:
        if names and not any(n.lower() in name.lower() for n in names):
            continue
        ok, detail = fn()
        out.append({"name": name, "pass": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}", flush=True)
    if not names:
        os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
        json.dump(out, open(os.path.join(ROOT, "models", "relay_scenarios.json"), "w"), indent=1)
    print(f"\n{sum(r['pass'] for r in out)}/{len(out)} passed")
    return 0 if all(r["pass"] for r in out) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
