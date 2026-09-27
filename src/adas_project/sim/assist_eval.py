"""Driving-assist scenarios on the real-car profile, each with a pass/fail criterion.

    python -m sim.assist_eval            prints the table, writes models/assist_eval.json

Every assist is checked for what it should do AND for staying out of the way when the driver is already
handling it (the intent-aware part).
"""
import json
import math
import os
import sys

from .demo_tests import make_sim
from .intent_data import MODEL_DIR
from .world import Box, Wall, World

ALL = ("evasive", "centring", "limiter", "narrow", "proximity")


def room(w, x0=-1.0, x1=8.0, y0=-1.6, y1=1.6):
    for a, b, c, d in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        w.add(Wall(a, b, c, d))
    return w


def run(world, driver, seconds, assists=(), start=(0.0, 0.0, 0.0), mode="active", seed=1):
    sim, rp = make_sim(world, start=start, mode=mode, seed=seed)
    for k in ALL:
        sim.adas.assists.enabled[k] = k in assists
    rec = {"evading_ticks": 0, "max_level": 0, "max_lat_acc": 0.0, "infos": set(), "max_abs_y": 0.0,
           "min_speed_in": math.inf, "yield_ok": True}
    while sim.t < seconds and not sim.car.collided:
        steer, pwm = driver(sim.t, sim)
        sim.step(steer, pwm)
        a = sim.adas.assists
        rec["evading_ticks"] += int(a.evading)
        rec["max_level"] = max(rec["max_level"], sim.adas.assist_level)
        rec["infos"].update(f"{k}: {v}" for k, v in a.info.items())
        if sim.t > 2.0:
            k = math.tan(__import__("adas.vehicle_params", fromlist=["x"]).steer_to_delta(sim.applied_steer, sim.p)) / sim.p.wheelbase
            rec["max_lat_acc"] = max(rec["max_lat_acc"], sim.car.v ** 2 * abs(k))
        rec["max_abs_y"] = max(rec["max_abs_y"], abs(sim.car.y))
        rec["trace_last"] = (sim.t, sim.car.x, sim.car.y, sim.car.v)
    rec["collided"] = bool(sim.car.collided)
    rec["min_clear"] = float(max(sim.min_clearance, 0.0))
    rec["x"], rec["y"], rec["v"] = sim.car.x, sim.car.y, sim.car.v
    rec["evading_end"] = sim.adas.assists.evading
    return sim, rec


def steady(pwm, steer=0.0, t0=0.3):
    return lambda t, s: (steer, pwm if t > t0 else 0.0)


# ------------------------------------------------------------------ scenarios
def evasive_box():
    world = room(World())
    world.add(Box(2.2, 0.0, 0.22, 0.22))
    _, on = run(world, steady(150), 12, ("evasive",))
    world = room(World())
    world.add(Box(2.2, 0.0, 0.22, 0.22))
    _, off = run(world, steady(150), 12, ())
    ok = (not on["collided"]) and on["x"] > 2.8 and on["min_clear"] > 0.03 and not on["evading_end"] and \
        (not off["collided"]) and off["x"] < 2.0
    return ok, (f"with assist: steered around, passed the block (x {on['x']:.1f} m), closest {on['min_clear'] * 100:.0f} cm, "
                f"control handed back; without: emergency braking stopped at x {off['x']:.2f} m")


def evasive_driver_already_avoiding():
    world = room(World())
    world.add(Box(2.2, 0.0, 0.22, 0.22))
    drv = lambda t, s: (0.35 if 0.8 < t < 2.2 else (-0.2 if 2.2 <= t < 3.4 else 0.0), 150 if t > 0.3 else 0.0)
    _, r = run(world, drv, 10, ("evasive",))
    ok = (not r["collided"]) and r["evading_ticks"] == 0
    return ok, f"driver steered around it themselves: assist intervened for {r['evading_ticks']} ticks, no contact"


def evasive_blocked():
    world = room(World())
    world.add(Wall(2.2, -1.6, 2.2, 1.6))
    _, r = run(world, steady(150), 10, ("evasive",))
    ok = (not r["collided"]) and r["x"] < 2.2
    return ok, f"wall across the whole room: no escape path, emergency braking stopped it {max(0, 2.2 - r['x'] - 0.28) * 100:.0f} cm short"


def corridor(assist):
    world = World()
    world.add(Wall(-1.0, 0.36, 7.0, 0.36))
    world.add(Wall(-1.0, -0.36, 7.0, -0.36))
    world.add(Wall(7.0, -0.36, 7.0, 0.36))
    return run(world, steady(140, 0.06), 12, ("centring",) if assist else ())


def centring():
    _, on = corridor(True)
    _, off = corridor(False)
    ok = (not on["collided"]) and on["max_abs_y"] < 0.08 and on["x"] > 5.0 and (off["collided"] or off["x"] < on["x"] - 1.0)
    return ok, (f"driver's steering drifts left: with centring the car stays within {on['max_abs_y'] * 100:.0f} cm of the "
                f"centre for {on['x']:.1f} m; without it {'hits the wall' if off['collided'] else f'is stopped by braking at {off[chr(120)]:.1f} m'}")


def centring_yields():
    world = room(World())
    world.add(Wall(-1.0, 0.45, 2.0, 0.45))
    world.add(Wall(-1.0, -0.45, 2.0, -0.45))
    drv = lambda t, s: (0.6 if t > 2.0 else 0.0, 130 if t > 0.3 else 0.0)
    sim, r = run(world, drv, 4.5, ("centring",))
    ok = (not r["collided"]) and r["y"] > 0.3
    return ok, f"driver turns out of the corridor on purpose: centring lets go (ends {r['y']:.2f} m to the left)"


def limiter():
    _, on = run(room(World()), steady(255, 1.0), 6, ("limiter",))
    _, off = run(room(World()), steady(255, 1.0), 6, ())
    ok = on["max_lat_acc"] <= 1.2 * 1.15 and off["max_lat_acc"] > on["max_lat_acc"] * 1.2
    return ok, f"full throttle at full lock: lateral acceleration {on['max_lat_acc']:.2f} m/s^2 with the limiter, {off['max_lat_acc']:.2f} without"


def gap(width_m, assists=("narrow",)):
    world = room(World())
    half = width_m / 2
    world.add(Box(2.4, half + 0.15, 0.3, 0.3))
    world.add(Box(2.4, -half - 0.15, 0.3, 0.3))
    return run(world, steady(150), 10, assists)


def narrow_wont_fit():
    _, r = gap(0.19)
    ok = (not r["collided"]) and r["max_level"] >= 2 and r["x"] < 2.3
    return ok, f"18 cm gap for a 20 cm car: 'won't fit' warning, stopped before it (x {r['x']:.2f} m), no contact"


def narrow_tight_fits():
    sim, r = gap(0.32)
    ok = (not r["collided"]) and r["x"] > 3.0
    return ok, f"32 cm gap: slowed, passed through (x {r['x']:.1f} m), closest {r['min_clear'] * 100:.0f} cm"


def proximity():
    world = room(World())
    world.add(Box(0.1, 0.30, 0.6, 0.18))              # object alongside, left
    drv = lambda t, s: (0.5 if t > 1.0 else 0.0, 0.0)  # parked, driver starts steering left
    _, r = run(world, drv, 2.0, ("proximity",))
    ok = r["max_level"] >= 2 and any("steering toward" in i for i in r["infos"])
    return ok, "object beside the car: side alert, raised to a warning when the driver steers toward it"


def deadman():
    from adas.assists import deadman_pwm
    pwm, t, series = 200.0, 0.0, []
    while pwm > 0 and t < 3:
        pwm = deadman_pwm(pwm, input_age=t, dt=0.02)
        series.append(pwm)
        t += 0.02
    steps = [abs(series[i] - series[i - 1]) for i in range(1, len(series))]
    ok = series[0] == 200 and pwm == 0 and max(steps) <= 300 * 0.02 + 1e-9 and t < 1.3
    return ok, f"driver link lost at PWM 200: held 0.5 s, then ramped to 0 by {t:.2f} s (largest step {max(steps):.0f} PWM)"


SCENARIOS = [("Evasive steer around a block", evasive_box),
             ("Evasive steer stays out when the driver is already avoiding", evasive_driver_already_avoiding),
             ("Evasive steer: no way around -> braking only", evasive_blocked),
             ("Corridor centring", centring),
             ("Centring yields to a deliberate turn", centring_yields),
             ("Speed-vs-steering limiter", limiter),
             ("Narrow gap: won't fit", narrow_wont_fit),
             ("Narrow gap: tight but fits", narrow_tight_fits),
             ("Side proximity alert", proximity),
             ("Dead-man smooth stop", deadman)]


def main():
    out = []
    for name, fn in SCENARIOS:
        ok, detail = fn()
        out.append({"name": name, "pass": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}", flush=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "assist_eval.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n{sum(r['pass'] for r in out)}/{len(out)} passed")
    return 0 if all(r["pass"] for r in out) else 1


if __name__ == "__main__":
    sys.exit(main())
