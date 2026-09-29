"""Closed-loop vs open-loop speed tracking on the twin, with the car off its model (battery / carpet / gearbox).

    python -m sim.speed_ctrl_eval           # writes models/speed_ctrl.json

The car follows target-speed steps 0.30 -> 0.55 -> 0.25 -> -0.20 (reverse creep) -> 0. The speed the controller sees is the
true speed plus 4 cm/s noise through a 0.15 s low-pass (about what the relay's EKF delivers). Reported: RMS tracking error
over the settled parts of each step, for the car being 30 % slower ... 15 % faster than its model.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

STEPS = [(0.0, 0.0), (0.5, 0.30), (4.5, 0.55), (8.5, 0.25), (12.5, -0.20), (15.5, 0.0), (17.0, 0.0)]


def target_at(t):
    v = 0.0
    for t0, val in STEPS:
        if t >= t0:
            v = val
    return v


def run(scale, closed, seed=0):
    from adas.config import load_tuning
    from adas.speed_control import SpeedController
    from pi.relay_assists import car_params
    from sim.hw_sim import VirtualCar
    from sim.world import World
    tun = load_tuning(os.path.join(HERE, "..", "pi", "tuning_real_car.json"))
    p = car_params(tun.mount)
    car = VirtualCar(World(), p, (0.0, 0.0, 0.0), threaded=False)
    car.m = dict(car.m, v_max=car.m["v_max"] * scale)
    car.last_cmd_t = 0.0
    ctl = SpeedController(tun.speed_model)
    rng = np.random.default_rng(seed)
    dt, t, meas = 0.05, 0.0, 0.0
    err = []
    while t < STEPS[-1][0]:
        v_t = target_at(t)
        u = ctl.pwm(v_t, meas, dt) if closed else (np.sign(v_t) * tun.speed_model.pwm_for_speed(abs(v_t)) if v_t else 0.0)
        car.command(f"M {-int(u)}", now=t)
        t += dt
        car.step_to(t)
        v = car.pose()[3]
        meas += (v + rng.normal(0, 0.04) - meas) * (dt / 0.15)
        settled = t - max(t0 for t0, _ in STEPS if t >= t0) > 1.5
        if settled and v_t != 0.0:
            err.append(v - v_t)
    e = np.array(err)
    return {"rmse_cm_s": float(100 * np.sqrt(np.mean(e ** 2))), "bias_cm_s": float(100 * np.mean(e))}


def main():
    out = {}
    print(f"{'car vs model':>14} | {'open loop: bias / RMSE (cm/s)':>32} | {'closed loop: bias / RMSE (cm/s)':>34}")
    for scale in (0.70, 0.85, 1.0, 1.15):
        o, c = run(scale, False), run(scale, True)
        out[str(scale)] = {"open": o, "closed": c}
        print(f"{scale:14.2f} | {o['bias_cm_s']:14.1f} {o['rmse_cm_s']:14.1f}   | {c['bias_cm_s']:16.1f} {c['rmse_cm_s']:16.1f}")
    json.dump(out, open(os.path.join(HERE, "..", "models", "speed_ctrl.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
