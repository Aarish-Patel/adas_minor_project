"""Physics-informed ML car model (adas/car_model.py) on a real logging drive, with held-out legs.

    python -m sim.car_model_eval [LOG] [--save pi/car_model.json]

Training uses some legs; the held-out legs (other PWMs and steering angles) test interpolation/extrapolation.
Compared: the earlier first-order fit (sim/log_fit.py), pure physics, physics + ML. Metrics: speed error when the
held-out commands are replayed open loop, and where the car ends up 3 s later vs where it really went.
"""
import argparse
import glob
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adas.car_model import CarModel, training_data  # noqa: E402
from pi.drive_log import load  # noqa: E402
from sim.log_fit import command_series, fit_speed, hold, kinematics, reconstruct, simulate_speed  # noqa: E402

TEST_LEGS = {1, 3, 6, 7}          # PWM 130, PWM 170, wheels +16, wheels -16 (pi/log_drive.py LEGS order)


def leg_windows(log):
    ev = [e for e in log["ev"] if e["msg"] in ("leg forward", "leg reverse")]
    out = []
    for i, e in enumerate(ev):
        t1 = ev[i + 1]["t"] if i + 1 < len(ev) else e["t"] + 6
        out.append((e["leg"], e["t"], t1))
    return out


def in_windows(t, windows):
    m = np.zeros(len(t), bool)
    for _, a, b in windows:
        m |= (t >= a) & (t < b)
    return m


def rollout(model, kind, t0, t1, x0, y0, th0, v0, cmd_servo, cmd_pwm, old=None, dt=0.01):
    """Open-loop replay of the logged commands (pi frame: th positive = right)."""
    x, y, th, v = x0, y0, th0, v0
    t = t0
    delay = old["delay_s"] if kind == "old" else model.delay
    while t < t1:
        u = float(hold(cmd_pwm, t - delay))
        s = float(hold(cmd_servo, t - delay - 0.05))
        if kind == "old":
            vmax, db, tau, coast = old["v_max"], old["deadband"], old["tau_motor"], old["coast_decel"]
            if u == 0:
                v -= math.copysign(min(abs(v), min(abs(v) / tau, coast) * dt), v) if v else 0.0
            else:
                vss = math.copysign(vmax * min(1.0, max(0.0, (abs(u) - db) / (255.0 - db))), u)
                v += max(-4.5, min(3.0, (vss - v) / tau)) * dt
            kap = old["k"] * (s - old["s0"])
        else:
            model.use_ml = kind == "ml"
            v = model.step_speed(v, u, dt)
            kap = float(model.curvature(s, abs(v)))
        th += kap * v * dt
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        t += dt
    return x, y, th, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    path = a.log or sorted(glob.glob(os.path.join(root, "logs", "*log_drive*.jsonl.gz")), key=os.path.getsize)[-1]
    log = load(path)
    servo, pwm = command_series(log)
    rows = reconstruct(log)
    wins = leg_windows(log)
    test_w = [w for w in wins if w[0] in TEST_LEGS]
    train_w = [w for w in wins if w[0] not in TEST_LEGS]
    t = rows[:, 0]
    tr_rows = rows.copy()
    tr_rows[~in_windows(t, train_w), 4] = 0                 # mark held-out time as untrusted for training
    data = training_data(tr_rows, servo, pwm)

    tv, v_obs, w_obs, good = kinematics(tr_rows)
    model = CarModel()
    model.fit_speed(tv, v_obs, good, pwm)
    model.fit_steering(data["servo"], data["v_turn"], data["kappa"])

    # the earlier model, fitted on the same training legs
    sp = fit_speed(tv, v_obs, good, pwm)
    sel = good & (np.abs(v_obs) > 0.08)
    s_at = hold(servo, tv - sp["delay_s"] - 0.05)
    kk, b0 = np.polyfit(s_at[sel], (w_obs / np.where(np.abs(v_obs) > 1e-3, v_obs, 1))[sel], 1)
    old = {**sp, "k": kk, "s0": -b0 / kk}

    # evaluate: 3 s open-loop windows starting inside held-out legs
    tv_all, v_all, _, good_all = kinematics(rows)
    errs = {"old": [], "physics": [], "ml": []}
    verr = {"old": [], "physics": [], "ml": []}
    starts = [i for i in range(len(t)) if rows[i, 4] > 0 and in_windows(t[i:i + 1], test_w)[0]]
    for i in starts[::3]:
        j = np.searchsorted(t, t[i] + 3.0)
        if j >= len(t) or rows[j, 4] == 0:
            continue
        v0 = float(np.interp(t[i], tv_all, v_all))
        for kind in errs:
            x, y, th, v = rollout(model, kind, t[i], t[j], rows[i, 1], rows[i, 2], rows[i, 3], v0, servo, pwm, old)
            errs[kind].append(math.hypot(x - rows[j, 1], y - rows[j, 2]))
            verr[kind].append(abs(v - float(np.interp(t[j], tv_all, v_all))))
    print(f"log {os.path.basename(path)}: {len(data['u'])} speed samples, {len(data['kappa'])} steering samples "
          f"(training legs only); {len(errs['ml'])} held-out 3 s windows")
    print("physics fit:", {k: round(v, 4) for k, v in model.report["speed"].items()})
    print("steering fit:", {k: round(v, 4) for k, v in model.report["steering"].items()})
    print(f"{'model':28s} {'3 s position error median / 90th':>34s}   speed error median")
    for kind, name in (("old", "earlier first-order fit"), ("physics", "physics (DC motor + Ackermann)"),
                       ("ml", "physics + ML correction")):
        e = np.array(errs[kind]) * 100
        print(f"{name:28s} {np.median(e):14.1f} cm / {np.percentile(e, 90):5.1f} cm        {np.median(verr[kind]) * 100:5.1f} cm/s")
    if a.save:
        full = training_data(rows, servo, pwm)
        m = CarModel()
        m.fit_speed(tv_all, v_all, good_all, pwm)
        m.fit_steering(full["servo"], full["v_turn"], full["kappa"])
        m.report["source_log"] = os.path.basename(path)
        m.report["held_out_eval_cm"] = {k: float(np.median(v) * 100) for k, v in errs.items()}
        # use the ML correction only if it actually beat pure physics on the held-out legs
        m.use_ml = np.median(errs["ml"]) < np.median(errs["physics"])
        m.report["selected"] = "physics + ML" if m.use_ml else "physics"
        m.save(a.save)
        print("saved", a.save)


if __name__ == "__main__":
    main()
