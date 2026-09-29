"""Fit the simulator's car model to real drive logs (pi/drive_log.py format).

    python -m sim.log_fit LOG [LOG ...] [--out sim/fitted_car.json]

1. Rebuild the car's path from the logged LiDAR scans (scan matching, pi/scanmatch.py). Only scan steps
   that came from a real scan match are used for fitting, never ones filled in from the speed prediction.
2. Speed model: first-order motor lag, dead-band, top speed, coast-down deceleration and command delay,
   fitted to the measured speed under the logged motor PWM.
3. Steering: path curvature per servo degree and the straight-ahead servo angle, from yaw rate / speed.
4. Validation: from many start points, re-drive the logged commands through the fitted model for 3 s
   (open loop) and measure how far that lands from where the car really went.

Writes the fitted values, the ranges the logs covered (anything outside them is extrapolation) and the
validation errors. sim/real_car.py uses the file automatically when it exists.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pi.drive_log import load  # noqa: E402
from pi.scanmatch import Odometry, polar_to_xy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "fitted_car.json")
DT = 0.01


# ------------------------------------------------------------------ commands
def command_series(log):
    tun = (log["meta"] or {}).get("tuning") or {}
    reversed_motor = (tun.get("servo") or {}).get("motor_reversed", True)
    servo, pwm = [], []
    for t, line in log["cmds"]:
        p = line.split()
        try:
            if p[0] == "A" and len(p) == 3:
                servo.append((t, (float(p[1]) + float(p[2])) / 2.0))
            elif p[0] == "M" and len(p) == 2:
                w = float(p[1])
                pwm.append((t, -w if reversed_motor else w))
            elif p[0] == "STOP":
                pwm.append((t, 0.0))
        except (ValueError, IndexError):
            pass
    return np.array(servo or [(0.0, 90.0)]), np.array(pwm or [(0.0, 0.0)])


def hold(series, t):
    """Value of a step signal (commands hold until the next one) at time(s) t."""
    i = np.searchsorted(series[:, 0], t, side="right") - 1
    return np.where(i >= 0, series[np.clip(i, 0, len(series) - 1), 1], series[0, 1])


# ------------------------------------------------------------------ path from scans
def reconstruct(log, yaw=None, centre=None):
    tun = (log["meta"] or {}).get("tuning") or {}
    mount = tun.get("mount") or {}
    yaw = mount.get("yaw_offset_deg", 90.0) if yaw is None else yaw
    dmin = mount.get("min_valid_range_m", 0.2)
    centre = (tun.get("servo") or {}).get("left_center", 90.0) if centre is None else centre
    servo, pwm = command_series(log)
    odo = Odometry()
    rows = []
    for t, ang, dist, _q in log["scans"]:
        pts = []
        for a, d in zip(ang, dist):
            if d >= dmin:
                c = (a - yaw) % 360.0
                pts.append((c if c <= 180 else c - 360, d))
        xy = polar_to_xy(pts)
        u = float(hold(pwm, t))
        v_pred = math.copysign(max(0.0, (abs(u) - 45) / 210.0) * 0.9, u) if u else 0.0   # rough, ICP init only
        kappa = 0.068 * (float(hold(servo, t)) - centre)
        pose = odo.update(xy, t, v_pred, kappa)
        rows.append((t, pose[0], pose[1], pose[2], 1.0 if odo.last_source in (None, "icp") else 0.0))
    return np.array(rows)          # t, x, y, th, trusted


def kinematics(path):
    """Speed (forward component, signed) and yaw rate between consecutive scans, 3-scan smoothed."""
    t, x, y, th, ok = path.T
    dt = np.diff(t)
    thm = th[:-1] + np.diff(th) / 2
    ds = np.diff(x) * np.cos(thm) + np.diff(y) * np.sin(thm)
    v = ds / dt
    w = np.diff(th) / dt
    good = (ok[1:] > 0) & (ok[:-1] > 0) & (dt > 0.02) & (dt < 0.4)
    k = np.ones(3) / 3
    vs = np.convolve(v, k, mode="same")
    ws = np.convolve(w, k, mode="same")
    good[0] = good[-1] = False
    return (t[:-1] + t[1:]) / 2, vs, ws, good


# ------------------------------------------------------------------ speed model
def simulate_speed(params, pwm, t0, t1, v0=0.0, accel_max=3.0):
    vmax, db, tau, coast, delay = params
    ts = np.arange(t0, t1, DT)
    u = hold(pwm, ts - delay)
    v = np.empty_like(ts)
    cur = v0
    for i, ui in enumerate(u):
        if ui == 0:
            cur -= math.copysign(min(abs(cur), min(abs(cur) / tau, coast) * DT), cur) if cur else 0.0
        else:
            vss = math.copysign(vmax * min(1.0, max(0.0, (abs(ui) - db) / (255.0 - db))), ui)
            a = max(-accel_max * 1.5, min(accel_max, (vss - cur) / tau))
            cur += a * DT
        v[i] = cur
    return ts, v


def fit_speed(tv, v, good, pwm):
    from scipy.optimize import least_squares
    t0, t1 = tv[0] - 0.5, tv[-1] + 0.1
    sel = good

    def resid(p, delay):
        ts, vs = simulate_speed((*p, delay), pwm, t0, t1)
        return np.interp(tv[sel], ts, vs) - v[sel]

    best = None
    for delay in np.arange(0.0, 0.32, 0.04):
        r = least_squares(resid, x0=[1.0, 40.0, 0.2, 2.0], args=(delay,),
                          bounds=([0.2, 0.0, 0.02, 0.2], [3.0, 150.0, 1.0, 12.0]))
        if best is None or r.cost < best[0].cost:
            best = (r, delay)
    r, delay = best
    rms = float(np.sqrt(np.mean(r.fun ** 2)))
    vmax, db, tau, coast = (float(x) for x in r.x)
    return {"v_max": vmax, "deadband": db, "tau_motor": tau, "coast_decel": coast, "delay_s": float(delay),
            "rms_speed_error": rms, "n": int(sel.sum())}


# ------------------------------------------------------------------ steering
def fit_steer(tv, v, w, good, servo, delay):
    s = hold(servo, tv - delay - 0.05)          # servo also takes a moment to reach the angle
    sel = good & (np.abs(v) > 0.08)
    if sel.sum() < 10 or np.ptp(s[sel]) < 4:
        return {"identifiable": False, "reason": "not enough turning in the logs", "n": int(sel.sum())}
    kap = w[sel] / v[sel]
    A = np.stack([s[sel], np.ones(sel.sum())], 1)
    (k, b), *_ = np.linalg.lstsq(A, kap, rcond=None)
    pred = A @ np.array([k, b])
    ss = float(np.sum((kap - kap.mean()) ** 2)) or 1e-9
    return {"identifiable": True, "k_curv_per_deg": float(k), "servo_centre": float(-b / k),
            "r2": float(1 - np.sum((kap - pred) ** 2) / ss), "n": int(sel.sum()),
            "servo_range": [float(s[sel].min()), float(s[sel].max())]}


# ------------------------------------------------------------------ validation
def validate(path, tv, v, good, servo, pwm, sp, st, horizon=3.0, every=1.0):
    t, x, y, th, ok = path.T
    errs, heading_errs = [], []
    k = st.get("k_curv_per_deg", 0.068) if st.get("identifiable") else 0.068
    c = st.get("servo_centre", 90.0) if st.get("identifiable") else 90.0
    next_t = t[0] + 1.0
    for i in range(len(t)):
        if t[i] < next_t or t[i] + horizon > t[-1] or not ok[i]:
            continue
        next_t = t[i] + every
        j = np.searchsorted(t, t[i] + horizon)
        if j >= len(t) or not ok[j]:
            continue
        v0 = float(np.interp(t[i], tv, v))
        ts, vs = simulate_speed((sp["v_max"], sp["deadband"], sp["tau_motor"], sp["coast_decel"], sp["delay_s"]),
                                pwm, t[i], t[j], v0=v0)
        px, py, pth = x[i], y[i], th[i]
        ss = hold(servo, ts - sp["delay_s"] - 0.05)
        for vi, si in zip(vs, ss):
            pth += vi * k * (si - c) * DT
            px += vi * math.cos(pth) * DT
            py += vi * math.sin(pth) * DT
        errs.append(math.hypot(px - x[j], py - y[j]))
        heading_errs.append(abs(math.degrees((pth - th[j] + math.pi) % (2 * math.pi) - math.pi)))
    if not errs:
        return {"windows": 0}
    return {"windows": len(errs), "horizon_s": horizon, "median_pos_err_m": float(np.median(errs)),
            "p90_pos_err_m": float(np.percentile(errs, 90)), "median_heading_err_deg": float(np.median(heading_errs))}


# ------------------------------------------------------------------ main
def fit_logs(paths, out=DEFAULT_OUT, yaw=None, centre=None):
    parts = []
    for p in paths:
        log = load(p)
        if len(log["scans"]) < 20:
            print(f"{p}: only {len(log['scans'])} scans, skipped")
            continue
        servo, pwm = command_series(log)
        path = reconstruct(log, yaw, centre)
        tv, v, w, good = kinematics(path)
        parts.append((p, log, servo, pwm, path, tv, v, w, good))
    if not parts:
        raise SystemExit("no usable logs")
    # fit on the longest log with motion (speed model needs one continuous command history)
    parts.sort(key=lambda q: -int(q[8].sum()))
    p, log, servo, pwm, path, tv, v, w, good = parts[0]
    sp = fit_speed(tv, v, good, pwm)
    st = fit_steer(tv, v, w, good, servo, sp["delay_s"])
    val = [validate(q[4], q[5], q[6], q[8], q[2], q[3], sp, st) for q in parts]
    moving = pwm[pwm[:, 1] != 0, 1]
    result = {
        "logs": [os.path.basename(q[0]) for q in parts],
        "speed": sp, "steer": st, "validation": val,
        "coverage": {"pwm_range": [float(moving.min()), float(moving.max())] if len(moving) else None,
                     "max_speed_seen": float(np.abs(v[good]).max()) if good.any() else None,
                     "trusted_scan_fraction": float(path[:, 4].mean()), "seconds": float(path[-1, 0] - path[0, 0])},
    }
    truth = (log["meta"] or {}).get("truth")
    if truth:
        result["truth"] = truth
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--yaw", type=float, default=None, help="override the LiDAR yaw offset stored in the log")
    a = ap.parse_args()
    r = fit_logs(a.logs, a.out, a.yaw)
    sp, st = r["speed"], r["steer"]
    print(f"speed: v_max {sp['v_max']:.3f} m/s, deadband {sp['deadband']:.1f} PWM, lag {sp['tau_motor']:.3f} s, "
          f"coast {sp['coast_decel']:.2f} m/s^2, delay {sp['delay_s']:.2f} s   (rms {sp['rms_speed_error']:.3f} m/s, n={sp['n']})")
    if st.get("identifiable"):
        print(f"steer: {st['k_curv_per_deg']:.4f} rad/m per servo deg, straight at {st['servo_centre']:.1f} deg "
              f"(r2 {st['r2']:.2f}, n={st['n']}, servo {st['servo_range'][0]:.0f}-{st['servo_range'][1]:.0f})")
    else:
        print("steer: not identifiable -", st["reason"])
    for name, vv in zip(r["logs"], r["validation"]):
        if vv.get("windows"):
            print(f"validation {name}: {vv['windows']} x {vv['horizon_s']:.0f} s open-loop: median {vv['median_pos_err_m'] * 100:.1f} cm, "
                  f"90th {vv['p90_pos_err_m'] * 100:.1f} cm, heading {vv['median_heading_err_deg']:.1f} deg")
    if "truth" in r:
        print("truth (synthetic log):", r["truth"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
