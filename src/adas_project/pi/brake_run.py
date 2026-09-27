"""Measured stopping distance: cruise at a steady speed, cut throttle, record how far the
car travels before it stops (includes the real command/motor latency). Fits
  stop_dist(v) = v*T + v^2/(2*a)
which is exactly the shape wifi_drive_safety.required_margin() uses (REACTION_TIME_S=T,
ASSUMED_DECEL=a) - so this replaces two guesses with measured numbers."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/pi/rc_car")
import pi.lidar_steering_diag as D  # noqa: E402
from pi.lidar_steering_diag import Rig, floor_violated, TICK_S  # noqa: E402
from pi.speed_run import back_up_to_rear_limit, unique_scan_sample
from pi.test_gui import TestGui, hold, save_result  # noqa: E402

D.RAMP_STEP_PWM = 25
LEVELS = [90, 110, 130, 150, 180]
V_REF = ([90, 110, 130, 150, 180, 210], [0.163, 0.258, 0.353, 0.480, 0.592, 0.666])  # speed_run.py measurements
CUT_FRONT_M = 0.95
REPORT = "/home/pi/rc_car/pi/brake_run_report.json"


def run_level(rig, pwm, gui):
    gui.set(activity='FINDING OPEN SPACE - backing up to get run-up room', servo=float(D.CENTER0), message=f'stopping test at PWM {pwm}')
    back_up_to_rear_limit(rig)
    gui.set(activity=f'TESTING BRAKING - cruising at PWM {pwm}, then cutting throttle')
    f0 = rig.front()
    if f0 is None or f0 < 1.0:
        return {"pwm": pwm, "skipped": f"front room {f0}"}
    samples, last_t, t0 = [], None, rig.now()
    ramp_s = pwm / D.RAMP_STEP_PWM * TICK_S
    cut = None
    while rig.now() - t0 < 4.0:
        f = rig.front()
        if not rig.is_fresh() or (f is not None and f < 0.42):
            break        # only the FRONT matters while driving forward away from the rear wall
        if rig.now() - t0 > ramp_s + 0.25 + 0.75 or (f is not None and f < 0.72 and rig.now() - t0 > ramp_s + 0.5):
            break
        rig.motor_forward(pwm)
        last_t, s = unique_scan_sample(rig, last_t)
        if s:
            samples.append(s)
        time.sleep(TICK_S)
    cruise = [(t - t0, f) for t, f in samples if t - t0 > ramp_s + 0.25]
    if len(cruise) < 5:
        rig.stop()
        return {"pwm": pwm, "skipped": "too little cruise data"}
    ts = np.array([a for a, _ in cruise]); fs = np.array([b for _, b in cruise])
    v = float(-np.polyfit(ts, fs, 1)[0])
    f_cut = float(fs[-1])
    t_cut_wall = rig.now()
    rig.stop()                                  # immediate M 0 (coast)
    rest, last_t2 = [], last_t
    end = time.time() + 1.8
    while time.time() < end:
        last_t2, s = unique_scan_sample(rig, last_t2)
        if s:
            rest.append(s)
        time.sleep(0.03)
    f_rest = float(np.median([f for _, f in rest[-4:]])) if rest else None
    return {"pwm": pwm, "v_cruise": v, "front_at_cut": f_cut, "front_at_rest": f_rest,
            "stop_dist_m": (f_cut - f_rest) if f_rest is not None else None,
            "cut_to_last_cruise_scan_s": t_cut_wall - (t0 + ts[-1])}


def main():
    rig = Rig()
    gui = TestGui(rig)
    gui.set("Speed + stopping-distance calibration", "each level: cruise (speed measured), cut throttle (roll-out measured)", 0.0)
    out = {"levels": []}
    try:
        for i, pwm in enumerate(LEVELS):
            r = run_level(rig, pwm, gui); out["levels"].append(r)
            if "front_at_rest" in r:
                r["v_ref"] = float(np.interp(pwm, *V_REF))
                gui.log(f"PWM {pwm}: measured {r['v_cruise']:.2f} m/s -> stopped in {r['stop_dist_m']*100:.0f} cm")
            else:
                gui.log(f"PWM {pwm}: {r.get('skipped')}")
            gui.set(progress=(i + 1) / len(LEVELS))
            time.sleep(0.5)
    finally:
        rig.stop(); rig.steer(D.CENTER0); time.sleep(0.2)
        json.dump(out, open(REPORT, "w"), indent=2)
        try:
            good = [r for r in out["levels"] if r.get("v_cruise") and r.get("v_cruise") > 0.05 and r.get("stop_dist_m") is not None]
            if len(good) >= 3:
                p = np.array([r["pwm"] for r in good], float); v = np.array([r["v_cruise"] for r in good])
                m, c = np.polyfit(p, v, 1)                       # v = m*pwm + c  ->  deadband = -c/m
                dead = float(-c / m)
                vmax = float(m * (255 - dead))
                d = np.array([r["stop_dist_m"] for r in good])
                A = np.stack([v, v ** 2], 1)                     # stop = v*T + v^2/(2a)
                (T, k), *_ = np.linalg.lstsq(A, d, rcond=None)
                res = {"deadband": dead, "v_max": vmax,
                       "reaction_s": float(max(T, 0.0)), "decel": float(1 / (2 * k)) if k > 1e-3 else None,
                       "levels": [{"pwm": r["pwm"], "v": r["v_cruise"], "stop_cm": r["stop_dist_m"] * 100} for r in good]}
                save_result("speed", res)
                gui.log(f"FIT: deadband {dead:.0f} PWM, v_max {vmax:.2f} m/s, reaction {res['reaction_s']:.2f} s")
        except Exception as e:
            print("fit error", e, flush=True)
        gui.set("Speed + stopping-distance calibration - DONE", "see log", 1.0, activity="finished")
        hold(45); rig.close()


if __name__ == "__main__":
    main()
