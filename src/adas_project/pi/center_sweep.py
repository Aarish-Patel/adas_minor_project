"""Servo center + steering gain, measured with CONTINUOUS motion (no stop-start).

The car drives forward (then reverse) at a steady PWM while the servo command steps through
a set of angles every WINDOW_S; it only stops to turn around. For each window we measure the
car's rotation between two scans (full-360 range-profile matching, no wall needed), giving a
yaw RATE (deg/s) for that servo angle (sign-corrected for reverse). Rate vs servo angle:
  - zero crossing            -> ideal straight-ahead center
  - slope / large offsets    -> steering gain (the 'steering angle -> turn rate' table)
Everything is shown live on the browser GUI (port 8090)."""
import json
import random
import sys
import time

import numpy as np

sys.path.insert(0, "/home/pi/rc_car")
import pi.lidar_steering_diag as D  # noqa: E402
import pi.speed_run as SR  # noqa: E402
import pi.center_scan as CS  # noqa: E402
from pi.lidar_steering_diag import Rig, TICK_S  # noqa: E402
from pi.test_gui import TestGui  # noqa: E402

D.RAMP_STEP_PWM = 25
SERVOS = [55, 70, 82, 86, 90, 94, 98, 110, 125]
PWM_FWD = 125
PWM_REV = 115
WINDOW_S = 0.8
SETTLE_S = 0.25
PASSES = 8
FRONT_STOP_M = 0.55
REAR_STOP_M = 0.25
REPORT = "/home/pi/rc_car/pi/center_sweep_report.json"


def grab(rig, timeout=0.6):
    last_t, end = None, time.time() + timeout
    while time.time() < end:
        last_t, s = SR.unique_scan_sample(rig, last_t)
        if s:
            pts = rig.points()
            if len(pts) > 150:
                return pts
        time.sleep(0.02)
    return None


def drive_window(rig, servo, direction):
    """Keep driving for one window at `servo`; return yaw rate in deg/s (sign-corrected)."""
    rig.steer(servo)
    t0 = time.time()
    p0 = p1 = None
    t_a = t_b = None
    while time.time() - t0 < WINDOW_S:
        rig.motor_forward(PWM_FWD) if direction > 0 else rig.motor_reverse(PWM_REV)
        el = time.time() - t0
        if p0 is None and el >= SETTLE_S:
            p0 = grab(rig, 0.1); t_a = time.time() if p0 else None
        time.sleep(TICK_S)
        f, r = rig.front(), rig.rear()
        if (direction > 0 and f is not None and f < FRONT_STOP_M) or \
           (direction < 0 and r is not None and r < REAR_STOP_M) or not rig.is_fresh():
            return None, "edge"
    p1 = grab(rig, 0.15); t_b = time.time()
    if p0 is None or p1 is None or t_a is None:
        return None, "no scans"
    res = CS.rotation_between(p0, p1, max_shift=25)
    if not res or res[1] > 0.25:
        return None, "poor match"
    return direction * res[0] / (t_b - t_a), "ok"


def main():
    rig = Rig()
    gui = TestGui(rig)
    gui.set("Steering center + gain (continuous motion)",
            "yaw rate vs servo angle while driving; zero crossing = ideal center", 0.0)
    rates = {s: [] for s in SERVOS}
    try:
        for p in range(PASSES):
            for direction, label in ((+1, "forward"), (-1, "reverse")):
                order = SERVOS[:]
                random.shuffle(order)
                gui.set(activity=f"FINDING OPEN SPACE - turning around, straightening wheels",
                        servo=90.0)
                rig.steer(90); rig.stop(); time.sleep(0.5)
                for servo in order:
                    gui.set(activity=f"TESTING WHEEL CENTER ({label}, pass {p + 1}/{PASSES})",
                            message=f"servo {servo} deg", servo=float(servo),
                            progress=(p * 2 + (0 if direction > 0 else 1)) / (PASSES * 2))
                    rate, why = drive_window(rig, servo, direction)
                    if rate is None:
                        gui.log(f"servo {servo}: {why} (end of run, turning around)")
                        break
                    rates[servo].append(rate)
                    gui.log(f"servo {servo} (offset {servo - 90:+d}): yaw {rate:+.1f} deg/s")
                rig.stop(); time.sleep(0.3)
    finally:
        rig.stop(); rig.steer(90)
        summ = {s: {"n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v))}
                for s, v in rates.items() if v}
        fit = None
        near = [(s, v) for s, v in rates.items() if 78 <= s <= 102 for v in v]
        if len(near) >= 8:
            x = np.array([a for a, _ in near], float); y = np.array([b for _, b in near], float)
            m, b = np.polyfit(x, y, 1)
            fit = {"gain_deg_s_per_servo_deg": float(m), "ideal_center": float(-b / m) if abs(m) > 1e-6 else None}
        json.dump({"rates": rates, "summary": summ, "fit": fit}, open(REPORT, "w"), indent=2)
        try:
            for s in sorted(summ):
                v = summ[s]
                gui.log(f"servo {s}: {v['mean']:+.1f} deg/s (sd {v['sd']:.1f}, n={v['n']})")
            if fit:
                gui.log(f"IDEAL CENTER ~ {fit['ideal_center']:.1f} deg, gain {fit['gain_deg_s_per_servo_deg']:.2f} deg/s per servo deg")
            gui.set("Steering center + gain - DONE", "results in the log", 1.0, activity="finished")
            print("SUMMARY", json.dumps(summ), "FIT", json.dumps(fit), flush=True)
            time.sleep(90)
        except Exception as e:
            print("summary error", e)
        rig.close()


if __name__ == "__main__":
    main()
