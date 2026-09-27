"""PWM -> speed with SUSTAINED runs, measured only on unique LiDAR scans after acceleration.

The old per-level test fitted ~5 samples over 0.35s from a ~8Hz LiDAR (only 2-3 distinct
readings) taken mid-acceleration, and came back non-monotonic (230 PWM slower than 110).
Here each level: back up to the rear-most safe spot (maximizes forward room), ramp up,
cruise, sample the front distance once per NEW scan, fit only the constant-speed tail,
stop before the front cone gets close. Straight steering, direct hardware, relay stopped.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/pi/rc_car")
import pi.lidar_steering_diag as D  # noqa: E402
from pi.lidar_steering_diag import Rig, floor_violated, TICK_S  # noqa: E402

D.RAMP_STEP_PWM = 25          # ~0.5s to 250: still gentle, but leaves time to cruise
LEVELS = [90, 110, 130, 150, 180, 210, 240]
STOP_FRONT_M = 0.50           # end the run when the front bumper clearance gets this low
MAX_RUN_S = 3.0
REAR_STOP_M = 0.24   # user confirmed ~50cm free behind; bumper clearance to stop at
REPORT = "/home/pi/rc_car/pi/speed_run_report.json"


def unique_scan_sample(rig, last_t):
    with rig._lock:
        t = rig._last_scan_t
        f = rig._front
    if t is None or t == last_t or f is None:
        return last_t, None
    return t, (t, f)


def back_up_to_rear_limit(rig):
    """Reverse (slowly, straight) until the rear cone reads REAR_STOP_M - deliberately does NOT
    use the 360 hard floor, which would refuse to back up at all with a wall this close
    behind; the rear cone is still watched every tick and the run is time-capped."""
    rig.steer(D.CENTER0)             # calibrated straight-ahead, not raw 90
    end = time.time() + 4.5
    while time.time() < end:
        rear = rig.rear()
        if not rig.is_fresh() or rear is None or rear < REAR_STOP_M:
            break
        rig.motor_reverse(105)
        time.sleep(TICK_S)
    rig.stop()
    time.sleep(0.6)


def run_level(rig, pwm):
    back_up_to_rear_limit(rig)
    f0 = rig.front()
    if f0 is None or f0 < STOP_FRONT_M + 0.25:
        return {"pwm": pwm, "skipped": f"not enough front room ({f0})"}
    samples, last_t = [], None
    t0 = rig.now()
    reason = "time_cap"
    while rig.now() - t0 < MAX_RUN_S:
        bad, _, _ = floor_violated(rig)
        f = rig.front()
        if bad:
            reason = "floor"; break
        if f is not None and f < STOP_FRONT_M:
            reason = "front_room"; break
        rig.motor_forward(pwm)
        last_t, s = unique_scan_sample(rig, last_t)
        if s:
            samples.append(s)
        time.sleep(TICK_S)
    rig.stop()
    ramp_s = pwm / D.RAMP_STEP_PWM * TICK_S
    tail = [(t - t0, f) for t, f in samples if t - t0 > ramp_s + 0.25]
    v = None
    if len(tail) >= 4:
        ts = np.array([a for a, _ in tail]); fs = np.array([b for _, b in tail])
        v = float(max(0.0, -np.polyfit(ts, fs, 1)[0]))
    return {"pwm": pwm, "speed_m_s": v, "n_tail": len(tail), "n_all": len(samples),
            "stop": reason, "run_s": rig.now() - t0, "front_start": f0,
            "front_end": samples[-1][1] if samples else None}


def main():
    rig = Rig()
    out = {"levels": []}
    try:
        for pwm in LEVELS:
            r = run_level(rig, pwm)
            print(r, flush=True)
            out["levels"].append(r)
            time.sleep(0.5)
    finally:
        rig.stop(); rig.steer(D.CENTER0); time.sleep(0.2); rig.close()
        json.dump(out, open(REPORT, "w"), indent=2)


if __name__ == "__main__":
    main()
