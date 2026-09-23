"""Steering angle -> yaw rate -> turn radius with the final grips.

For each servo offset (from the 90deg center): from the same start spot, drive a short forward
arc at a fixed PWM, measure the car's total rotation by registering the scan before/after
(ICP: rotation+translation solved together), then reverse straight back to the start spot.
radius = arc_length / rotation, arc_length from the measured speed table."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/pi/rc_car")
import pi.lidar_steering_diag as D  # noqa: E402
import pi.center_scan as CS  # noqa: E402
from pi.lidar_steering_diag import Rig, TICK_S  # noqa: E402
from pi.test_gui import TestGui, save_result  # noqa: E402

D.RAMP_STEP_PWM = 25
OFFSETS = [-35, -20, 20, 35]
REPEATS = 3
PWM = 130
RUN_S = 1.1
V_REF = 0.353          # m/s at PWM 130, from speed_run.py
RAMP_LOSS_S = 0.14     # time lost accelerating (ramp) - arc length = V_REF * (RUN_S - this)
REPORT = "/home/pi/rc_car/pi/turn_test_report.json"


def arc(rig, offset, gui):
    servo = D.CENTER0 + offset
    rig.steer(servo + (8 if offset > 0 else -8))     # approach from the outside (backlash)
    time.sleep(0.35)
    rig.steer(servo)
    time.sleep(0.5)
    gui.set(activity=f"TESTING STEERING - arc at wheel offset {offset:+d} deg", servo=float(servo))
    s0 = CS.stable_scans(rig)
    f0, rear0 = rig.front(), rig.rear()
    if len(s0) < 3 or f0 is None or f0 < 0.65:
        return {"offset": offset, "skipped": f"scans={len(s0)} front={f0}"}, rear0
    t0 = time.time()
    while time.time() - t0 < RUN_S:
        f = rig.front()
        if not rig.is_fresh() or (f is not None and f < 0.45):
            break
        rig.motor_forward(PWM)
        time.sleep(TICK_S)
    dur = time.time() - t0
    rig.stop()
    time.sleep(0.7)
    s1 = CS.stable_scans(rig)
    rot, n = CS.heading_change(s0, s1)
    if rot is None:
        return {"offset": offset, "skipped": f"poor scan match ({n} pairs)"}, rear0
    arc_len = V_REF * max(dur - RAMP_LOSS_S, 0.1)
    return {"offset": offset, "rotation_deg": rot, "duration_s": dur, "arc_m": arc_len,
            "yaw_rate_deg_s": rot / max(dur - RAMP_LOSS_S, 0.1),
            "radius_m": arc_len / abs(np.radians(rot)) if abs(rot) > 1.0 else None}, rear0


def main():
    rig = Rig()
    gui = TestGui(rig)
    gui.set("Steering angle -> turn rate/radius (final grips)", "arcs at fixed wheel offsets", 0.0)
    fc, rc = CS.go_to_open_center(rig, gui)
    gui.log(f"start spot: front {fc}, rear {rc}")
    out, total, k = [], len(OFFSETS) * REPEATS, 0
    try:
        for rep in range(REPEATS):
            for off in OFFSETS:
                gui.set(message=f"arc {k + 1}/{total}: offset {off:+d} deg", progress=k / total)
                r, rear0 = arc(rig, off, gui)
                out.append(r); k += 1
                if "rotation_deg" in r:
                    rad = f"{r['radius_m']:.2f} m" if r["radius_m"] else "n/a"
                    gui.log(f"offset {off:+d}: rotated {r['rotation_deg']:+.1f} deg, radius ~{rad}")
                else:
                    gui.log(f"offset {off:+d}: SKIPPED ({r['skipped']})")
                gui.set(activity="FINDING OPEN SPACE - returning to the start spot", servo=90.0)
                CS.return_to_start(rig, rear0 if rear0 is not None else 0.9)
    finally:
        rig.stop(); rig.steer(90)
        summ = {}
        for off in OFFSETS:
            rs = [r for r in out if r.get("offset") == off and "rotation_deg" in r]
            if rs:
                rots = [r["rotation_deg"] for r in rs]
                summ[off] = {"n": len(rs), "rot_median_deg": float(np.median(rots)),
                             "rot_sd": float(np.std(rots)),
                             "radius_m": float(np.median([r["radius_m"] for r in rs if r["radius_m"]] or [np.nan]))}
        json.dump({"runs": out, "summary": summ}, open(REPORT, "w"), indent=2)
        ks = [1.0 / (r["radius_m"] * abs(r["offset"])) for r in out if r.get("radius_m")]
        if len(ks) >= 3:
            k = float(np.median(ks))
            save_result("turn", {"k_curv_per_deg": k, "n": len(ks), "spread": float(np.std(ks)),
                                 "min_radius_m": float(min(r["radius_m"] for r in out if r.get("radius_m")))})
            gui.log(f"turn gain: {k:.4f} rad/m of curvature per servo degree")
        try:
            for off, v in summ.items():
                gui.log(f"offset {off:+d}: median rot {v['rot_median_deg']:+.1f} deg (sd {v['rot_sd']:.1f}), radius {v['radius_m']:.2f} m")
            gui.set("Steering test - DONE", "see log", 1.0, activity="finished")
            print("SUMMARY", json.dumps(summ), flush=True)
            time.sleep(45)
        except Exception as e:
            print("summary error", e)
        rig.close()


if __name__ == "__main__":
    main()
