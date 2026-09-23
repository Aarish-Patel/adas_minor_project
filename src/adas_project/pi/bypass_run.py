"""Run the obstacle bypass on the real car (relay must be stopped - this owns the LiDAR/ESP32).

Backs up (straight) to get run-up room, then drives straight; when an obstacle is in the path it
goes around the side with the most free room and rejoins the original line. Everything is shown
on the browser GUI (port 8090). Independent hardware stops, on top of the controller's own:
front cone < 0.28 m, stale LiDAR, run-time cap, distance cap."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/pi/rc_car")
import pi.lidar_steering_diag as D  # noqa: E402
from pi.lidar_steering_diag import Rig, TICK_S  # noqa: E402
from pi.test_gui import TestGui  # noqa: E402
import pi.speed_run as SR  # noqa: E402
from pi.bypass_core import Bypass, SERVO_STRAIGHT, K_CURV_PER_DEG  # noqa: E402
from pi.scanmatch import Odometry, polar_to_xy  # noqa: E402

D.RAMP_STEP_PWM = 25
PWM = 115
V_PRED = 0.27              # m/s at PWM 115 (speed_run.py table)
MAX_RUN_S = 30.0
MAX_X_M = 4.5
HW_FRONT_STOP_M = 0.28
WANT_OBSTACLE_M = 1.05     # back up until the obstacle is at least this far (or the rear limit)
REPORT = "/home/pi/rc_car/pi/bypass_report.json"

ACTIVITY = {"CRUISE": "DRIVING STRAIGHT", "AVOID": "OBSTACLE - ARCING AROUND IT", "PASS": "PASSING THE OBSTACLE",
            "RETURN": "RETURNING TO THE ORIGINAL LINE", "DONE": "DONE", "ABORT": "STOPPED"}


def main():
    rig = Rig()
    gui = TestGui(rig)
    gui.set("Obstacle bypass", "go around the obstacle on the more open side, then rejoin the straight line", 0.0)
    trace, ctl, odo = [], Bypass(pwm=PWM), Odometry()
    end_reason = "?"
    try:
        # ---- run-up room: straight back until the obstacle is far enough or the rear limit ----
        gui.set(activity="FINDING OPEN SPACE - backing up for run-up room", servo=SERVO_STRAIGHT)
        rig.steer(SERVO_STRAIGHT)
        t0 = time.time()
        while time.time() - t0 < 5.0:
            f, r = rig.front(), rig.rear()
            if not rig.is_fresh() or (f is not None and f >= WANT_OBSTACLE_M + 0.6) or (r is not None and r < SR.REAR_STOP_M):
                break
            rig.motor_reverse(105)
            time.sleep(TICK_S)
        rig.stop()
        time.sleep(0.8)
        gui.log(f"start: front {rig.front()}, rear {rig.rear()}")

        # ---- main loop ------------------------------------------------------------------------
        start, last_t, last_state, servo_cmd = time.time(), None, None, SERVO_STRAIGHT
        while True:
            if time.time() - start > MAX_RUN_S:
                end_reason = "time cap"; break
            if not rig.is_fresh():
                end_reason = "no fresh LiDAR scan"; break
            f = rig.front()
            if f is not None and f < HW_FRONT_STOP_M:
                end_reason = f"hardware stop: front {f:.2f} m"; break
            rig.motor_forward(PWM)
            last_t2, s = SR.unique_scan_sample(rig, last_t)
            if s:
                last_t = last_t2
                xy = polar_to_xy(rig.points())
                est = odo.update(xy, s[0], V_PRED, K_CURV_PER_DEG * (servo_cmd - SERVO_STRAIGHT))
                r = ctl.step(est, xy)
                servo_cmd = r["servo_deg"]
                rig.steer(servo_cmd)
                trace.append({"t": round(time.time() - start, 2), "x": float(est[0]), "y": float(est[1]),
                              "th_deg": float(np.degrees(est[2])), "servo": servo_cmd, "state": r["state"]})
                gui.set(activity=ACTIVITY.get(r["state"], r["state"]), servo=float(servo_cmd),
                        message=f"{r['msg']}   |   lateral {est[1]:+.2f} m, heading {np.degrees(est[2]):+.1f} deg "
                                f"(+ side = right of the screen)")
                if r["state"] != last_state:
                    gui.log(f"{r['state']}: {r['msg']}")
                    last_state = r["state"]
                if r["pwm"] == 0:
                    end_reason = r["state"]; break
                if est[0] > MAX_X_M:
                    end_reason = "distance cap"; break
            time.sleep(TICK_S)
    finally:
        rig.stop(); rig.steer(SERVO_STRAIGHT)
        fin = trace[-1] if trace else {}
        json.dump({"end": end_reason, "state": ctl.state, "msg": ctl.msg, "side": ctl.side, "obstacle": ctl.obs,
                   "gap_pos": ctl.gap_pos, "gap_neg": ctl.gap_neg, "fallbacks": odo.fallbacks,
                   "ref_fixes": odo.ref_fixes, "scans": odo.n, "trace": trace}, open(REPORT, "w"), indent=1)
        try:
            gui.log(f"END: {end_reason}")
            if fin:
                gui.log(f"final lateral {fin['y']:+.3f} m, heading {fin['th_deg']:+.1f} deg, odom fallbacks {odo.fallbacks}/{odo.n}")
            gui.set("Obstacle bypass - " + ("DONE" if ctl.state == "DONE" else "STOPPED"), ctl.msg, 1.0,
                    activity="finished")
            time.sleep(60)
        except Exception as e:
            print("end error", e)
        rig.close()


if __name__ == "__main__":
    main()
