"""Scripted logging drive: exercises throttle and steering so the simulator can be fitted to the real car
(sim/log_fit.py). Not a calibration - nothing is written to the tuning file; the value is the log itself
(pi/drive_log.py records every scan and every command automatically).

Each leg is driven forward and then the same leg in reverse with the same steering, so the car ends up
roughly where it started. Throttle ramps up and down smoothly; after each leg the motor is cut and the car
coasts to a stop (coast-down is part of what gets fitted). Before every leg the LiDAR must show enough free
room in that direction - the leg is shortened to fit, or skipped. During a leg it stops at 0.35 m ahead
(0.30 m behind when reversing).
"""
import sys
import time

sys.path.insert(0, "/home/pi/rc_car")
import pi.lidar_steering_diag as D  # noqa: E402
from pi.lidar_steering_diag import Rig, TICK_S  # noqa: E402
from pi.test_gui import TestGui, hold  # noqa: E402

D.RAMP_STEP_PWM = 10               # ~200 PWM/s: smooth, and the ramp itself is useful data
FRONT_STOP_M, REAR_STOP_M = 0.35, 0.30
V_GUESS = {100: 0.22, 110: 0.27, 120: 0.31, 130: 0.35, 140: 0.40, 150: 0.45, 170: 0.53}   # only to size legs
COAST_S = 1.5

# (pwm, servo offset from straight in degrees (+ = right), seconds)
LEGS = [(110, 0, 2.0), (130, 0, 2.0), (150, 0, 1.6), (170, 0, 1.3),
        (120, 8, 2.2), (120, -8, 2.2), (120, 16, 2.2), (120, -16, 2.2), (120, 24, 2.0), (120, -24, 2.0),
        (100, 0, 2.5), (140, 12, 1.8), (140, -12, 1.8)]


def room(rig, forward):
    d = rig.front() if forward else rig.rear()
    return None if d is None else d - (FRONT_STOP_M if forward else REAR_STOP_M)


def drive(rig, gui, pwm, servo, seconds, forward):
    """One leg; returns (seconds actually driven, why it ended)."""
    rig.steer(servo)
    time.sleep(0.4)                                   # let the servo arrive before moving
    t0 = time.time()
    why = "done"
    while time.time() - t0 < seconds:
        if not rig.is_fresh():
            why = "no fresh scan"
            break
        r = room(rig, forward)
        if r is not None and r < 0:
            why = "obstacle - stopped"
            rig.stop()                                # an emergency always wins over smoothness
            return time.time() - t0, why
        rig.motor_forward(pwm) if forward else rig.motor_reverse(pwm)
        time.sleep(TICK_S)
    driven = time.time() - t0
    for _ in range(60):                               # ramp the throttle down (no jerks); below the dead-band
        if abs(rig._pwm_now) < 1:                     # the car is coasting, and that is logged too
            break
        r = room(rig, forward)
        if r is not None and r < 0:
            break                                     # obstacle while slowing: stop now
        rig._step_toward(0.0)
        time.sleep(TICK_S)
    rig.stop()
    time.sleep(COAST_S)
    return driven, why


def main():
    rig = Rig()
    gui = TestGui(rig)
    gui.set("Logging drive (data for the simulator)", "forward + reverse legs at several speeds and steering angles", 0.0,
            activity="STARTING")
    rig.log.event("log_drive start", legs=LEGS, centre=D.CENTER0)
    gui.set(activity="WAITING FOR THE LIDAR to spin up")
    t_wait = time.time()
    while time.time() - t_wait < 20 and not (rig.is_fresh() and rig.front() is not None and rig.rear() is not None):
        time.sleep(0.2)
    if not rig.is_fresh():
        gui.log("no LiDAR scans after 20 s - nothing driven")
    done = 0
    try:
        for i, (pwm, off, secs) in enumerate(LEGS):
            servo = D.CENTER0 + off
            need = V_GUESS.get(pwm, 0.4) * secs + 0.15
            gui.set(progress=i / len(LEGS), servo=float(servo),
                    message=f"leg {i + 1}/{len(LEGS)}: PWM {pwm}, wheels {off:+d} deg, {secs:.1f} s")
            for _ in range(10):                       # a single missing reading is not "no room"
                r = room(rig, True)
                if r is not None:
                    break
                time.sleep(0.2)
            if r is None or r < 0.3:
                gui.log(f"leg {i + 1}: skipped - only {r if r is None else round(r, 2)} m free ahead")
                rig.log.event("leg skipped", leg=i, room=r)
                continue
            fwd_s = secs if r >= need else max(0.8, secs * r / need)
            gui.set(activity=f"DRIVING FORWARD - PWM {pwm}, wheels {off:+d} deg")
            rig.log.event("leg forward", leg=i, pwm=pwm, servo=servo, planned_s=fwd_s)
            t_f, why_f = drive(rig, gui, pwm, servo, fwd_s, True)
            gui.set(activity=f"REVERSING THE SAME LEG - PWM {pwm}")
            rig.log.event("leg reverse", leg=i, pwm=pwm, servo=servo, planned_s=t_f)
            t_r, why_r = drive(rig, gui, pwm, servo, t_f, False)
            gui.log(f"leg {i + 1}: PWM {pwm}, {off:+d} deg: forward {t_f:.1f} s ({why_f}), back {t_r:.1f} s ({why_r})")
            done += 1
        gui.log(f"finished: {done}/{len(LEGS)} legs driven; log: {rig.log.path}")
        gui.set("Logging drive - DONE", f"{done} legs logged", 1.0, activity="finished", servo=float(D.CENTER0))
    finally:
        rig.stop()
        rig.steer(D.CENTER0)
        rig.log.event("log_drive end", legs_done=done)
        hold(30)
        rig.close()


if __name__ == "__main__":
    main()
