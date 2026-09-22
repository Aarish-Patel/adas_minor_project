"""Minimal, low-stress sanity check after a wheel reattachment: one short, slow, straight
pulse, watching front distance for smooth (not erratic/jumpy) motion, then stop. No hard
turns, no rapid direction reversals - those are exactly what stressed the wheel before."""
import sys
import time

sys.path.insert(0, "/home/pi/rc_car")
from pi.lidar_steering_diag import Rig, floor_violated  # noqa: E402

PWM = 90   # gentle - near the deadband, just enough to confirm it turns cleanly
DURATION_S = 0.6

rig = Rig()
time.sleep(1.0)

print("pre-check clearance: front=%s rear=%s floor=%s" % (rig.front(), rig.rear(), rig.floor()))

bad, d, a = floor_violated(rig)
if bad:
    print(f"ABORT: something already too close (floor={d} @ {a} deg), not moving")
else:
    rig.steer(90)
    time.sleep(0.2)
    readings = []
    end = time.time() + DURATION_S
    while time.time() < end:
        bad, d, a = floor_violated(rig)
        if bad:
            print(f"ABORT mid-pulse: floor={d} @ {a} deg")
            break
        rig.motor_forward(PWM)
        readings.append((rig.now(), rig.front()))
        time.sleep(0.05)
    rig.stop()
    time.sleep(0.3)
    print("readings (t, front_dist):")
    for t, d in readings:
        print(f"  {t:.2f}  {d}")

rig.stop()
rig.steer(90)
time.sleep(0.2)
rig.close()
print("done")
