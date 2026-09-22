"""One-off: reposition the car to a spot with good clearance in all directions, using the
fixed escape logic, and stop. Does not run any test phases."""
import sys
import time

sys.path.insert(0, "/home/pi/rc_car")
from pi.lidar_steering_diag import Rig, ensure_clearance, _clearance_ok  # noqa: E402

log = []
rig = Rig()
time.sleep(1.0)
print("start: front=%s rear=%s floor=%s" % (rig.front(), rig.rear(), rig.floor()))

ok = ensure_clearance(rig, log, purpose_bearing=0)

print(f"\nresult: {'OK' if ok else 'FAILED'}")
print("final: front=%s rear=%s floor=%s" % (rig.front(), rig.rear(), rig.floor()))
print(f"\n{len(log)} log events:")
for e in log:
    print(" ", e)

rig.stop()
rig.steer(90)
time.sleep(0.2)
rig.close()
