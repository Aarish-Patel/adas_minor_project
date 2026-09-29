"""Stationary full-360 scan printout - no motion. Sanity-check the space before driving."""
import sys
import time

sys.path.insert(0, "/home/pi/rc_car")
from pi.lidar_steering_diag import Rig  # noqa: E402

rig = Rig()
time.sleep(1.0)
for _ in range(5):
    front = rig.front()
    rear = rig.rear()
    floor_d, floor_a = rig.floor()
    print(f"front={front} rear={rear} floor={floor_d} @ {floor_a} deg")
    time.sleep(0.3)
rig.close()
