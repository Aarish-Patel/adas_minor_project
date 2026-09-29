import sys, time
sys.path.insert(0, "/home/pi/rc_car")
from pi.lidar_steering_diag import Rig
from pi.center_scan import wall_angle
rig = Rig(); time.sleep(1.0)
pts = rig.points()
front = sorted([(a, d) for a, d in pts if abs(a) <= 60])
print("n front(+-60):", len(front))
for a, d in front[::6]: print(f"  a={a:6.1f} d={d:.2f}")
for lim in (20, 30, 38, 50):
    sel = [(a, d) for a, d in pts if abs(a) <= lim and 0.45 <= d <= 3.0]
    print(lim, len(sel))
print("wall_angle:", wall_angle(pts))
rig.close()
