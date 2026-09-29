import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rplidar import RPLidar

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/lidar_front.png"
FRONT_OFFSET_DEG = 96.5   # raw LiDAR angle that is actually the car's straight-ahead (measured)

lidar = RPLidar(PORT, baudrate=256000, timeout=3)
try:
    print("health:", lidar.get_health())
    all_points = []
    scan_count = 0
    warmup = 5
    for scan in lidar.iter_scans(max_buf_meas=6000, min_len=5):
        scan_count += 1
        if scan_count <= warmup:
            continue
        all_points.extend([(a, d) for _, a, d in scan if d > 0])
        if scan_count - warmup >= 15:
            break
finally:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()

# report exactly what's in the forward +/-30deg cone
def to_car_frame(a):
    a = (a - FRONT_OFFSET_DEG) % 360
    return a if a <= 180 else a - 360

front = [(to_car_frame(a), d / 1000.0) for a, d in all_points]
CONE = 15
front = sorted([(a, d) for a, d in front if abs(a) <= CONE], key=lambda x: x[1])
print(f"\n{len(front)} points within +/-30 deg of straight ahead, closest first:")
for a, d in front[:20]:
    print(f"  angle {a:+6.1f} deg   distance {d:.3f} m")

angles = np.radians([to_car_frame(a) for a, d in all_points])
dists = np.array([d for a, d in all_points]) / 1000.0
fig = plt.figure(figsize=(7.5, 7.5), facecolor="#0b1120")
ax = fig.add_subplot(111, projection="polar")
ax.set_facecolor("#0b1120")
ax.scatter(angles, dists, s=3, c=dists, cmap="viridis", alpha=0.75)
ax.set_theta_zero_location("N")
ax.set_thetamin(-40); ax.set_thetamax(40)
ax.set_rmax(1.5)
ax.set_title(f"Forward sector ({len(all_points)} total pts)", color="white", pad=20)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")
ax.grid(color="#334155", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT, dpi=130, facecolor="#0b1120")
print("saved", OUT)
