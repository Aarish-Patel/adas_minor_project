import sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rplidar import RPLidar

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB1"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/lidar_map_pi.png"
ROTATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 20

lidar = RPLidar(PORT, baudrate=256000, timeout=3)
try:
    print("info:", lidar.get_info())
    print("health:", lidar.get_health())
    all_points = []
    scan_count = 0
    warmup = 5
    for scan in lidar.iter_scans(max_buf_meas=6000, min_len=5):
        scan_count += 1
        if scan_count <= warmup:
            continue
        pts = [(a, d) for _, a, d in scan if d > 0]
        all_points.extend(pts)
        if scan_count % 5 == 0:
            print(f"rotation {scan_count - warmup}: {len(scan)} samples, {len(all_points)} total")
        if scan_count - warmup >= ROTATIONS:
            break
finally:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()

print(f"total points: {len(all_points)}")
angles = np.radians([a for a, d in all_points])
dists = np.array([d for a, d in all_points]) / 1000.0

fig = plt.figure(figsize=(7.5, 7.5), facecolor="#0b1120")
ax = fig.add_subplot(111, projection="polar")
ax.set_facecolor("#0b1120")
ax.scatter(angles, dists, s=3, c=dists, cmap="viridis", alpha=0.75)
ax.set_theta_zero_location("N")
ax.set_title(f"Live RPLIDAR scan on the Pi  ({len(all_points)} points, {ROTATIONS} rotations)", color="white", pad=20)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")
ax.grid(color="#334155", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT, dpi=130, facecolor="#0b1120")
print("saved", OUT)
