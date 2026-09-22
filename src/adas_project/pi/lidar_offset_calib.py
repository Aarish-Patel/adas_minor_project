"""Static LiDAR yaw-offset calibration: no motor commands at all. Point a single object
directly in front of the car, run this, and it reports the RAW LiDAR angle (before any
offset correction) where that object shows up - that raw angle becomes the new
mount.yaw_offset_deg. Averages across several scans and looks for a tight, consistent
close-range cluster (a real placed object gives a narrow, stable angular cluster; the
old self-hit/clutter signature looked the same way, so use the closest STABLE cluster,
not just the single closest point which can be noise).
"""
import glob
import time
from collections import defaultdict

import serial
from rplidar import RPLidar


def find_lidar_port(retries=5, delay=1.5):
    for _ in range(retries):
        for p in sorted(glob.glob("/dev/ttyUSB*")):
            try:
                s = serial.Serial(p, 256000, timeout=1)
                s.dtr = False
                s.rts = False
                time.sleep(0.3)
                s.reset_input_buffer()
                s.write(bytes([0xA5, 0x50]))
                time.sleep(0.3)
                resp = s.read(s.in_waiting or 1)
                s.close()
                if resp[:2] == bytes([0xA5, 0x5A]):
                    return p
            except Exception:
                pass
        time.sleep(delay)
    return None


port = find_lidar_port()
if not port:
    print("could not find LiDAR port")
    raise SystemExit(1)

lidar = RPLidar(port, baudrate=256000, timeout=3)
print("health:", lidar.get_health())

N_SCANS = 8
all_points = []
gen = lidar.iter_scans(max_buf_meas=6000, min_len=5)
for i in range(N_SCANS):
    scan = next(gen)
    for _, angle, dist in scan:
        if dist <= 0:
            continue
        all_points.append((angle, dist / 1000.0))

lidar.stop()
lidar.stop_motor()
lidar.disconnect()

print(f"collected {len(all_points)} points over {N_SCANS} scans")

# bucket into 2-degree bins, look at the closest consistent bins (median distance, not just
# the single closest sample, to reject one-off noise)
buckets = defaultdict(list)
for angle, dist in all_points:
    buckets[round(angle / 2) * 2].append(dist)

summarized = []
for bin_angle, dists in buckets.items():
    dists.sort()
    med = dists[len(dists) // 2]
    summarized.append((bin_angle, med, len(dists)))
summarized.sort(key=lambda x: x[1])

print("\nclosest angular bins (raw angle, median distance, sample count):")
for bin_angle, med, n in summarized[:25]:
    print(f"  angle={bin_angle:6.1f}deg  dist={med:.3f}m  n={n}")

print("\nIf the placed object is closer than everything else in the room, its raw angle")
print("(from the tight cluster above, not scattered singletons) is the new yaw_offset_deg.")
