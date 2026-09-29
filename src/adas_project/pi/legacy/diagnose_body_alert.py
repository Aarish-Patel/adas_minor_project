"""Which bearing(s) are triggering the body alert, and are they consistent (self-hit) or
varying (a real nearby object)?"""
import glob
import json
import math
import time
from collections import defaultdict

import serial
from rplidar import RPLidar

TUNING_PATH = "/home/pi/rc_car/pi/tuning_real_car.json"
cfg = json.load(open(TUNING_PATH))
YAW_OFFSET = cfg["mount"]["yaw_offset_deg"]
FRONT_OH, REAR_OH = cfg["mount"]["front_overhang_m"], cfg["mount"]["rear_overhang_m"]
LEFT_OH, RIGHT_OH = cfg["mount"]["left_overhang_m"], cfg["mount"]["right_overhang_m"]
MIN_VALID = cfg["mount"]["min_valid_range_m"]


def to_car_angle(raw):
    a = (raw - YAW_OFFSET) % 360
    return a if a <= 180 else a - 360


def new_overhang(a):
    rad = math.radians(a)
    cx, sy = math.cos(rad), math.sin(rad)
    cands = []
    if cx > 1e-6:
        cands.append(FRONT_OH / cx)
    elif cx < -1e-6:
        cands.append(REAR_OH / -cx)
    if sy > 1e-6:
        cands.append(LEFT_OH / sy)
    elif sy < -1e-6:
        cands.append(RIGHT_OH / -sy)
    return min(cands) if cands else FRONT_OH


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
lidar = RPLidar(port, baudrate=256000, timeout=3)
gen = lidar.iter_scans(max_buf_meas=6000, min_len=5)

worst_per_scan = []
bucket_hits = defaultdict(list)   # 10deg bucket -> list of body_d values across scans

for i in range(15):
    scan = next(gen)
    worst = None
    for _, angle, dist in scan:
        d_m = dist / 1000.0
        if d_m <= 0 or d_m < MIN_VALID:
            continue
        a = to_car_angle(angle)
        body_d = d_m - new_overhang(a)
        bucket = round(a / 10) * 10
        bucket_hits[bucket].append(body_d)
        if worst is None or body_d < worst[0]:
            worst = (body_d, a, d_m)
    worst_per_scan.append(worst)

lidar.stop()
lidar.stop_motor()
lidar.disconnect()

print("worst point per scan (body_clearance, raw_bearing, raw_dist):")
for w in worst_per_scan:
    print(f"  {w}")

print("\nbuckets with body_d consistently under 0.30m (bucket, n_samples, min, median):")
import statistics
for bucket, vals in sorted(bucket_hits.items()):
    under = [v for v in vals if v < 0.30]
    if len(under) >= len(vals) * 0.5 and len(vals) >= 3:
        print(f"  bucket={bucket:5.0f}deg  n={len(vals):3d}  min={min(vals):.3f}  median={statistics.median(vals):.3f}")
