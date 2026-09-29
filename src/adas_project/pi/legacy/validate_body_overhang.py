"""Static, zero-motion comparison: old flat-overhang body_min vs the new ray-box per-bearing
overhang, against the room's actual current geometry. No relay/motor involvement."""
import glob
import json
import math
import time

import serial
from rplidar import RPLidar

TUNING_PATH = "/home/pi/rc_car/pi/tuning_real_car.json"
cfg = json.load(open(TUNING_PATH))
YAW_OFFSET = cfg["mount"]["yaw_offset_deg"]
FRONT_OH, REAR_OH = cfg["mount"]["front_overhang_m"], cfg["mount"]["rear_overhang_m"]
LEFT_OH, RIGHT_OH = cfg["mount"]["left_overhang_m"], cfg["mount"]["right_overhang_m"]
MIN_VALID = cfg["mount"]["min_valid_range_m"]
HARD_FLOOR = 0.30


def to_car_angle(raw):
    a = (raw - YAW_OFFSET) % 360
    return a if a <= 180 else a - 360


def old_overhang(a):
    return min(FRONT_OH, REAR_OH)


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

old_alerts, new_alerts, total = 0, 0, 0
old_mins, new_mins = [], []
worst_old, worst_new = None, None

for i in range(15):
    scan = next(gen)
    old_min = new_min = None
    for _, angle, dist in scan:
        d_m = dist / 1000.0
        if d_m <= 0 or d_m < MIN_VALID:
            continue
        a = to_car_angle(angle)
        od = d_m - old_overhang(a)
        nd = d_m - new_overhang(a)
        if old_min is None or od < old_min:
            old_min = od
        if new_min is None or nd < new_min:
            new_min = nd
    total += 1
    if old_min is not None:
        old_mins.append(old_min)
        if old_min < HARD_FLOOR:
            old_alerts += 1
    if new_min is not None:
        new_mins.append(new_min)
        if new_min < HARD_FLOOR:
            new_alerts += 1

lidar.stop()
lidar.stop_motor()
lidar.disconnect()

print(f"scans analyzed: {total}")
print(f"OLD (flat overhang):  alert rate {old_alerts}/{total} = {old_alerts/total*100:.0f}%   "
      f"min body clearance seen: {min(old_mins):.3f}m   avg: {sum(old_mins)/len(old_mins):.3f}m")
print(f"NEW (ray-box overhang): alert rate {new_alerts}/{total} = {new_alerts/total*100:.0f}%   "
      f"min body clearance seen: {min(new_mins):.3f}m   avg: {sum(new_mins)/len(new_mins):.3f}m")
