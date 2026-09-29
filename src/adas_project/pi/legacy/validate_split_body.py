"""Confirm the front/rear split correctly isolates the known close wall (~-30deg) to
front-only, leaving rear clear - i.e. reverse should now be unblocked."""
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
WIDE_CONE = 90


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

front_alerts, rear_alerts, total = 0, 0, 0
front_mins, rear_mins = [], []

for i in range(15):
    scan = next(gen)
    bf = br = None
    for _, angle, dist in scan:
        d_m = dist / 1000.0
        if d_m <= 0 or d_m < MIN_VALID:
            continue
        a = to_car_angle(angle)
        body_d = d_m - new_overhang(a)
        if abs(a) <= WIDE_CONE:
            if bf is None or body_d < bf:
                bf = body_d
        else:
            if br is None or body_d < br:
                br = body_d
    total += 1
    if bf is not None:
        front_mins.append(bf)
        if bf < HARD_FLOOR:
            front_alerts += 1
    if br is not None:
        rear_mins.append(br)
        if br < HARD_FLOOR:
            rear_alerts += 1

lidar.stop()
lidar.stop_motor()
lidar.disconnect()

print(f"scans: {total}")
print(f"FRONT-half: alert {front_alerts}/{total} = {front_alerts/total*100:.0f}%   min={min(front_mins):.3f}  avg={sum(front_mins)/len(front_mins):.3f}")
print(f"REAR-half:  alert {rear_alerts}/{total} = {rear_alerts/total*100:.0f}%   min={min(rear_mins):.3f}  avg={sum(rear_mins)/len(rear_mins):.3f}")
print("\n-> if REAR alert rate is low/zero, reverse is now correctly unblocked by this wall.")
