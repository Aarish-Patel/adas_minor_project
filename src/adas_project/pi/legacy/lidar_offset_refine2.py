"""Refine the yaw offset: raw angle stats for points within the placed object's known
distance band (confirmed ~20-25cm), for a more precise angle than the coarse 2deg bucketing."""
import glob
import time

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
lidar = RPLidar(port, baudrate=256000, timeout=3)
gen = lidar.iter_scans(max_buf_meas=6000, min_len=5)
hits = []
for i in range(25):
    scan = next(gen)
    for _, angle, dist in scan:
        d = dist / 1000.0
        if 0.30 <= d <= 0.55:
            hits.append((angle, d))
lidar.stop()
lidar.stop_motor()
lidar.disconnect()

print(f"points in 0.17-0.28m band over 12 scans: {len(hits)}")
for a, d in sorted(hits):
    print(f"  angle={a:.2f}  dist={d:.3f}")
if hits:
    avg_a = sum(a for a, d in hits) / len(hits)
    print(f"\nmean angle: {avg_a:.2f} deg")
