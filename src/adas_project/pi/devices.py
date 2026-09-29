"""Real hardware adapters for the Pi: RPLIDAR A3M1, USB webcam (ArUco), Xbox controller.

These need the hardware (and the libraries named below), so unlike the rest of the
project they have NOT been run in the simulator. Test each one on its own first:
    python -m pi.devices lidar /dev/ttyUSB0
    python -m pi.devices camera 0
    python -m pi.devices gamepad
"""

import math
import sys
import threading
import time

import numpy as np

from adas.markers import Camera, detect_image


class RPLidarSource:
    """RPLIDAR A3M1 over USB serial (256000 baud). pip install rplidar-roboticia

    Angles from the RPLIDAR are degrees clockwise; the ADAS wants radians counter-clockwise from the
    car's forward axis, so set `angle_offset_deg` to where the LiDAR's zero points relative to the front.
    If your driver library cannot start the A3 in its default mode, use Slamtec's own SDK instead.
    """

    def __init__(self, port="/dev/ttyUSB0", baud=256000, angle_offset_deg=0.0, min_range=0.15, max_range=8.0):
        from rplidar import RPLidar
        self.lidar = RPLidar(port, baudrate=baud)
        self.offset = math.radians(angle_offset_deg)
        self.min_range, self.max_range = min_range, max_range
        self._latest = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        for scan in self.lidar.iter_scans(max_buf_meas=6000):
            if not self._running:
                break
            arr = np.array([(a, d) for _, a, d in scan], dtype=float)
            if not len(arr):
                continue
            angles = (-np.radians(arr[:, 0]) + self.offset) % (2 * math.pi)
            ranges = arr[:, 1] / 1000.0
            valid = (ranges > self.min_range) & (ranges < self.max_range)
            order = np.argsort(angles)
            with self._lock:
                self._latest = (time.time(), angles[order], ranges[order], valid[order])

    def poll(self):
        with self._lock:
            out, self._latest = self._latest, None
        return out

    def close(self):
        self._running = False
        self.lidar.stop()
        self.lidar.stop_motor()
        self.lidar.disconnect()


class WebcamMarkers:
    """USB webcam -> ArUco detections at ~15 Hz. Run tools/calibrate camera first for real intrinsics."""

    def __init__(self, index=0, camera=None, marker_sizes=None, fps=15.0):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cam = camera or Camera()
        self.sizes = marker_sizes or {7: 0.11, 21: 0.10, 22: 0.10, 23: 0.10}
        self.period = 1.0 / fps
        self._next = 0.0

    def poll(self):
        now = time.time()
        if now < self._next:
            return None
        self._next = now + self.period
        ok, frame = self.cap.read()
        if not ok:
            return None
        return now, [o for o in detect_image(frame, self.cam, self.sizes) if o.id in self.sizes]   # ignore unknown IDs


class XboxDriver:
    """Xbox controller via pygame. Left stick X steers, triggers drive. Same axes as rc_controller.py."""

    def __init__(self, max_pwm=255, deadzone=0.08):
        import pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("no controller found")
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        self.pg = pygame
        self.max_pwm, self.dz = max_pwm, deadzone

    def _dz(self, v):
        return 0.0 if abs(v) < self.dz else (abs(v) - self.dz) / (1 - self.dz) * (1 if v > 0 else -1)

    def read(self):
        self.pg.event.pump()
        stick = self._dz(self.js.get_axis(0))
        steer = -stick                                       # ADAS convention: + = physically left
        rt = (self.js.get_axis(5) + 1) / 2
        lt = (self.js.get_axis(4) + 1) / 2
        pwm = (self._dz(rt) - self._dz(lt)) * self.max_pwm
        b = self.js.get_button
        buttons = {"mode": b(2), "park": b(0), "follow": b(3), "estop": b(1)}    # X, A, Y, B
        return steer, pwm, buttons


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "lidar":
        src = RPLidarSource(sys.argv[2] if len(sys.argv) > 2 else "/dev/ttyUSB0")
        for _ in range(30):
            s = src.poll()
            if s:
                _, a, r, v = s
                print(f"scan: {int(v.sum())} valid points, nearest {r[v].min():.2f} m" if v.any() else "scan: no valid points")
            time.sleep(0.2)
        src.close()
    elif what == "camera":
        cam = WebcamMarkers(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
        for _ in range(60):
            f = cam.poll()
            if f and f[1]:
                print([(o.id, round(o.dist, 2)) for o in f[1]])
            time.sleep(0.05)
    elif what == "gamepad":
        pad = XboxDriver()
        for _ in range(100):
            print(pad.read(), end="\r")
            time.sleep(0.05)
    else:
        print(__doc__)
