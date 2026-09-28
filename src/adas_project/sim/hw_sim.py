"""The real car, emulated for the laptop: a virtual ESP32 and a virtual RPLIDAR that the UNCHANGED relay
(pi/wifi_drive_safety.py) talks to instead of the serial ports. Run it with tools/sim_car.py and drive it
with rc_controller.py --ip 127.0.0.1; the relay's own GUI shows on http://localhost:8090.

Everything the car was measured to do is reproduced (sim/fitted_car.json, fitted from the drive logs):
  motor      first-order lag to a steady speed that is linear in PWM above a dead-band, coast-down decel,
             0.12 s command delay, ESP32 500 ms failsafe (no command -> motor off)
  steering   path curvature per servo degree about the straight-ahead servo angle, finite servo slew
  LiDAR      Slamtec "Sensitivity" mode: ~1360 samples per rotation at ~10 Hz, raw clockwise angles offset by
             the mount yaw, nothing closer than 0.20 m, range noise, random dropouts, no return from
             dark/absorbing surfaces, and (optional) the USB stalls seen on the car
The ground truth (walls, the true car pose, collisions) is exposed so the GUI can overlay it.
"""
import json
import math
import os
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


# Braking of the real car (the logging drive only ramps down, so these come from the relay drive of 28 Sep,
# pi/drive_logs/drive_20260928_015209.csv): after the 0.12 s command delay, a throttle cut at 0.7 m/s rolls ~6 cm
# (~4 m/s^2) and an active brake pulse stops it within 1-3 cm (~8 m/s^2), as the user also observed ("brakes almost
# instantly, 1-2 cm drift"). The old 2 m/s^2 coast was an unfitted default.
MEASURED_COAST_DECEL = 4.0
MEASURED_BRAKE_DECEL = 8.0


def load_car_model():
    fit_path = os.path.join(HERE, "fitted_car.json")
    m = {"v_max": 0.834, "deadband": 11.2, "tau_motor": 0.033, "coast_decel": MEASURED_COAST_DECEL, "delay_s": 0.12,
         "k_curv_per_deg": 0.0656, "servo_centre": 86.8, "brake_decel": MEASURED_BRAKE_DECEL}
    if os.path.exists(fit_path):
        f = json.load(open(fit_path, encoding="utf-8"))
        if "truth" not in f:
            m.update({k: f["speed"][k] for k in ("v_max", "deadband", "tau_motor", "delay_s")})
            if f["steer"].get("identifiable"):
                m["k_curv_per_deg"] = f["steer"]["k_curv_per_deg"]
                m["servo_centre"] = f["steer"]["servo_centre"]
    return m


class VirtualCar:
    """Rear-axle pose in the world (x, y, theta; y left, theta counter-clockwise), driven by ESP32 lines."""

    SERVO_SLEW = 400.0          # deg/s
    FAILSAFE_S = 0.5            # ESP32 firmware: motor off after this long without a command
    DT = 0.005

    def __init__(self, world, params, start=(0.0, 0.0, 0.0), motor_reversed=True, threaded=True):
        self.world, self.p = world, params
        self.m = load_car_model()
        self.x, self.y, self.th = start
        self.v = 0.0
        self.servo = self.servo_cmd = self.m["servo_centre"]
        self.pwm = 0.0                      # physical, + = forward, as applied by the motor driver
        self.reversed = motor_reversed
        self.queue = []                     # (t_apply, kind, value)
        self.last_cmd_t = time.time()
        self.crashed = False
        self.crash_count = 0
        self.lock = threading.Lock()
        self.running = True
        self.clock = 0.0                    # simulated time when not threaded (Monte Carlo)
        if threaded:
            threading.Thread(target=self._run, daemon=True).start()

    # ---------------------------------------------------------------- ESP32 command handling
    def command(self, line, now=None):
        p = line.split()
        now = time.time() if now is None else now
        with self.lock:
            try:
                if p[0] == "A" and len(p) == 3:
                    self.queue.append((now + self.m["delay_s"], "A", (float(p[1]) + float(p[2])) / 2))
                elif p[0] == "M" and len(p) == 2:
                    w = float(p[1])
                    self.queue.append((now + self.m["delay_s"], "M", -w if self.reversed else w))
                    self.last_cmd_t = now
                elif p[0] == "STOP":
                    self.queue.append((now, "M", 0.0))
            except (ValueError, IndexError):
                pass

    # ---------------------------------------------------------------- physics
    def _vss(self, u):
        m = self.m
        mag = max(0.0, (abs(u) - m["deadband"]) / (255.0 - m["deadband"]))
        return math.copysign(m["v_max"] * min(1.0, mag), u) if u else 0.0

    def _run(self):
        last = time.time()
        while self.running:
            time.sleep(self.DT)
            now = time.time()
            dt = min(0.05, now - last)
            last = now
            with self.lock:
                due = [q for q in self.queue if q[0] <= now]
                self.queue = [q for q in self.queue if q[0] > now]
                for _, kind, val in due:
                    if kind == "A":
                        self.servo_cmd = max(35.0, min(145.0, val))
                    else:
                        self.pwm = val
                if now - self.last_cmd_t > self.FAILSAFE_S:
                    self.pwm = 0.0
                self._step(dt)

    def _step(self, dt):
        m = self.m
        ds = self.servo_cmd - self.servo
        self.servo += max(-self.SERVO_SLEW * dt, min(self.SERVO_SLEW * dt, ds))
        target = self._vss(self.pwm)
        if self.pwm == 0:
            dec = min(abs(self.v) / max(m["tau_motor"], 1e-3), m["coast_decel"]) * dt
            self.v = 0.0 if abs(self.v) <= dec else self.v - math.copysign(dec, self.v)
        else:
            a = (target - self.v) / max(m["tau_motor"], 0.02)
            if self.v * target < 0:                  # reverse throttle while moving: active braking
                lim = m["brake_decel"]
            elif abs(target) < abs(self.v):          # less throttle: slows like a throttle cut
                lim = m["coast_decel"]
            else:
                lim = 4.0
            if self.v > 0:
                a = max(-lim, min(3.0, a))
            else:
                a = max(-3.0, min(lim, a))
            self.v += a * dt
        kappa = -m["k_curv_per_deg"] * (self.servo - m["servo_centre"])      # + = left
        nth = self.th + kappa * self.v * dt
        nx = self.x + self.v * math.cos((self.th + nth) / 2) * dt
        ny = self.y + self.v * math.sin((self.th + nth) / 2) * dt
        if self.world.clearance(nx, ny, nth, self.p) <= 0.0:
            if not self.crashed:
                self.crash_count += 1
            self.crashed = True
            self.v = 0.0                               # blocked by the obstacle, like the real car
            return
        self.crashed = False
        self.x, self.y, self.th = nx, ny, nth

    def step_to(self, t):
        """Advance the physics to simulated time t (non-threaded use)."""
        while self.clock + self.DT <= t:
            self.clock += self.DT
            due = [q for q in self.queue if q[0] <= self.clock]
            self.queue = [q for q in self.queue if q[0] > self.clock]
            for _, kind, val in due:
                if kind == "A":
                    self.servo_cmd = max(35.0, min(145.0, val))
                else:
                    self.pwm = val
            if self.clock - self.last_cmd_t > self.FAILSAFE_S:
                self.pwm = 0.0
            self._step(self.DT)

    def pose(self):
        with self.lock:
            return self.x, self.y, self.th, self.v, self.servo, self.pwm, self.crashed

    def reset(self, start):
        with self.lock:
            self.x, self.y, self.th = start
            self.v, self.pwm, self.crashed = 0.0, 0.0, False


class SimESP32:
    """Stands in for serial.Serial on the ESP32 port."""

    def __init__(self, car):
        self.car = car
        self._reply = b""
        self.port, self.baudrate, self.timeout, self.dtr, self.rts = "SIM_ESP32", 115200, 0.2, False, False

    def open(self):
        pass

    def write(self, data):
        for line in data.decode(errors="replace").splitlines():
            line = line.strip()
            if line == "PING":
                self._reply = b"PONG\n"
            elif line:
                self.car.command(line)
        return len(data)

    @property
    def in_waiting(self):
        return len(self._reply)

    def read(self, n=1):
        r, self._reply = self._reply[:max(n, len(self._reply))], b""
        return r

    def reset_input_buffer(self):
        self._reply = b""

    def close(self):
        pass


class SimLidar:
    """Stands in for the RPLIDAR (same methods as pi/lidar_dense.DenseLidar / the rplidar library)."""

    def __init__(self, car, yaw_offset_deg, lidar_x=0.12, n=1360, rate_hz=10.0, noise=0.008, dropout=0.04,
                 min_range=0.20, max_range=12.0, usb_stalls=False, seed=1):
        self.car, self.yaw, self.lx = car, yaw_offset_deg, lidar_x
        self.n, self.period = n, 1.0 / rate_hz
        self.noise, self.dropout, self.min_r, self.max_r = noise, dropout, min_range, max_range
        self.usb_stalls = usb_stalls
        self.rng = np.random.default_rng(seed)
        self.running = True
        self.ccw = np.linspace(0.0, 2 * math.pi, n, endpoint=False)

    def get_info(self):
        return {"driver": "simulated A3", "mode": "Sensitivity"}

    def get_health(self):
        return ("Good", 0)

    def _raycast(self, ox, oy, th):
        world = self.car.world
        ang = self.ccw + th
        d = np.column_stack([np.cos(ang), np.sin(ang)])
        best = np.full(self.n, np.inf)
        dark = np.zeros(self.n, dtype=bool)
        seg = world.segments()
        if len(seg):
            ex, ey = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
            wx, wy = seg[:, 0] - ox, seg[:, 1] - oy
            den = d[:, 0, None] * ey[None, :] - d[:, 1, None] * ex[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (wx * ey - wy * ex)[None, :] / den
                u = (wx[None, :] * d[:, 1, None] - wy[None, :] * d[:, 0, None]) / den
            hit = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
            tt = np.where(hit, t, np.inf)
            j = tt.argmin(axis=1)
            best = tt[np.arange(self.n), j]
            dark_seg = getattr(world, "dark_segments", None)
            if dark_seg is not None and len(dark_seg):
                dark = dark_seg[j] & np.isfinite(best)
        circ = world.circles()
        if len(circ):
            fx, fy = ox - circ[:, 0], oy - circ[:, 1]
            b = d[:, 0, None] * fx[None, :] + d[:, 1, None] * fy[None, :]
            cc = (fx * fx + fy * fy - circ[:, 2] ** 2)[None, :]
            disc = b * b - cc
            with np.errstate(invalid="ignore"):
                t = -b - np.sqrt(disc)
            hit = (disc >= 0) & (t > 0)
            ct = np.where(hit, t, np.inf).min(axis=1)
            closer = ct < best
            best = np.where(closer, ct, best)
            dark &= ~closer
        return best, dark

    def iter_scans(self, max_buf_meas=None, min_len=5):
        next_t = time.time()
        stall_at = time.time() + self.rng.uniform(20, 60)
        while self.running:
            next_t += self.period
            time.sleep(max(0.0, next_t - time.time()))
            if self.usb_stalls and time.time() > stall_at:          # the cp210x timeouts seen on the car
                time.sleep(self.rng.uniform(0.3, 1.2))
                stall_at = time.time() + self.rng.uniform(20, 60)
                next_t = time.time()
            x, y, th, *_ = self.car.pose()
            ox, oy = x + self.lx * math.cos(th), y + self.lx * math.sin(th)
            best, dark = self._raycast(ox, oy, th)
            r = best + self.rng.normal(0.0, self.noise + 0.004 * np.where(np.isfinite(best), best, 0), self.n)
            ok = np.isfinite(best) & (r >= self.min_r) & (r <= self.max_r) & ~dark
            ok &= self.rng.random(self.n) > self.dropout
            raw = (-np.degrees(self.ccw) + self.yaw) % 360.0        # the sensor turns clockwise
            q = np.where(ok, 47, 0)
            dist = np.where(ok, r * 1000.0, 0.0)
            order = np.argsort(raw)
            scan = [(int(q[i]), float(raw[i]), float(dist[i])) for i in order]
            yield scan

    def stop(self):
        self.running = False

    def stop_motor(self):
        pass

    def disconnect(self):
        pass


def truth_overlay(car, lidar_x=0.12, max_range=4.0):
    """The world's walls in the GUI's polar convention (car angle clockwise-positive, distance from the LiDAR),
    so the relay GUI can draw ground truth under the scan in simulation."""
    x, y, th, v, servo, pwm, crashed = car.pose()
    ox, oy = x + lidar_x * math.cos(th), y + lidar_x * math.sin(th)
    polys = []
    for x1, y1, x2, y2 in car.world.segments():
        pts = []
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / 0.1) + 1)
        for t in np.linspace(0, 1, n):
            px, py = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
            dx, dy = px - ox, py - oy
            d = math.hypot(dx, dy)
            if d > max_range:
                if len(pts) > 1:
                    polys.append(pts)
                pts = []
                continue
            a = -math.degrees(math.atan2(dy, dx) - th)
            a = (a + 180) % 360 - 180
            pts.append([round(a, 1), round(d, 3)])
        if len(pts) > 1:
            polys.append(pts)
    return {"walls": polys, "crashed": crashed, "crashes": car.crash_count, "v_true": round(v, 3),
            "pose": [round(x, 3), round(y, 3), round(math.degrees(th), 1)]}
