"""Vehicle health supervision with degraded modes (TODO N8), the way production EVs handle faults (ISO 26262 ideas:
monitor every input the safety function depends on, and fall back to a safe reduced mode instead of carrying on as if
nothing happened).

Watched every tick of the relay:
  LiDAR          scan rate (the brake and the planner are only as good as the scans)
  control loop   time from a driver packet to the motor command
  driver link    packet rate from the controller (WiFi drops)
  ESP32 link     pi/esp_link.py state (answering / silent / lost)
  temperature    the Pi's SoC temperature (it throttles at 80-85 C, and a throttled Pi brakes late)
States:
  normal    nothing wrong
  limp      reduced power: throttle capped at LIMP_PWM (slower = shorter stopping distance) until the cause is gone
            for CLEAR_S - LiDAR slow, loop slow, driver link lossy, hot
  fault     the car cannot be driven safely: motor held at zero - no LiDAR scans, ESP32 silent or unplugged
Each cause is latched until it has been absent for CLEAR_S, so the state does not flicker.
"""
import collections
import time

LIMP_PWM = 120.0
CLEAR_S = 2.0
WINDOW_S = 2.0
LIDAR_MIN_HZ = 6.0            # the RPLIDAR runs at ~10 Hz
LIDAR_LOST_S = 1.0
LOOP_P95_MAX_MS = 60.0
LINK_MIN_HZ = 12.0            # the controller sends 20 Hz; below 12 Hz is >40 % loss
HOT_C = 80.0


class HealthMonitor:
    def __init__(self, temp_path="/sys/class/thermal/thermal_zone0/temp"):
        self.temp_path = temp_path
        self.scans = collections.deque()
        self.loops = collections.deque()
        self.packets = collections.deque()
        self.last_seq, self.last_scan_t = None, None
        self.temp_c, self.temp_t = None, 0.0
        self.active = {}              # cause -> (level, detail, last seen)
        self.state, self.causes = "normal", []
        self.driving = False
        self.t0 = None

    # ---------------------------------------------------------------- inputs
    def tick(self, now, scan_seq=None, loop_ms=None, driver_packet=False, esp_state=None, driving=False):
        if self.t0 is None:
            self.t0 = now
        if scan_seq is not None and scan_seq != self.last_seq:
            self.last_seq, self.last_scan_t = scan_seq, now
            self.scans.append(now)
        if loop_ms is not None:
            self.loops.append((now, loop_ms))
        if driver_packet:
            self.packets.append(now)
        self.driving = driving
        for dq in (self.scans, self.packets):
            while dq and dq[0] < now - WINDOW_S:
                dq.popleft()
        while self.loops and self.loops[0][0] < now - WINDOW_S:
            self.loops.popleft()
        if now - self.temp_t > 2.0:
            self.temp_t = now
            self.temp_c = self._read_temp()
        self._evaluate(now, esp_state)

    def _read_temp(self):
        try:
            with open(self.temp_path) as f:
                return int(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            return None

    # ---------------------------------------------------------------- rules
    def _evaluate(self, now, esp_state):
        seen = {}
        warm = now - self.t0 > WINDOW_S                  # rates need a full window first
        if self.last_scan_t is None or now - self.last_scan_t > LIDAR_LOST_S:
            if now - self.t0 > LIDAR_LOST_S:
                seen["lidar"] = ("fault", "no LiDAR scans")
        elif warm and len(self.scans) / WINDOW_S < LIDAR_MIN_HZ:
            seen["lidar"] = ("limp", f"LiDAR slow ({len(self.scans) / WINDOW_S:.1f} Hz)")
        if esp_state in ("silent", "lost"):
            seen["esp32"] = ("fault", f"ESP32 {esp_state}")
        if len(self.loops) >= 10:
            ms = sorted(v for _, v in self.loops)
            p95 = ms[int(0.95 * (len(ms) - 1))]
            if p95 > LOOP_P95_MAX_MS:
                seen["loop"] = ("limp", f"control loop slow (95th {p95:.0f} ms)")
        if warm and self.driving and len(self.packets) / WINDOW_S < LINK_MIN_HZ:
            seen["link"] = ("limp", f"driver link lossy ({len(self.packets) / WINDOW_S:.0f} packets/s)")
        if self.temp_c is not None and self.temp_c >= HOT_C:
            seen["temp"] = ("limp", f"Pi hot ({self.temp_c:.0f} C)")
        for k, (lvl, det) in seen.items():
            self.active[k] = (lvl, det, now)
        for k in [k for k, (_, _, t) in self.active.items() if k not in seen and now - t > CLEAR_S]:
            del self.active[k]
        levels = [lvl for lvl, _, _ in self.active.values()]
        self.state = "fault" if "fault" in levels else "limp" if "limp" in levels else "normal"
        self.causes = [det for _, det, _ in self.active.values()]

    # ---------------------------------------------------------------- output
    def cap(self, physical):
        """The throttle allowed in the current state (+ forward)."""
        if self.state == "fault":
            return 0.0
        if self.state == "limp":
            return max(-LIMP_PWM, min(LIMP_PWM, physical))
        return physical

    def status(self):
        return {"state": self.state, "causes": list(self.causes),
                "lidar_hz": round(len(self.scans) / WINDOW_S, 1),
                "link_hz": round(len(self.packets) / WINDOW_S, 1),
                "temp_c": None if self.temp_c is None else round(self.temp_c, 1)}
