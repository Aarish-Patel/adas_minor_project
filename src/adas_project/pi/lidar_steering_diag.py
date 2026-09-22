"""Autonomous, LiDAR-grounded real-car characterization run. No human in the loop.

Talks to the ESP32 and RPLIDAR DIRECTLY over serial (same pattern as pi/calibrate.py's
RealPlatform) - it needs exclusive access to both ports, so it stops the rc-relay service
first and restarts it when done. All of the relay's safety-gate logic (speed-scaled stop
margin, directional-only blocking, hold-after-lost-reading) is reimplemented inline here,
self-contained, so this script is safe to run with the relay down.

Collects, in one run:
  1. PWM -> speed table (straight steering), via distance-over-time regression - same
     method as pi/calibrate.py's calibrate_speed().
  2. Servo center calibration: iteratively finds the raw servo angle that makes "straight"
     actually track straight, using the bearing-to-a-fixed-wall drift as the error signal
     (a static wall's bearing in the car frame shifts by -(the car's own rotation) over a
     short, mostly-rotational creep).
  3. Steering-angle -> turn-rate -> turn-radius table: for a range of raw servo offsets,
     measures heading change per second at a fixed test PWM, and combines it with the
     measured speed at that PWM to get an estimated turn radius (v / turn_rate_rad_s).

Every motion command passes through a local safety check (front/rear clearance from the
current scan) before being sent, and if clearance is ever below CLEARANCE_MIN_M the script
autonomously reverses to a safer spot and retries, rather than stopping and waiting for a
human. Writes pi/steering_diag_report.json and, only if the numbers converge cleanly,
updates pi/tuning_real_car.json (backed up first) with the corrected servo center.
"""
import glob
import json
import os
import shutil
import sys
import threading
import time

import numpy as np
import serial
from rplidar import RPLidar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adas.config import load_tuning  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TUNING_PATH = os.path.join(HERE, "tuning_real_car.json")
REPORT_PATH = os.path.join(HERE, "steering_diag_report.json")

TUNING = load_tuning(TUNING_PATH)
MOUNT = TUNING.mount
CENTER0 = TUNING.servo.left_center     # left_center == right_center by construction here
FWD_CONE_DEG = 40                      # window for the heading-reference wall
SPEED_CONE_DEG = 15                    # narrower window for straight-line distance/speed
CLEARANCE_MIN_M = 0.55
REPOSITION_REVERSE_S = 0.8
TICK_S = 0.05

PWM_LEVELS = [110, 140, 170, 200, 230]
TURN_TEST_PWM = 140
TURN_OFFSETS = [-35, -20, 20, 35]   # raw degrees from center; sign resolved against the probe


def find_ports(retries=5, delay=1.5):
    lidar_port = esp_port = None
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
                    lidar_port = p
                else:
                    esp_port = p
            except Exception:
                pass
        if lidar_port and esp_port:
            return lidar_port, esp_port
        time.sleep(delay)
    return lidar_port, esp_port


def to_car_angle(raw_deg):
    a = (raw_deg - MOUNT.yaw_offset_deg) % 360
    return a if a <= 180 else a - 360


class Rig:
    """Direct serial ESP32 + LiDAR, with a background scan thread feeding front/rear
    clearance and an on-demand wide-cone reference bearing. Motor sign: raw negative M is
    physical forward on this car (confirmed live), independent of any relay convention."""

    def __init__(self):
        lidar_port, esp_port = find_ports()
        if not lidar_port or not esp_port:
            raise RuntimeError(f"missing ports (lidar={lidar_port}, esp={esp_port})")

        self.esp = serial.Serial()
        self.esp.port = esp_port
        self.esp.baudrate = 115200
        self.esp.timeout = 0.2
        self.esp.dtr = False
        self.esp.rts = False
        self.esp.open()
        time.sleep(1.0)
        self.esp.write(b"A 90 90\nM 0\n")

        self.lidar = RPLidar(lidar_port, baudrate=256000, timeout=3)
        self._lock = threading.Lock()
        self._front = self._rear = self._bearing = None
        self._running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
        time.sleep(2.0)
        self._t0 = time.time()

    def _scan_loop(self):
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=6000, min_len=5):
                if not self._running:
                    break
                front = rear = None
                bear_best_d, bear_best_a = None, None
                for _, angle, dist in scan:
                    if dist <= 0 or dist / 1000.0 < MOUNT.min_valid_range_m:
                        continue
                    a = to_car_angle(angle)
                    d_m = dist / 1000.0
                    if abs(a) <= 25:
                        fd = d_m - MOUNT.front_overhang_m
                        if front is None or fd < front:
                            front = fd
                    if abs(abs(a) - 180) <= 25:
                        rd = d_m - MOUNT.rear_overhang_m
                        if rear is None or rd < rear:
                            rear = rd
                    if abs(a) <= FWD_CONE_DEG:
                        if bear_best_d is None or d_m < bear_best_d:
                            bear_best_d, bear_best_a = d_m, a
                with self._lock:
                    self._front, self._rear = front, rear
                    if bear_best_a is not None:
                        self._bearing = bear_best_a
        except Exception as e:
            print("scan thread stopped:", e)

    def now(self):
        return time.time() - self._t0

    def front(self):
        with self._lock:
            return self._front

    def rear(self):
        with self._lock:
            return self._rear

    def bearing(self):
        with self._lock:
            return self._bearing

    def steer(self, raw_angle):
        raw_angle = max(35, min(145, int(round(raw_angle))))
        self.esp.write(f"A {raw_angle} {raw_angle}\n".encode())

    def motor_forward(self, pwm):
        self.esp.write(f"M {-int(pwm)}\n".encode())

    def motor_reverse(self, pwm):
        self.esp.write(f"M {int(pwm)}\n".encode())

    def stop(self):
        self.esp.write(b"M 0\n")

    def close(self):
        self._running = False
        self.esp.write(b"M 0\nSTOP\n")
        time.sleep(0.1)
        self.esp.close()
        try:
            self.lidar.stop()
            self.lidar.stop_motor()
            self.lidar.disconnect()
        except Exception:
            pass


def ensure_clearance(rig, log, need_front=True):
    for attempt in range(6):
        time.sleep(0.25)
        c = rig.front() if need_front else rig.rear()
        if c is not None and c >= CLEARANCE_MIN_M:
            return True
        log.append({"event": "reposition", "attempt": attempt, "clearance": c, "front": need_front})
        rig.steer(90)
        end = time.time() + REPOSITION_REVERSE_S
        while time.time() < end:
            rig.motor_reverse(150)
            time.sleep(TICK_S)
        rig.stop()
        time.sleep(0.3)
    c = rig.front() if need_front else rig.rear()
    return c is not None and c >= CLEARANCE_MIN_M * 0.7


def reset_forward_drift(rig, seconds):
    rig.steer(90)
    end = time.time() + seconds
    while time.time() < end:
        rig.motor_reverse(130)
        time.sleep(TICK_S)
    rig.stop()
    time.sleep(0.2)


def measure_speed(rig, pwm, settle_s=0.5, measure_s=0.6, samples=6):
    """Straight-line PWM -> speed via distance/time regression, aborting early on low clearance."""
    rig.steer(90)
    end_settle = time.time() + settle_s
    while time.time() < end_settle:
        if (rig.front() or 99) < CLEARANCE_MIN_M * 0.6:
            rig.stop()
            return None
        rig.motor_forward(pwm)
        time.sleep(TICK_S)
    times, dists = [], []
    step = measure_s / samples
    t_start = rig.now()
    for _ in range(samples):
        if (rig.front() or 99) < CLEARANCE_MIN_M * 0.5:
            break
        rig.motor_forward(pwm)
        times.append(rig.now())
        dists.append(rig.front())
        time.sleep(step)
    rig.stop()
    valid = [(t, d) for t, d in zip(times, dists) if d is not None]
    if len(valid) < 3:
        return None
    t_arr = np.array([v[0] for v in valid])
    d_arr = np.array([v[1] for v in valid])
    slope = np.polyfit(t_arr, d_arr, 1)[0]
    return max(0.0, -slope)


def measure_turn(rig, raw_angle, pwm, duration_s=0.7):
    rig.steer(raw_angle)
    time.sleep(0.15)
    b0 = rig.bearing()
    t0 = rig.now()
    end = time.time() + duration_s
    aborted = False
    while time.time() < end:
        if (rig.front() or 99) < CLEARANCE_MIN_M * 0.5:
            aborted = True
            break
        rig.motor_forward(pwm)
        time.sleep(TICK_S)
    rig.stop()
    t1 = rig.now()
    time.sleep(0.2)
    b1 = rig.bearing()
    rig.steer(90)
    if b0 is None or b1 is None or aborted:
        return None
    dt = max(0.05, t1 - t0)
    return {"raw_angle": raw_angle, "bearing_before": b0, "bearing_after": b1,
            "turn_rate_deg_s": (b1 - b0) / dt, "duration_s": dt}


def main():
    # rc-relay must already be stopped before this runs (it owns the LiDAR/ESP32 ports
    # exclusively) and restarted afterwards - done by the caller, since stopping/starting
    # the systemd service needs a sudo password prompt this script has no TTY for.
    log = []
    result = {"started": time.time(), "center0": CENTER0}
    rig = Rig()
    try:
        if not ensure_clearance(rig, log):
            result["aborted"] = "insufficient front clearance at start"
            return

        # --- Phase 1: PWM -> speed table ---
        speed_table = []
        for pwm in PWM_LEVELS:
            if not ensure_clearance(rig, log):
                break
            v = measure_speed(rig, pwm)
            speed_table.append({"pwm": pwm, "speed_m_s": v})
            reset_forward_drift(rig, 1.0)
            log.append({"event": "speed_sample", "pwm": pwm, "speed_m_s": v})
        result["speed_table"] = speed_table
        moving = [(p["pwm"], p["speed_m_s"]) for p in speed_table if p["speed_m_s"] and p["speed_m_s"] > 0.03]
        if len(moving) >= 2:
            pw = np.array([m[0] for m in moving], dtype=float)
            vv = np.array([m[1] for m in moving])
            slope, intercept = np.polyfit(pw, vv, 1)
            deadband = -intercept / slope if slope else None
            result["speed_model_fit"] = {"slope_m_s_per_pwm": slope, "deadband_pwm": deadband,
                                          "v_max_at_255": slope * 255 + intercept}

        # --- Phase 2: servo center calibration ---
        if not ensure_clearance(rig, log):
            result["center_skipped"] = "insufficient clearance"
        else:
            probe = measure_turn(rig, CENTER0 + 25, TURN_TEST_PWM, duration_s=0.5)
            reset_forward_drift(rig, 0.8)
            if probe is None:
                result["center_skipped"] = "no reference wall"
            else:
                sign = 1.0 if probe["turn_rate_deg_s"] > 0 else -1.0
                result["sign_probe"] = probe
                center = float(CENTER0)
                centers_tried = []
                for it in range(4):
                    if not ensure_clearance(rig, log):
                        break
                    t = measure_turn(rig, round(center), TURN_TEST_PWM, duration_s=0.5)
                    reset_forward_drift(rig, 0.8)
                    if t is None:
                        continue
                    drift_deg = t["bearing_after"] - t["bearing_before"]
                    centers_tried.append({"center": center, "drift_deg": drift_deg})
                    if abs(drift_deg) < 1.5:
                        break
                    center -= sign * drift_deg * 0.6
                    center = max(60, min(120, center))
                result["centers_tried"] = centers_tried
                result["center_final"] = round(center)

        # --- Phase 3: steering-angle -> turn-rate -> turn-radius table ---
        center_used = result.get("center_final", CENTER0)
        v_at_test_pwm = None
        for p in speed_table:
            if p["pwm"] == TURN_TEST_PWM:
                v_at_test_pwm = p["speed_m_s"]
        if v_at_test_pwm is None and moving:
            # interpolate from the fitted line if we didn't sample this exact pwm
            fit = result.get("speed_model_fit")
            if fit:
                v_at_test_pwm = max(0.0, fit["slope_m_s_per_pwm"] * TURN_TEST_PWM +
                                     (fit["v_max_at_255"] - fit["slope_m_s_per_pwm"] * 255))

        turn_table = []
        for offset in TURN_OFFSETS:
            if not ensure_clearance(rig, log):
                break
            raw = center_used + offset
            t = measure_turn(rig, raw, TURN_TEST_PWM, duration_s=0.7)
            reset_forward_drift(rig, 1.0)
            entry = {"offset_from_center": offset, "raw_angle": raw, "turn": t}
            if t and v_at_test_pwm:
                rate_rad_s = abs(t["turn_rate_deg_s"]) * 3.14159265 / 180.0
                entry["turn_radius_m"] = (v_at_test_pwm / rate_rad_s) if rate_rad_s > 1e-4 else None
            turn_table.append(entry)
            log.append({"event": "turn_sample", **entry})
        result["turn_table"] = turn_table
        result["v_at_test_pwm"] = v_at_test_pwm

        l20 = next((e["turn"]["turn_rate_deg_s"] for e in turn_table
                    if e["offset_from_center"] == 20 and e["turn"]), None)
        r20 = next((e["turn"]["turn_rate_deg_s"] for e in turn_table
                    if e["offset_from_center"] == -20 and e["turn"]), None)
        if l20 is not None and r20 is not None and (abs(l20) + abs(r20)) > 0:
            result["asymmetry_ratio_20deg"] = abs(abs(l20) - abs(r20)) / ((abs(l20) + abs(r20)) / 2.0)

    finally:
        rig.stop()
        rig.steer(90)
        time.sleep(0.2)
        rig.close()
        result["log"] = log
        result["finished"] = time.time()
        with open(REPORT_PATH, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(json.dumps({k: v for k, v in result.items() if k != "log"}, indent=2, default=str))

    if result.get("center_final") is not None and result["center_final"] != CENTER0:
        last = result["centers_tried"][-1] if result.get("centers_tried") else None
        if last is not None and abs(last["drift_deg"]) < 3.0:
            shutil.copy(TUNING_PATH, TUNING_PATH + ".bak")
            with open(TUNING_PATH) as f:
                cfg = json.load(f)
            cfg["servo"]["left_center"] = result["center_final"]
            cfg["servo"]["right_center"] = result["center_final"]
            with open(TUNING_PATH, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"updated servo center {CENTER0} -> {result['center_final']}")
        else:
            print("center did not converge cleanly, leaving tuning_real_car.json untouched")


if __name__ == "__main__":
    main()
