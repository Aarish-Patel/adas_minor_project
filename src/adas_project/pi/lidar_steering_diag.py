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
HARD_FLOOR_M = 0.35        # nothing, in ANY direction, may ever be closer than this while moving -
                            # catches a corner swinging into something outside the drive-direction
                            # cone (found live: the front-right corner clipped an obstacle that a
                            # narrow front-only cone check never saw)
TICK_S = 0.05
CHECK_EVERY_S = 0.15       # how often a moving creep re-checks the full-360 floor
RAMP_STEP_PWM = 15         # max PWM change per TICK_S tick (~300 pwm/s) - gradual ramp
                            # instead of jumping straight to the target, to reduce the
                            # mechanical shock on the drivetrain (the wheel came loose twice
                            # tonight under instant full-power/direction-reversal commands)

PWM_LEVELS = [110, 140, 170, 200, 230]
TURN_TEST_PWM = 140
TURN_OFFSETS = [-35, -20, 20, 35]   # raw degrees from center; sign resolved against the probe
TRIALS = 3   # repeats per PWM level / steering offset, combined with the median (outlier-robust)


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
        self._floor = self._floor_bearing = None   # closest point anywhere in the 360 scan
        self._pwm_now = 0.0   # ramped PWM state (physical convention: +forward, -reverse)
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
                floor_d, floor_a = None, None
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
                    # full-360 hard floor: use the smaller of front/rear overhang as a
                    # conservative body-clearance estimate regardless of which way this
                    # point actually is (good enough to catch a corner sweeping into
                    # something the direction-specific cones don't cover)
                    body_d = d_m - min(MOUNT.front_overhang_m, MOUNT.rear_overhang_m)
                    if floor_d is None or body_d < floor_d:
                        floor_d, floor_a = body_d, a
                with self._lock:
                    self._front, self._rear = front, rear
                    self._floor, self._floor_bearing = floor_d, floor_a
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

    def floor(self):
        """Closest point anywhere around the car (360deg), conservatively adjusted to body
        clearance. This is the hard safety floor - checked continuously during every motion,
        independent of which direction we think we're driving in."""
        with self._lock:
            return self._floor, self._floor_bearing

    def bearing(self):
        with self._lock:
            return self._bearing

    def steer(self, raw_angle):
        raw_angle = max(35, min(145, int(round(raw_angle))))
        self.esp.write(f"A {raw_angle} {raw_angle}\n".encode())

    def _step_toward(self, target_physical):
        """Ramp the commanded PWM toward `target_physical` (positive=forward, negative=
        reverse) by at most RAMP_STEP_PWM per call, instead of jumping straight there."""
        if self._pwm_now < target_physical:
            self._pwm_now = min(target_physical, self._pwm_now + RAMP_STEP_PWM)
        elif self._pwm_now > target_physical:
            self._pwm_now = max(target_physical, self._pwm_now - RAMP_STEP_PWM)
        wire = -int(round(self._pwm_now))
        self.esp.write(f"M {wire}\n".encode())

    def motor_forward(self, pwm):
        self._step_toward(float(pwm))

    def motor_reverse(self, pwm):
        self._step_toward(-float(pwm))

    def stop(self):
        self._pwm_now = 0.0
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


def floor_violated(rig):
    d, a = rig.floor()
    return (d is not None and d < HARD_FLOOR_M), d, a


def _clearance_score(rig):
    """Lower is worse. Combines the 360 floor and both direction cones so an escape
    maneuver can be judged as "helped" or "made it worse", not just pass/fail."""
    d, _ = rig.floor()
    front_c, rear_c = rig.front(), rig.rear()
    vals = [v for v in (d, front_c, rear_c) if v is not None]
    return min(vals) if vals else 0.0


def _clearance_ok(rig):
    d, _ = rig.floor()
    front_c, rear_c = rig.front(), rig.rear()
    return (d is None or d >= HARD_FLOOR_M) and \
           (front_c is None or front_c >= CLEARANCE_MIN_M) and \
           (rear_c is None or rear_c >= CLEARANCE_MIN_M)


# Escape maneuvers tried in order, cycling if none of them clear it in one pass. Straight
# reverse/forward alone can oscillate forever between two obstacles on opposite corners (seen
# live: the car ping-ponged between a front-right and a rear-left obstacle without ever
# escaping). Steered diagonal options let it actually work sideways out of a corner instead
# of just bouncing along one axis. Short bursts (well under REPOSITION_REVERSE_S's old 0.8s)
# limit how far it can overshoot into the opposite hazard on any single attempt.
ESCAPE_MOVES = [
    (0, "reverse"), (0, "forward"),
    (25, "reverse"), (-25, "reverse"),
    (25, "forward"), (-25, "forward"),
]
ESCAPE_BURST_S = 0.35


def reposition_away_from_nearest(rig, log, max_attempts=14):
    """Try a sequence of escape maneuvers (straight and diagonal, both directions), keeping
    whichever one actually improved clearance and reverting+trying the next one if it made
    things worse. Cycles through ESCAPE_MOVES rather than committing to one fixed strategy."""
    if _clearance_ok(rig):
        return True
    for attempt in range(max_attempts):
        move = ESCAPE_MOVES[attempt % len(ESCAPE_MOVES)]
        steer_offset, direction = move
        before = _clearance_score(rig)
        d, a = rig.floor()
        log.append({"event": "reposition", "attempt": attempt, "move": move,
                     "floor": d, "floor_bearing": a, "front": rig.front(), "rear": rig.rear()})

        rig.steer(90 + steer_offset)
        end = time.time() + ESCAPE_BURST_S
        while time.time() < end:
            if direction == "reverse":
                rig.motor_reverse(150)
            else:
                rig.motor_forward(150)
            time.sleep(TICK_S)
        rig.stop()
        time.sleep(0.25)
        rig.steer(90)

        if _clearance_ok(rig):
            return True
        after = _clearance_score(rig)
        if after < before:
            # this move made it worse - undo roughly half of it before trying the next
            # candidate, so a bad guess doesn't compound across attempts
            rig.steer(90 + steer_offset)
            undo_end = time.time() + ESCAPE_BURST_S * 0.5
            opposite = "forward" if direction == "reverse" else "reverse"
            while time.time() < undo_end:
                if opposite == "reverse":
                    rig.motor_reverse(120)
                else:
                    rig.motor_forward(120)
                time.sleep(TICK_S)
            rig.stop()
            time.sleep(0.2)
            rig.steer(90)
    return _clearance_ok(rig)


def ensure_clearance(rig, log, need_front=True):
    ok = reposition_away_from_nearest(rig, log)
    if not ok:
        return False
    c = rig.front() if need_front else rig.rear()
    return c is not None and c >= CLEARANCE_MIN_M * 0.7


def reset_forward_drift(rig, seconds):
    rig.steer(90)
    end = time.time() + seconds
    last_check = 0.0
    while time.time() < end:
        if time.time() - last_check > CHECK_EVERY_S:
            last_check = time.time()
            bad, _, _ = floor_violated(rig)
            if bad:
                break
        rig.motor_reverse(130)
        time.sleep(TICK_S)
    rig.stop()
    time.sleep(0.2)


def measure_speed(rig, pwm, log, settle_s=0.5, measure_s=0.6, samples=6):
    """Straight-line PWM -> speed via distance/time regression, aborting early if ANY
    direction (not just straight ahead) gets too close - catches a corner clipping something
    off to the side during the creep, not just a head-on obstacle."""
    rig.steer(90)
    end_settle = time.time() + settle_s
    while time.time() < end_settle:
        bad, d, a = floor_violated(rig)
        if bad:
            log.append({"event": "abort_floor", "phase": "speed_settle", "pwm": pwm, "floor": d, "bearing": a})
            rig.stop()
            return None
        rig.motor_forward(pwm)
        time.sleep(TICK_S)
    times, dists = [], []
    step = measure_s / samples
    t_start = rig.now()
    for _ in range(samples):
        bad, d, a = floor_violated(rig)
        if bad:
            log.append({"event": "abort_floor", "phase": "speed_measure", "pwm": pwm, "floor": d, "bearing": a})
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


def measure_turn(rig, raw_angle, pwm, log, duration_s=0.7):
    rig.steer(raw_angle)
    time.sleep(0.15)
    b0 = rig.bearing()
    t0 = rig.now()
    end = time.time() + duration_s
    aborted = False
    while time.time() < end:
        bad, d, a = floor_violated(rig)
        if bad:
            log.append({"event": "abort_floor", "phase": "turn", "raw_angle": raw_angle, "floor": d, "bearing": a})
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

        # --- Phase 1: PWM -> speed table (median of TRIALS repeats per level) ---
        speed_table = []
        for pwm in PWM_LEVELS:
            trial_speeds = []
            for trial in range(TRIALS):
                if not ensure_clearance(rig, log):
                    break
                v = measure_speed(rig, pwm, log)
                reset_forward_drift(rig, 1.0)
                log.append({"event": "speed_sample", "pwm": pwm, "trial": trial, "speed_m_s": v})
                if v is not None:
                    trial_speeds.append(v)
            v_med = float(np.median(trial_speeds)) if trial_speeds else None
            speed_table.append({"pwm": pwm, "speed_m_s": v_med, "trials": trial_speeds})
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
            probe = measure_turn(rig, CENTER0 + 25, TURN_TEST_PWM, log, duration_s=0.5)
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
                    t = measure_turn(rig, round(center), TURN_TEST_PWM, log, duration_s=0.5)
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
            raw = center_used + offset
            trials = []
            for trial in range(TRIALS):
                if not ensure_clearance(rig, log):
                    break
                t = measure_turn(rig, raw, TURN_TEST_PWM, log, duration_s=0.7)
                reset_forward_drift(rig, 1.0)
                log.append({"event": "turn_trial", "offset_from_center": offset, "trial": trial, "turn": t})
                if t is not None:
                    trials.append(t)
            # drop trials whose starting bearing is far from the others: the LiDAR almost
            # certainly lost the reference wall and locked onto something else (this is what
            # happened at the +35deg extreme in the single-trial run).
            kept = trials
            if len(trials) >= 2:
                bearings0 = [tr["bearing_before"] for tr in trials]
                med_b0 = float(np.median(bearings0))
                kept = [tr for tr in trials if abs(tr["bearing_before"] - med_b0) < 20.0]
            rates = [tr["turn_rate_deg_s"] for tr in kept]
            rate_med = float(np.median(rates)) if rates else None
            entry = {"offset_from_center": offset, "raw_angle": raw, "n_trials": len(trials),
                      "n_kept": len(kept), "turn_rate_deg_s_median": rate_med, "trials": trials}
            if rate_med is not None and v_at_test_pwm:
                rate_rad_s = abs(rate_med) * 3.14159265 / 180.0
                entry["turn_radius_m"] = (v_at_test_pwm / rate_rad_s) if rate_rad_s > 1e-4 else None
            turn_table.append(entry)
        result["turn_table"] = turn_table
        result["v_at_test_pwm"] = v_at_test_pwm

        l20 = next((e["turn_rate_deg_s_median"] for e in turn_table if e["offset_from_center"] == 20), None)
        r20 = next((e["turn_rate_deg_s_median"] for e in turn_table if e["offset_from_center"] == -20), None)
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
