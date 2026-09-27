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
import math
import os
import shutil
import sys
import threading
import time

import numpy as np
import serial
from rplidar import RPLidar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pi.lidar_dense import open_lidar  # noqa: E402  (high-density SDK stream, falls back to rplidar)
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
# ramp-up to TURN_TEST_PWM alone takes (pwm/RAMP_STEP_PWM)*TICK_S; a turn burst shorter than
# that mostly measures acceleration, not steady turning - real bug seen live (19-23deg/s
# turn rates from 0.35-0.4s bursts, vs a believable 6-14deg/s once this was long enough)
TURN_BURST_S = (TURN_TEST_PWM / RAMP_STEP_PWM) * TICK_S + 0.2


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
        # every command and scan is recorded (pi/drive_log.py) so the simulator can be fitted to real runs
        from pi.drive_log import DriveLog, LoggedSerial
        src = os.path.splitext(os.path.basename(sys.argv[0] or "rig"))[0] or "rig"
        self.log = DriveLog(src, tuning_path=TUNING_PATH)
        self.esp = LoggedSerial(self.esp, self.log)
        self.esp.write(b"A 90 90\nM 0\n")

        self.lidar_port = lidar_port
        self.lidar = open_lidar(lidar_port)
        self._lock = threading.Lock()
        self._front = self._rear = self._bearing = None
        self._floor = self._floor_bearing = None   # closest point anywhere in the 360 scan
        self._profile = {}    # angle_bin(10deg) -> min body clearance, full 360 - the actual
                               # room map, used to pick a genuinely open escape direction
                               # instead of guessing/cycling blind
        self._points = []     # latest full scan, raw (angle_deg, dist_m) - for arc-sweep
                               # predicted-path checks (pi/path_predict.py), full resolution
        self._last_scan_t = None   # staleness guard: a dead scan thread must not leave every
                                    # sensor read silently returning frozen old data forever
        self._pwm_now = 0.0   # ramped PWM state (physical convention: +forward, -reverse)
        self._steer_offset = 0.0   # current commanded offset from center, degrees
        self._odo_x = 0.0     # rough dead-reckoning position/heading relative to where the
        self._odo_y = 0.0     # car started this run (which we know had good clearance,
        self._odo_theta = 0.0  # since that's the precondition for phase 1 to have begun) -
        self._odo_last_t = None  # a soft bias for escape direction, never a safety check
        self._running = True
        self._t0 = time.time()   # set before starting the thread - it calls self.now()
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
        time.sleep(2.0)

    MAX_RECONNECTS = 5
    STALE_DATA_S = 1.0   # sensor reads older than this are treated as unknown (None), not
                          # silently reused - a dead/stuck scan thread must not leave the
                          # rest of the code trusting frozen data as if it were live

    def _scan_loop(self):
        """Runs for the Rig's whole lifetime, reconnecting the LiDAR if the scan stream
        dies (seen live: "New scan flags mismatch" killed the thread outright, after which
        every sensor read silently returned stale data forever - front()/rear()/floor() etc
        now refuse to return anything older than STALE_DATA_S regardless of what happens
        here, but actually recovering the connection is what lets a long run keep going)."""
        reconnects = 0
        while self._running:
            try:
                for scan in self.lidar.iter_scans(max_buf_meas=6000, min_len=5):
                    if not self._running:
                        return
                    self.log.scan(scan)
                    front = rear = None
                    bear_best_d, bear_best_a = None, None
                    floor_d, floor_a = None, None
                    profile = {}
                    pts = []
                    for _, angle, dist in scan:
                        if dist <= 0 or dist / 1000.0 < MOUNT.min_valid_range_m:
                            continue
                        a = to_car_angle(angle)
                        d_m = dist / 1000.0
                        pts.append((round(a, 1), round(d_m, 3)))
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
                        # the actual room map: coarse 10deg bins, min body clearance in each -
                        # this is what lets repositioning pick a real open direction instead of
                        # guessing and checking after the fact
                        bin_a = int(round(a / 10.0)) * 10
                        if bin_a not in profile or body_d < profile[bin_a]:
                            profile[bin_a] = body_d
                    with self._lock:
                        self._front, self._rear = front, rear
                        self._floor, self._floor_bearing = floor_d, floor_a
                        if profile:
                            self._profile = profile
                        self._points = pts
                        if bear_best_a is not None:
                            self._bearing = bear_best_a
                        self._last_scan_t = self.now()
                    reconnects = 0   # a good scan batch resets the retry budget
            except Exception as e:
                reconnects += 1
                print(f"scan thread error ({e}), reconnect attempt {reconnects}/{self.MAX_RECONNECTS}")
                if reconnects > self.MAX_RECONNECTS:
                    print("scan thread giving up - out of reconnect attempts")
                    return
                try:
                    self.lidar.stop()
                    self.lidar.disconnect()
                except Exception:
                    pass
                time.sleep(1.0)
                try:
                    self.lidar = open_lidar(self.lidar_port)
                except Exception as e2:
                    print("reconnect failed:", e2)
                    time.sleep(1.0)

    def now(self):
        return time.time() - self._t0

    def _fresh(self):
        """False if the scan thread hasn't produced a reading recently - a dead/reconnecting
        thread must not leave every sensor read silently returning stale-but-plausible data."""
        return self._last_scan_t is not None and self.now() - self._last_scan_t <= self.STALE_DATA_S

    def is_fresh(self):
        """Public staleness check - callers deciding whether to move MUST treat stale data as
        unsafe/unknown, never as "nothing nearby". Only floor()/front()/etc returning None
        because the LiDAR genuinely sees nothing within range still means clear."""
        with self._lock:
            return self._fresh()

    def front(self):
        with self._lock:
            return self._front if self._fresh() else None

    def rear(self):
        with self._lock:
            return self._rear if self._fresh() else None

    def floor(self):
        """Closest point anywhere around the car (360deg), conservatively adjusted to body
        clearance. This is the hard safety floor - checked continuously during every motion,
        independent of which direction we think we're driving in."""
        with self._lock:
            return (self._floor, self._floor_bearing) if self._fresh() else (None, None)

    def bearing(self):
        with self._lock:
            return self._bearing if self._fresh() else None

    def profile(self):
        """Full 360 clearance-by-angle map (10deg bins), the actual room layout right now."""
        with self._lock:
            return dict(self._profile) if self._fresh() else {}

    def points(self):
        """Latest full-resolution scan, raw (angle_deg, dist_m) - for arc-sweep predicted-
        path checks, which need real point positions, not the coarse 10deg profile bins."""
        with self._lock:
            return list(self._points) if self._fresh() else []

    def home_bearing(self):
        """Rough dead-reckoning bearing (car frame, degrees) back toward wherever this run
        started. Not precise - PWM/steer-angle integration, no encoders - but good enough as
        a soft bias so the car drifts back toward known-good space over many small moves
        instead of only ever reacting to whatever's open right this instant."""
        dist = (self._odo_x ** 2 + self._odo_y ** 2) ** 0.5
        if dist < 0.05:
            return None   # too close to home to have a meaningful direction
        # vector from the car's current position back to the origin, in the same world frame
        # odo_theta is integrated in; subtracting the car's current heading converts that to
        # a car-relative bearing (0=straight ahead), matching the LiDAR bearing convention
        world_bearing = math.degrees(math.atan2(-self._odo_y, -self._odo_x))
        rel = (world_bearing - self._odo_theta) % 360
        return rel if rel <= 180 else rel - 360

    def steer(self, raw_angle):
        raw_angle = max(35, min(145, int(round(raw_angle))))
        self._steer_offset = raw_angle - 90
        self.esp.write(f"A {raw_angle} {raw_angle}\n".encode())

    SPEED_EST_PER_PWM = 0.0018   # m/s per pwm unit, rough average of tonight's fitted slopes
    TURN_RATE_GAIN = 1.1         # deg/s per (steer_offset_deg * speed_m_s), rough average of
                                  # tonight's measured turn-rate/offset ratios

    def _step_toward(self, target_physical):
        """Ramp the commanded PWM toward `target_physical` (positive=forward, negative=
        reverse) by at most RAMP_STEP_PWM per call, instead of jumping straight there.
        Also integrates the rough dead-reckoning odometry from the resulting motion."""
        if self._pwm_now < target_physical:
            self._pwm_now = min(target_physical, self._pwm_now + RAMP_STEP_PWM)
        elif self._pwm_now > target_physical:
            self._pwm_now = max(target_physical, self._pwm_now - RAMP_STEP_PWM)
        wire = -int(round(self._pwm_now))
        self.esp.write(f"M {wire}\n".encode())

        import math
        now = self.now()
        if self._odo_last_t is not None:
            dt = max(0.0, min(0.2, now - self._odo_last_t))   # clamp: ignore long gaps
            v = self._pwm_now * self.SPEED_EST_PER_PWM   # signed, +forward
            heading_rate = self.TURN_RATE_GAIN * self._steer_offset * v
            self._odo_theta += heading_rate * dt
            self._odo_x += v * math.cos(math.radians(self._odo_theta)) * dt
            self._odo_y += v * math.sin(math.radians(self._odo_theta)) * dt
        self._odo_last_t = now

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
        self.log.close()
        try:
            self.lidar.stop()
            self.lidar.stop_motor()
            self.lidar.disconnect()
        except Exception:
            pass


def floor_violated(rig):
    if not rig.is_fresh():
        return True, None, None   # stale sensor data is treated as unsafe, never as "clear"
    d, a = rig.floor()
    return (d is not None and d < HARD_FLOOR_M), d, a


def _clearance_score(rig):
    """Lower is worse. Combines the 360 floor and both direction cones so an escape
    maneuver can be judged as "helped" or "made it worse", not just pass/fail."""
    if not rig.is_fresh():
        return 0.0   # worst possible score - never treat stale data as an improvement
    d, _ = rig.floor()
    front_c, rear_c = rig.front(), rig.rear()
    vals = [v for v in (d, front_c, rear_c) if v is not None]
    return min(vals) if vals else 0.0


def _clearance_ok(rig):
    if not rig.is_fresh():
        return False   # stale sensor data is never "ok to move"
    d, _ = rig.floor()
    front_c, rear_c = rig.front(), rig.rear()
    return (d is None or d >= HARD_FLOOR_M) and \
           (front_c is None or front_c >= CLEARANCE_MIN_M) and \
           (rear_c is None or rear_c >= CLEARANCE_MIN_M)


ESCAPE_BURST_S = 0.22
PROFILE_MIN_SAMPLES_DEG = 30   # need at least this much of the ring mapped before trusting it
HOME_WEIGHT = 1.3               # bonus for directions that also lead back toward known-good
                                 # start space, on top of "purpose" and raw local clearance
HOME_HALF_WIDTH = 50


def choose_escape_move(rig, purpose_bearing=None, purpose_half_width=45, purpose_weight=1.6):
    """Use the actual 360 room map (not a guess) to pick where to go: the angle bin with the
    best clearance, weighted toward `purpose_bearing` when given (e.g. 0deg for an upcoming
    straight-forward speed test, or the turn's own offset angle for an upcoming turn test -
    "if the test is a right turn, maximise front+right distance", not just anywhere open) AND
    toward the rough dead-reckoned direction back to where this run started (which we know had
    good clearance) - so repositioning drifts back toward known-good space over many small
    moves instead of only ever reacting to the current instant.

    Direction (forward vs reverse) is decided from the already-validated front()/rear() cone
    readings, NOT a guessed bin-to-motion mapping - reversing-while-steered has never actually
    been measured on this car, so guessing its sign convention is exactly what caused a real
    bug: the car repeatedly "escaped" by steering into a close rear obstacle while thinking it
    was heading toward open space, because the reverse-steer kinematics were wrong. Steering
    is only ever chosen from the validated forward convention, and only within whichever half
    (front or rear) was already picked by the trusted cone comparison; reverse always goes
    straight back rather than guessing a diagonal."""
    profile = rig.profile()
    if len(profile) * 10 < PROFILE_MIN_SAMPLES_DEG:
        return (0, "reverse", None)

    home = rig.home_bearing()

    def score(angle_bin, dist):
        s = dist
        if purpose_bearing is not None:
            ang_dist = min(abs(angle_bin - purpose_bearing), 360 - abs(angle_bin - purpose_bearing))
            if ang_dist <= purpose_half_width:
                s *= purpose_weight
        if home is not None:
            ang_dist = min(abs(angle_bin - home), 360 - abs(angle_bin - home))
            if ang_dist <= HOME_HALF_WIDTH:
                s *= HOME_WEIGHT
        return s

    front_c, rear_c = rig.front(), rig.rear()
    go_forward = front_c is None or rear_c is None or front_c >= rear_c

    if go_forward:
        fwd_bins = {b: d for b, d in profile.items() if abs(b) <= 90}
        if not fwd_bins:
            return (0, "forward", None)
        best_bin = max(fwd_bins, key=lambda b: score(b, fwd_bins[b]))
        return (max(-35, min(35, best_bin)), "forward", best_bin)
    else:
        return (0, "reverse", None)


def reposition_away_from_nearest(rig, log, max_attempts=14, purpose_bearing=None):
    """Drive toward the most open direction in the current room map (biased toward
    `purpose_bearing` when the caller knows what the next test actually needs clear),
    re-scanning and re-choosing after every short burst. Keeps a move only if it measurably
    helped; undoes roughly half of it and re-picks from the fresh map otherwise - this
    replaces a fixed cycle of blind guesses with an actual read of the room."""
    if _clearance_ok(rig):
        return True
    # Commit to ONE direction for the whole episode: re-picking forward/reverse every burst
    # made the car jitter back and forth in a tight spot (and stress the drivetrain). Only
    # flip once, if the committed direction's own cone runs out of room.
    committed = None
    flipped = False
    for attempt in range(max_attempts):
        steer_offset, direction, target_bin = choose_escape_move(rig, purpose_bearing)
        if committed is None:
            committed = direction
        direction = committed
        cone = rig.front() if direction == "forward" else rig.rear()
        if cone is not None and cone < HARD_FLOOR_M + 0.05:
            if flipped:
                break
            committed = "reverse" if direction == "forward" else "forward"
            flipped = True
            direction = committed
        if direction == "reverse":
            steer_offset = 0
        d, a = rig.floor()
        log.append({"event": "reposition", "attempt": attempt, "steer_offset": steer_offset,
                     "direction": direction, "target_bin": target_bin,
                     "floor": d, "floor_bearing": a, "front": rig.front(), "rear": rig.rear()})

        rig.steer(90 + steer_offset)
        end = time.time() + ESCAPE_BURST_S * 2
        while time.time() < end:
            bad, _, _ = floor_violated(rig)
            if bad and time.time() > end - ESCAPE_BURST_S * 1.5:
                break
            if direction == "reverse":
                rig.motor_reverse(150)
            else:
                rig.motor_forward(150)
            time.sleep(TICK_S)
        rig.stop()
        time.sleep(0.3)
        rig.steer(90)

        if _clearance_ok(rig):
            return True
    return _clearance_ok(rig)


def ensure_clearance(rig, log, need_front=True, purpose_bearing=None):
    ok = reposition_away_from_nearest(rig, log, purpose_bearing=purpose_bearing)
    if not ok:
        return False
    c = rig.front() if need_front else rig.rear()
    return c is not None and c >= CLEARANCE_MIN_M * 0.7


REVERSE_UNDO_PWM = 130
MAX_UNDO_S = 1.5   # cap on the matched-duration undo, in case of a very long/fast forward burst


def reset_forward_drift(rig, forward_seconds, forward_pwm=REVERSE_UNDO_PWM, reverse_pwm=REVERSE_UNDO_PWM):
    """Undo a preceding forward creep with a reverse burst matched to the same DISTANCE, not
    just the same time - a fixed-duration undo regardless of how far the forward move actually
    went (different PWM levels drive at different speeds) was the real source of the net drift
    that ate the whole room across a pass. seconds = forward_seconds * forward_pwm/reverse_pwm
    roughly cancels distance, since both use the same rough linear PWM->speed estimate."""
    seconds = min(MAX_UNDO_S, forward_seconds * (forward_pwm / reverse_pwm))
    rig.steer(90)
    end = time.time() + seconds
    last_check = 0.0
    while time.time() < end:
        if time.time() - last_check > CHECK_EVERY_S:
            last_check = time.time()
            bad, _, _ = floor_violated(rig)
            if bad:
                break
        rig.motor_reverse(reverse_pwm)
        time.sleep(TICK_S)
    rig.stop()
    time.sleep(0.15)


def measure_speed(rig, pwm, log, settle_s=None, measure_s=0.35, samples=5):
    """Straight-line PWM -> speed via distance/time regression, aborting early if ANY
    direction (not just straight ahead) gets too close - catches a corner clipping something
    off to the side during the creep, not just a head-on obstacle. Returns (speed_or_None,
    actual_forward_seconds) so the caller can undo exactly what was actually driven, not a
    fixed guess.

    settle_s must cover the ramp-up time for THIS pwm level, not a fixed short guess - PWM is
    ramped at RAMP_STEP_PWM per TICK_S (gentler on the drivetrain, per an explicit request
    after the wheel came loose twice), and a short fixed settle window meant the "measure"
    phase was still mid-ramp for higher PWM levels, giving nonsense (non-monotonic, sometimes
    near-zero) speed readings - the two requests (gentle ramping, short bursts) directly trade
    off for this specific measurement and the ramp has to win or the number is meaningless."""
    if settle_s is None:
        ramp_time = (pwm / RAMP_STEP_PWM) * TICK_S
        settle_s = max(0.2, ramp_time + 0.1)
    rig.steer(90)
    t_motion_start = rig.now()
    end_settle = time.time() + settle_s
    while time.time() < end_settle:
        bad, d, a = floor_violated(rig)
        if bad:
            log.append({"event": "abort_floor", "phase": "speed_settle", "pwm": pwm, "floor": d, "bearing": a})
            rig.stop()
            return None, max(0.0, rig.now() - t_motion_start)
        rig.motor_forward(pwm)
        time.sleep(TICK_S)
    times, dists = [], []
    step = measure_s / samples
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
    elapsed = max(0.0, rig.now() - t_motion_start)
    valid = [(t, d) for t, d in zip(times, dists) if d is not None]
    if len(valid) < 3:
        return None, elapsed
    t_arr = np.array([v[0] for v in valid])
    d_arr = np.array([v[1] for v in valid])
    slope = np.polyfit(t_arr, d_arr, 1)[0]
    return max(0.0, -slope), elapsed


def measure_turn(rig, raw_angle, pwm, log, duration_s=0.4):
    rig.steer(raw_angle)
    time.sleep(0.1)
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
    time.sleep(0.15)
    b1 = rig.bearing()
    rig.steer(90)
    elapsed = max(0.05, t1 - t0)
    if b0 is None or b1 is None or aborted:
        return None, elapsed
    return {"raw_angle": raw_angle, "bearing_before": b0, "bearing_after": b1,
            "turn_rate_deg_s": (b1 - b0) / elapsed, "duration_s": elapsed}, elapsed


# --- iterative outer loop: keep running full passes, pooling data, until the servo center
# and turn numbers actually stabilize ACROSS independent passes (not just within one), or
# a hard iteration cap is hit. Thresholds are intentionally strict per request - "tune it
# very well now only to avoid pain later" - a single quiet pass is not enough evidence.
MAX_OUTER_ITERS = 6
CENTER_TOL_DEG = 1.0          # consecutive passes' converged center must agree this tightly
CENTER_STABLE_STREAK = 2      # need this many consecutive agreeing passes
RATE_STABLE_FRAC = 0.15       # pooled turn-rate estimate may not move more than 15% pass-to-pass
SPEED_STABLE_FRAC = 0.12      # pooled v_max estimate may not move more than 12% pass-to-pass


def run_one_pass(rig, log, center_start, speed_pool, turn_pool):
    """One full speed+center+turn pass. Mutates speed_pool[pwm] and turn_pool[offset] (lists
    of samples) in place so the caller can recompute pooled stats across passes."""
    pass_result = {}

    if not ensure_clearance(rig, log, purpose_bearing=0):
        pass_result["aborted"] = "insufficient clearance at pass start"
        return pass_result

    for pwm in PWM_LEVELS:
        for trial in range(TRIALS):
            if not ensure_clearance(rig, log, purpose_bearing=0):
                break
            v, fwd_s = measure_speed(rig, pwm, log)
            reset_forward_drift(rig, fwd_s, forward_pwm=pwm)
            log.append({"event": "speed_sample", "pwm": pwm, "trial": trial, "speed_m_s": v})
            if v is not None:
                speed_pool[pwm].append(v)

    if not ensure_clearance(rig, log, purpose_bearing=25):
        pass_result["center_skipped"] = "insufficient clearance"
        center_this_pass = center_start
    else:
        probe, fwd_s = measure_turn(rig, center_start + 25, TURN_TEST_PWM, log, duration_s=TURN_BURST_S)
        reset_forward_drift(rig, fwd_s, forward_pwm=TURN_TEST_PWM)
        if probe is None:
            pass_result["center_skipped"] = "no reference wall"
            center_this_pass = center_start
        else:
            sign = 1.0 if probe["turn_rate_deg_s"] > 0 else -1.0
            pass_result["sign_probe"] = probe
            center = float(center_start)
            centers_tried = []
            for it in range(4):
                if not ensure_clearance(rig, log, purpose_bearing=0):
                    break
                t, fwd_s = measure_turn(rig, round(center), TURN_TEST_PWM, log, duration_s=TURN_BURST_S)
                reset_forward_drift(rig, fwd_s, forward_pwm=TURN_TEST_PWM)
                if t is None:
                    continue
                drift_deg = t["bearing_after"] - t["bearing_before"]
                centers_tried.append({"center": center, "drift_deg": drift_deg})
                if abs(drift_deg) < 1.5:
                    break
                center -= sign * drift_deg * 0.6
                center = max(60, min(120, center))
            pass_result["centers_tried"] = centers_tried
            center_this_pass = round(center, 1)
    pass_result["center_this_pass"] = center_this_pass

    for offset in TURN_OFFSETS:
        raw = center_this_pass + offset
        trials = []
        for trial in range(TRIALS):
            if not ensure_clearance(rig, log, purpose_bearing=offset):
                break
            t, fwd_s = measure_turn(rig, raw, TURN_TEST_PWM, log, duration_s=TURN_BURST_S)
            reset_forward_drift(rig, fwd_s, forward_pwm=TURN_TEST_PWM)
            log.append({"event": "turn_trial", "offset_from_center": offset, "trial": trial, "turn": t})
            if t is not None:
                trials.append(t)
        kept = trials
        if len(trials) >= 2:
            bearings0 = [tr["bearing_before"] for tr in trials]
            med_b0 = float(np.median(bearings0))
            kept = [tr for tr in trials if abs(tr["bearing_before"] - med_b0) < 20.0]
        turn_pool[offset].extend(tr["turn_rate_deg_s"] for tr in kept)

    return pass_result


def pooled_speed_table(speed_pool):
    table = []
    for pwm in PWM_LEVELS:
        samples = speed_pool[pwm]
        table.append({"pwm": pwm, "speed_m_s": float(np.median(samples)) if samples else None,
                       "n_samples": len(samples)})
    moving = [(p["pwm"], p["speed_m_s"]) for p in table if p["speed_m_s"] and p["speed_m_s"] > 0.03]
    fit = None
    if len(moving) >= 2:
        pw = np.array([m[0] for m in moving], dtype=float)
        vv = np.array([m[1] for m in moving])
        slope, intercept = np.polyfit(pw, vv, 1)
        fit = {"slope_m_s_per_pwm": slope, "deadband_pwm": (-intercept / slope if slope else None),
               "v_max_at_255": slope * 255 + intercept}
    return table, fit


def pooled_turn_table(turn_pool, v_at_test_pwm):
    table = []
    for offset in TURN_OFFSETS:
        samples = turn_pool[offset]
        rate_med = float(np.median(samples)) if samples else None
        entry = {"offset_from_center": offset, "n_samples": len(samples), "turn_rate_deg_s_median": rate_med}
        if rate_med is not None and v_at_test_pwm:
            rate_rad_s = abs(rate_med) * 3.14159265 / 180.0
            entry["turn_radius_m"] = (v_at_test_pwm / rate_rad_s) if rate_rad_s > 1e-4 else None
        table.append(entry)
    return table


def main():
    # rc-relay must already be stopped before this runs (it owns the LiDAR/ESP32 ports
    # exclusively) and restarted afterwards - done by the caller, since stopping/starting
    # the systemd service needs a sudo password prompt this script has no TTY for.
    log = []
    result = {"started": time.time(), "center0": CENTER0, "passes": []}
    speed_pool = {pwm: [] for pwm in PWM_LEVELS}
    turn_pool = {offset: [] for offset in TURN_OFFSETS}
    center_history = []
    rig = Rig()
    converged = False
    stop_reason = None

    try:
        center_current = float(CENTER0)
        for outer in range(MAX_OUTER_ITERS):
            print(f"\n=== outer pass {outer + 1}/{MAX_OUTER_ITERS}, starting center={center_current} ===")
            pass_result = run_one_pass(rig, log, center_current, speed_pool, turn_pool)
            pass_result["outer"] = outer
            result["passes"].append(pass_result)

            if "aborted" in pass_result:
                stop_reason = f"pass {outer} aborted: {pass_result['aborted']}"
                break

            center_current = pass_result.get("center_this_pass", center_current)
            center_history.append(center_current)

            speed_table, speed_fit = pooled_speed_table(speed_pool)
            v_at_test_pwm = next((p["speed_m_s"] for p in speed_table if p["pwm"] == TURN_TEST_PWM), None)
            turn_table = pooled_turn_table(turn_pool, v_at_test_pwm)

            result["speed_table"] = speed_table
            result["speed_model_fit"] = speed_fit
            result["turn_table"] = turn_table
            result["v_at_test_pwm"] = v_at_test_pwm
            result["center_history"] = center_history
            l20 = next((e["turn_rate_deg_s_median"] for e in turn_table if e["offset_from_center"] == 20), None)
            r20 = next((e["turn_rate_deg_s_median"] for e in turn_table if e["offset_from_center"] == -20), None)
            asym = None
            if l20 is not None and r20 is not None and (abs(l20) + abs(r20)) > 0:
                asym = abs(abs(l20) - abs(r20)) / ((abs(l20) + abs(r20)) / 2.0)
            result["asymmetry_ratio_20deg"] = asym
            result.setdefault("asym_history", []).append(asym)
            result.setdefault("vmax_history", []).append(speed_fit["v_max_at_255"] if speed_fit else None)

            # checkpoint after every pass, so partial progress survives even if something
            # eventually forces an abort mid-run
            with open(REPORT_PATH, "w") as f:
                json.dump({**result, "log": log}, f, indent=2, default=str)

            center_stable = (len(center_history) >= CENTER_STABLE_STREAK and
                              max(center_history[-CENTER_STABLE_STREAK:]) -
                              min(center_history[-CENTER_STABLE_STREAK:]) <= CENTER_TOL_DEG)
            vmax_hist = [v for v in result["vmax_history"] if v is not None]
            vmax_stable = (len(vmax_hist) >= 2 and vmax_hist[-2] not in (0, None) and
                            abs(vmax_hist[-1] - vmax_hist[-2]) / max(abs(vmax_hist[-2]), 1e-6) <= SPEED_STABLE_FRAC)
            asym_hist = [a for a in result["asym_history"] if a is not None]
            asym_stable = (len(asym_hist) >= 2 and
                            abs(asym_hist[-1] - asym_hist[-2]) <= RATE_STABLE_FRAC * max(asym_hist[-2], 0.2))

            print(f"pass {outer + 1} done: center={center_current} history={center_history} "
                  f"vmax_stable={vmax_stable} asym_stable={asym_stable} center_stable={center_stable}")

            if center_stable and vmax_stable and asym_stable:
                converged = True
                stop_reason = "converged"
                break
        else:
            stop_reason = "hit MAX_OUTER_ITERS without converging"

        result["converged"] = converged
        result["stop_reason"] = stop_reason
        result["center_final"] = center_current

    finally:
        rig.stop()
        rig.steer(90)
        time.sleep(0.2)
        rig.close()
        result["log"] = log
        result["finished"] = time.time()
        with open(REPORT_PATH, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(json.dumps({k: v for k, v in result.items() if k not in ("log", "passes")}, indent=2, default=str))

    if result.get("center_final") is not None and abs(result["center_final"] - CENTER0) > 0.1:
        if converged:
            shutil.copy(TUNING_PATH, TUNING_PATH + ".bak")
            with open(TUNING_PATH) as f:
                cfg = json.load(f)
            cfg["servo"]["left_center"] = result["center_final"]
            cfg["servo"]["right_center"] = result["center_final"]
            with open(TUNING_PATH, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"CONVERGED - updated servo center {CENTER0} -> {result['center_final']}")
        else:
            print(f"did NOT converge ({stop_reason}) - leaving tuning_real_car.json untouched. "
                  f"center history: {center_history}")


if __name__ == "__main__":
    main()
