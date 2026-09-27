"""Drive the car over WiFi from rc_controller.py, with the Pi blocking motion ONLY toward
a nearby obstacle. Steering is never touched, and the other direction of travel (e.g.
reverse, if forward is blocked) always stays available.

Architecture:
    laptop (rc_controller.py, TRANSPORT="wifi")  --UDP-->  this relay, on the Pi
    this relay  --USB serial-->  ESP32  (the already-proven-reliable link)

On the laptop, point rc_controller.py at the Pi instead of the ESP32:
    ESP32_IP = "192.168.1.3"     # the Pi's address, not the ESP32's

Direction convention: rc_controller.py's build_command() already applies MOTOR_REVERSED
before putting a value on the wire, so the WIRE value's sign is the OPPOSITE of the
physical direction on this car. Everything below converts wire -> physical once on
receipt, reasons about physical direction only, then converts back once before sending.
"""

import csv
import glob
import math
import os
import signal
import socket
import sys
import threading
import time

import numpy as np
import serial
from rplidar import RPLidar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pi.lidar_dense import open_lidar  # noqa: E402  (high-density SDK stream, falls back to rplidar)
from adas.config import load_tuning  # noqa: E402
from adas.tracking import Tracker, moving_object_contact  # noqa: E402
from adas.acc import FollowController  # noqa: E402
from pi.path_predict import VP, delta_for_offset  # noqa: E402

TUNING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning_real_car.json")
TUNING = load_tuning(TUNING_PATH)
# the car model fitted from drive logs (pi/car_model.json) replaces the hand-measured speed model
from pi.relay_assists import apply_car_model  # noqa: E402
_cm = apply_car_model(TUNING)
if _cm:
    print("car model from drive logs:", _cm)

UDP_PORT = 4210

FRONT_OFFSET_DEG = TUNING.mount.yaw_offset_deg       # raw LiDAR angle that is the car's straight-ahead
CONE_DEG = 25                                        # +/- around straight ahead / straight behind
                                                     # (widened from 15: gives the closing-speed
                                                     # check more lead time on something crossing
                                                     # in from the side before it's dead ahead)
FRONT_OVERHANG_M = TUNING.mount.front_overhang_m     # LiDAR to front bumper
REAR_OVERHANG_M = TUNING.mount.rear_overhang_m       # LiDAR to rear bumper
LEFT_OVERHANG_M = TUNING.mount.left_overhang_m       # LiDAR to left side edge
RIGHT_OVERHANG_M = TUNING.mount.right_overhang_m     # LiDAR to right side edge
MIN_VALID_RANGE_M = TUNING.mount.min_valid_range_m   # ignore raw readings closer than this
                                                     # (self-hits: wires/mount clutter)


def body_overhang(a_deg):
    """Distance from the LiDAR to the car's own body edge, in the direction of bearing
    `a_deg` (car frame, 0=front) - i.e. where a ray at this angle exits the car's own
    rectangular footprint. Replaces a flat min(front,rear) overhang that was subtracted from
    EVERY bearing including the sides, which is both wrong (the car is much narrower than it
    is long, so a fixed ~0.16m constant badly overestimates how much of a side reading is
    "inside the car") and, worse, wrong in exactly the direction that matters most for this
    check - it made side/diagonal points look closer to the body than they really are, which
    is why this 360deg backstop was firing on 68% of a real drive log and blocking BOTH
    directions almost the whole time, matching a real complaint ("won't let me steer away and
    leave or go back") that had nothing to do with actual obstacles."""
    rad = math.radians(a_deg)
    cx, sy = math.cos(rad), math.sin(rad)
    candidates = []
    if cx > 1e-6:
        candidates.append(FRONT_OVERHANG_M / cx)
    elif cx < -1e-6:
        candidates.append(REAR_OVERHANG_M / -cx)
    if sy > 1e-6:
        candidates.append(LEFT_OVERHANG_M / sy)
    elif sy < -1e-6:
        candidates.append(RIGHT_OVERHANG_M / -sy)
    return min(candidates) if candidates else FRONT_OVERHANG_M
HOLD_AFTER_LOST_READING_S = 1.0   # a lost reading right after a block does NOT mean "clear"

# Stop distance now SCALES with how fast the gap is closing (measured straight from
# consecutive LiDAR readings, no PWM/speed calibration needed): a fast approach needs a
# bigger margin than a slow creep, because the car covers more ground during the
# LiDAR/relay/motor reaction delay and while coasting to a stop.
BASE_MARGIN_M = 0.10        # margin kept even at ~zero speed (must stay above the ~0.04 m
                            # sensor blind-zone floor: MIN_VALID_RANGE_M - overhang)
REACTION_TIME_S = 0.25      # LiDAR scan interval + relay loop + motor response, combined
ASSUMED_DECEL = 1.0         # m/s^2 the car can coast-stop at - NOT calibrated, conservative
                            # guess; recalibrate with pi/calibrate.py once you have a real
                            # number, and tighten this if it proves too cautious.
MAX_CLOSING_SPEED_M_S = 3.0  # clip absurd speed spikes from noise

# The front/rear cones above only cover straight-ahead/straight-behind motion within +-25deg.
# They are blind to something close off to the SIDE that a turning front (or rear) corner can
# swing into - confirmed live: the car's front-right corner clipped an obstacle that never
# showed up in either narrow cone. WIDE_CONE_DEG widens that check to +-90deg (the whole front
# half vs. the whole rear half) for a corner-strike backstop, but - unlike an earlier version
# of this that used one single global "anything anywhere" flag blocking BOTH directions
# regardless of which side the close thing was actually on - it stays split front/rear, same
# as the narrow cones: a real drive log showed that single global flag true 68% of the session
# (a wall the car happened to be parked next to on one side) and blocking reverse right along
# with forward, matching a live complaint that the car "wouldn't let me steer away and leave
# or go back" even though going back had nothing to do with that wall.
WIDE_CONE_DEG = 90
# "path": emergency braking from the body swept along the commanded path (pi/path_gate.py) - passing beside
# something does not stop the car, turning into it does. "cone": the older fixed cones + +-90 deg body alarm.
GATE_MODE = "path"
SCAN_LOST_S = 0.5          # path gate: no new LiDAR scan for this long -> no throttle
BODY_HARD_FLOOR_M = 0.30

# Active braking: below this, cutting throttle to 0 (coast) is not enough - actively brake
# with a reverse pulse. Speed-scaled (faster closing = harder pulse) and intent-aware (if the
# driver's own recent commands already show them easing off toward this obstacle, we trust
# them and brake more gently instead of stacking a hard jolt on top of what they're already
# doing - full intensity only kicks in if they're still committing to it).
BRAKE_ZONE_FRAC = 0.55      # fraction of required_margin() below which we brake instead of coast
BRAKE_PWM_MAX = 140         # cap on the reverse pulse magnitude
INTENT_WINDOW = 5           # recent commanded-magnitude samples used to judge driver intent
INTENT_EASE_SCALE = 0.5     # brake intensity multiplier when the driver is already easing off

# Inside required_margin used to mean a hard stop, full throttle cancel, no matter what was
# commanded. That's needlessly restrictive for a driver who's deliberately trying to nose up
# close to something (parking, squeezing past): a genuinely slow, low-power creep is not the
# same risk as gunning it toward an obstacle you're already close to. So: a LOW-pwm command
# is still allowed through inside the margin (down to an absolute floor), but a HIGH-pwm
# command in that same zone is cancelled outright rather than just capped - the driver asked
# for a fast approach to something already close, which is exactly the case this gate exists
# to refuse, not soften.
CREEP_PWM_MAX = 90          # commands at or below this magnitude may still creep inside the margin
CREEP_FLOOR_M = 0.05        # absolute minimum standoff - never creep closer than this regardless
                            # of commanded pwm (must stay above the sensor's own blind-zone floor)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drive_logs")
LOG_RATE_HZ = 10.0   # raw driving data, for later real-world intent-model training
                     # (see adas/intent.py for the eventual feature/label pipeline;
                     # this just captures ground truth cheaply for now)

WIRE_MOTOR_REVERSED = TUNING.servo.motor_reversed  # must match rc_controller.py's MOTOR_REVERSED


def find_ports(retries=5, delay=1.5):
    """Probe each /dev/ttyUSB* for the RPLIDAR reply. Retries: right after a restart the
    port can still be settling (e.g. from the previous process's shutdown) and briefly
    fail to answer, which looks identical to "not the LiDAR" unless we just try again."""
    for attempt in range(retries):
        lidar_port, esp_port = None, None
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
            except Exception as e:
                print(f"  {p}: could not probe ({e})")
        if lidar_port and esp_port:
            return lidar_port, esp_port
        print(f"  attempt {attempt + 1}/{retries}: LiDAR={lidar_port}, ESP32={esp_port}, retrying...")
        time.sleep(delay)
    return lidar_port, esp_port


class DirectionTrack:
    """One direction's (front or rear) distance + a smoothed closing-speed estimate,
    computed straight from how fast consecutive LiDAR readings shrink."""

    def __init__(self):
        self.dist = None
        self.speed = 0.0        # m/s, positive = closing in
        self._prev_dist = None
        self._prev_t = None

    def update(self, dist, t):
        if dist is not None and self._prev_dist is not None and self._prev_t is not None:
            dt = t - self._prev_t
            if 0.01 < dt < 0.5:
                raw = (self._prev_dist - dist) / dt
                raw = max(0.0, min(raw, MAX_CLOSING_SPEED_M_S))
                self.speed = 0.5 * self.speed + 0.5 * raw
        elif dist is None:
            self.speed = 0.0     # lost the reading: don't keep assuming the old closing speed
        self.dist = dist
        if dist is not None:
            self._prev_dist, self._prev_t = dist, t

    def required_margin(self):
        """Stopping distance needed at the current closing speed (bigger when approaching fast)."""
        v = self.speed
        return BASE_MARGIN_M + v * REACTION_TIME_S + (v * v) / (2.0 * ASSUMED_DECEL)


class IntentTracker:
    """Cheap, real-time proxy for driver intent - not the trained ML model in adas/intent.py
    (that needs more real driving data than exists yet to be trustworthy), just: is the
    driver's own recent commanded throttle toward this obstacle trending down (they're
    already easing off - trust them, brake gently) or flat/up (they're still committing to
    it - brake at full intensity)."""

    def __init__(self):
        self.front_hist = []
        self.rear_hist = []

    def update(self, physical_pwm):
        hist = self.front_hist if physical_pwm > 0 else self.rear_hist
        hist.append(abs(physical_pwm))
        del hist[:-INTENT_WINDOW]

    def easing_off(self, which):
        hist = self.front_hist if which == "front" else self.rear_hist
        if len(hist) < INTENT_WINDOW:
            return False
        half = INTENT_WINDOW // 2
        recent = sum(hist[-half:]) / half
        earlier = sum(hist[:half]) / half
        return recent < earlier * 0.85   # meaningfully lower, not just noise


class Clearance:
    """Background thread: keeps the latest min clearance + closing speed, ahead and behind."""

    def __init__(self, port):
        self.lidar = open_lidar(port)
        self.front_track = DirectionTrack()
        self.rear_track = DirectionTrack()
        self.body_min_front = None   # closest point in the WIDE front half (+-90deg)
        self.body_min_rear = None    # closest point in the WIDE rear half (+-90deg)
        self.points = []       # latest full scan, car-frame (angle_deg, dist_m) - for the GUI
        self.scan_seq = 0      # increments with every scan (the path gate remembers each scan once)
        self.tracker = Tracker()   # moving-object tracking (adas/tracking.py, LiDAR-only)
        self.tracks_info = []      # [{"id","x","y","moving"}] - for the GUI overlay
        self.raw_tracks = []       # the actual Track objects - FollowController needs these,
                                    # not the serialized dicts above
        self.moving_contact = (math.inf, math.inf, None)   # (dist_m, time_s, track_id)
        self._motion_v = 0.0       # ego speed estimate (signed, +forward), set by the main
        self._motion_offset = 0.0  # loop from the last commanded pwm/steer so the tracker
                                    # can subtract the car's own motion from what it sees
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def set_motion_state(self, physical_pwm, steer_offset_deg):
        """Called by the main loop after every command, so the background scan thread knows
        the car's own current speed/steering when separating a moving object's real motion
        from the apparent motion caused by the car itself turning or driving."""
        v = TUNING.speed_model.speed(physical_pwm)
        with self.lock:
            self._motion_v = v
            self._motion_offset = steer_offset_deg

    def start(self):
        print("LiDAR info:", self.lidar.get_info())
        print("LiDAR health:", self.lidar.get_health())
        self.thread.start()

    def _loop(self):
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=6000, min_len=5):
                if not self.running:
                    break
                if getattr(self, "log", None) is not None:
                    self.log.scan(scan)
                best_front = best_rear = None
                best_body_front = best_body_rear = None
                pts = []
                for _, angle, dist in scan:
                    if dist <= 0 or dist / 1000.0 < MIN_VALID_RANGE_M:
                        continue
                    a = (angle - FRONT_OFFSET_DEG) % 360
                    a = a if a <= 180 else a - 360   # -180..180, 0 = car's straight ahead
                    d_m = dist / 1000.0
                    pts.append((round(a, 1), round(d_m, 3)))

                    if abs(a) <= CONE_DEG:
                        d = d_m - FRONT_OVERHANG_M
                        if best_front is None or d < best_front:
                            best_front = d

                    ra = a - 180 if a > 0 else a + 180   # angle relative to straight behind
                    if abs(ra) <= CONE_DEG:
                        d = d_m - REAR_OVERHANG_M
                        if best_rear is None or d < best_rear:
                            best_rear = d

                    # wide corner-strike backstop, using the ACTUAL body overhang for this
                    # bearing (front/rear/side, via the car's real rectangular footprint - see
                    # body_overhang()'s docstring), split front-half/rear-half so something
                    # close on one side only ever blocks the direction that actually goes
                    # toward it, same as the narrow cones above - a single global "anything
                    # anywhere" flag used to block BOTH directions and fired 68% of a real
                    # drive session on a wall the car was simply parked next to on one side.
                    body_d = d_m - body_overhang(a)
                    if abs(a) <= WIDE_CONE_DEG:
                        if best_body_front is None or body_d < best_body_front:
                            best_body_front = body_d
                    else:
                        if best_body_rear is None or body_d < best_body_rear:
                            best_body_rear = body_d

                now = time.time()
                with self.lock:
                    v, offset = self._motion_v, self._motion_offset
                delta = delta_for_offset(offset)
                omega = v * math.tan(delta) / VP.wheelbase if abs(v) > 1e-6 else 0.0
                # vehicle-frame (x,y) for the tracker - same conversion as
                # path_predict.points_to_vehicle_frame, inlined to avoid re-parsing pts
                if pts:
                    arr = np.array(pts, dtype=float)
                    rad = np.radians(arr[:, 0])
                    xy = np.stack([arr[:, 1] * np.cos(rad) + VP.lidar_x,
                                   arr[:, 1] * np.sin(rad) + VP.lidar_y], axis=1)
                else:
                    xy = np.zeros((0, 2))
                tracks = self.tracker.update(xy, now, v, omega)
                direction = 1 if v >= 0 else -1
                contact = moving_object_contact(tracks, delta, direction, abs(v), VP, horizon=1.5)

                with self.lock:
                    self.front_track.update(best_front, now)
                    self.rear_track.update(best_rear, now)
                    self.body_min_front = best_body_front
                    self.body_min_rear = best_body_rear
                    self.points = pts
                    self.scan_seq += 1
                    self.moving_contact = contact
                    self.tracks_info = [{"id": tr.id, "x": round(tr.pos[0], 3),
                                          "y": round(tr.pos[1], 3), "moving": tr.moving}
                                         for tr in tracks]
                    self.raw_tracks = tracks
        except Exception as e:
            print("LiDAR thread stopped:", e)

    def read_points_seq(self):
        with self.lock:
            return list(self.points), self.scan_seq

    def read_points(self):
        with self.lock:
            return list(self.points)

    def read_tracks(self):
        with self.lock:
            return list(self.tracks_info), self.moving_contact

    def read_raw_tracks(self):
        with self.lock:
            return list(self.raw_tracks)

    def read(self):
        with self.lock:
            return self.front_track, self.rear_track, self.body_min_front, self.body_min_rear

    def stop(self):
        self.running = False
        try:
            self.lidar.stop()
            self.lidar.stop_motor()
            self.lidar.disconnect()
        except Exception:
            pass


class DriveLogger:
    """Cheap raw-data recorder: one row per tick, a fresh timestamped file per service run.
    No feature engineering here on purpose - keep the raw numbers, decide what to do with
    them later (see adas/intent.py for the real feature/label pipeline this feeds into
    once there's enough real driving data to be worth it)."""

    FIELDS = ["t", "steer_a1", "steer_a2", "pwm_commanded", "pwm_sent",
              "front_dist", "front_speed", "front_blocked",
              "rear_dist", "rear_speed", "rear_blocked",
              "body_min_front", "body_min_rear", "body_alert_front", "body_alert_rear",
              "braking", "adas_override", "n_tracks", "n_moving", "moving_blocked"]

    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, time.strftime("drive_%Y%m%d_%H%M%S.csv"))
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.path = path
        self.period = 1.0 / LOG_RATE_HZ
        self.next_t = 0.0
        print(f"logging raw drive data to {path}")

    def maybe_log(self, now, row):
        if now < self.next_t:
            return
        self.next_t = now + self.period
        self.writer.writerow(row)
        self.file.flush()   # small, infrequent writes: fine to flush every row on an SD card

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


class SafetyGate:
    """Turns direction tracks into block/allow decisions: blocked when the clearance is
    under the SPEED-SCALED required margin, held briefly if the reading is lost right
    after a block (see obstacle_stop.py for why)."""

    def __init__(self):
        self.last_blocked = {"front": None, "rear": None}
        self.last_body_alert = {"front": None, "rear": None}

    def blocked(self, which, track, now):
        if track.dist is not None and track.dist < track.required_margin():
            self.last_blocked[which] = now
            return True
        last = self.last_blocked[which]
        if track.dist is None and last is not None and now - last < HOLD_AFTER_LOST_READING_S:
            return True
        return False

    def body_alert(self, which, body_min, now):
        """Same hold-after-lost-reading protection as blocked(), applied to the wide
        backstop: an obstacle that gets close enough to fall below MIN_VALID_RANGE_M is
        filtered out of the scan entirely (self-hit/clutter rejection), which made
        body_min flip to None and the alert silently vanish exactly when the car was
        closest - confirmed live: a low-speed creep test drove straight into contact
        because of this. A lost reading right after being in-alert does NOT mean clear."""
        if body_min is not None and body_min < BODY_HARD_FLOOR_M:
            self.last_body_alert[which] = now
            return True
        last = self.last_body_alert[which]
        if body_min is None and last is not None and now - last < HOLD_AFTER_LOST_READING_S:
            return True
        return False


# --- live LiDAR GUI: a small HTTP server sharing the relay's already-open LiDAR connection,
# instead of a second process fighting it for the serial port. Read-only - it never sends
# motor commands, just visualizes what the safety gate is currently seeing/deciding.
GUI_PORT = 8090
GUI_STATE = {"lock": threading.Lock(), "data": {
    "front_dist": None, "front_speed": 0.0, "front_blocked": False,
    "rear_dist": None, "rear_speed": 0.0, "rear_blocked": False,
    "body_min_front": None, "body_min_rear": None,
    "body_alert_front": False, "body_alert_rear": False,
    "steer_a1": None, "steer_a2": None, "pwm_sent": 0,
    "mode": "manual", "braking": False, "tracks": [], "moving_blocked": False,
    "follow_enabled": False, "follow_lead": None,
}}


def gui_snapshot(front_track, rear_track, body_min_front, body_min_rear, front_blocked,
                  rear_blocked, body_alert_front, body_alert_rear,
                  steer_a1, steer_a2, pwm_sent, mode, braking=False, tracks=None,
                  moving_blocked=False, follow_enabled=False, follow_lead=None):
    """Full update, called whenever a real driver packet is processed - this is the
    authoritative blocked/braking decision, since that's the only time it actually matters
    for control."""
    with GUI_STATE["lock"]:
        GUI_STATE["data"] = {
            "t": time.time(),
            "front_dist": front_track.dist, "front_speed": round(front_track.speed, 3),
            "front_blocked": bool(front_blocked),
            "rear_dist": rear_track.dist, "rear_speed": round(rear_track.speed, 3),
            "rear_blocked": bool(rear_blocked),
            "body_min_front": body_min_front, "body_min_rear": body_min_rear,
            "body_alert_front": bool(body_alert_front), "body_alert_rear": bool(body_alert_rear),
            "steer_a1": steer_a1, "steer_a2": steer_a2, "pwm_sent": pwm_sent,
            "mode": mode, "braking": bool(braking),
            "tracks": tracks or [], "moving_blocked": bool(moving_blocked),
            "follow_enabled": bool(follow_enabled), "follow_lead": follow_lead,
            "assist": GUI_STATE["data"].get("assist"),        # kept: set separately by the assist code
            "gate": GUI_STATE["data"].get("gate"),
            "sim": GUI_STATE["data"].get("sim"),              # ground truth, only when run in the laptop simulator
            "plan": GUI_STATE["data"].get("plan"),
            "drive": GUI_STATE["data"].get("drive"),
        }


def gui_set_mode(mode):
    """Standalone mode update (e.g. an override toggle), independent of a driver packet -
    without this the GUI's mode badge would only ever update while someone is actively
    driving, silently showing stale MANUAL even after override was actually enabled."""
    with GUI_STATE["lock"]:
        GUI_STATE["data"]["mode"] = mode


def gui_set_assists(enabled):
    """Which driving assists are on, shown on the GUI's toggle buttons even while nobody is driving."""
    with GUI_STATE["lock"]:
        a = dict(GUI_STATE["data"].get("assist") or {"level": 0, "info": {}, "changed": False})
        a["enabled"] = enabled
        GUI_STATE["data"]["assist"] = a


def gui_update_distances(front_track, rear_track, body_min_front, body_min_rear):
    """Lighter, continuous update (no driver packet needed) so the GUI's numbers stay live
    even when nobody is actively driving - only touches distance/speed, never the
    blocked/braking/mode decision, since that's only meaningful while something is actually
    being commanded."""
    with GUI_STATE["lock"]:
        GUI_STATE["data"]["front_dist"] = front_track.dist
        GUI_STATE["data"]["front_speed"] = round(front_track.speed, 3)
        GUI_STATE["data"]["rear_dist"] = rear_track.dist
        GUI_STATE["data"]["rear_speed"] = round(rear_track.speed, 3)
        GUI_STATE["data"]["body_min_front"] = body_min_front
        GUI_STATE["data"]["body_min_rear"] = body_min_rear


EXTRA_POST = {}     # path -> callable; filled by tools/sim_car.py (simulator reset), empty on the car


def start_gui_server(clr):
    import http.server
    import json as _json

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass   # the default access log would spam relay.log on every poll

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self._serve_file("lidar_gui.html", "text/html")
            elif self.path.split("?")[0] in ("/dash", "/dash/"):
                self._serve_file(os.path.join("dash", "index.html"), "text/html")
            elif self.path.startswith("/vendor/") or self.path.startswith("/dash/"):
                rel = self.path.split("?")[0].lstrip("/")
                rel = rel[len("dash/"):] if rel.startswith("dash/") else rel
                if ".." in rel:
                    self.send_response(404); self.end_headers(); return
                ctype = "text/javascript" if rel.endswith(".js") else "text/css" if rel.endswith(".css") else "application/octet-stream"
                self._serve_file(os.path.join("dash", rel), ctype)
            elif self.path.startswith("/api/scan"):
                with GUI_STATE["lock"]:
                    payload = dict(GUI_STATE["data"])
                payload["points"] = clr.read_points()
                body = _json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path in EXTRA_POST:                       # e.g. the laptop simulator's reset button
                EXTRA_POST[self.path]()
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            routes = {"/api/override/on": b"ADAS_OVERRIDE_ON", "/api/override/off": b"ADAS_OVERRIDE_OFF",
                      "/api/follow/on": b"FOLLOW_ON", "/api/follow/off": b"FOLLOW_OFF"}
            cmd = routes.get(self.path)
            parts = self.path.strip("/").split("/")          # /api/assist/<name>/<on|off>
            if cmd is None and len(parts) == 4 and parts[:2] == ["api", "assist"] and parts[3] in ("on", "off"):
                cmd = f"ASSIST {parts[2]} {parts[3].upper()}".encode()
            if cmd is not None:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.sendto(cmd, ("127.0.0.1", UDP_PORT))
                    s.close()
                    body = b'{"ok": true}'
                    self.send_response(200)
                except Exception as e:
                    body = f'{{"ok": false, "error": "{e}"}}'.encode()
                    self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def _serve_file(self, name, content_type):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
            try:
                with open(path, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()

    server = http.server.ThreadingHTTPServer(("0.0.0.0", GUI_PORT), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"live LiDAR GUI on http://<pi-ip>:{GUI_PORT}/")
    return server


def main():
    # systemctl restart/stop sends SIGTERM; without this, cleanup only ran on Ctrl+C (SIGINT),
    # so the motor and LiDAR were never stopped cleanly on a service restart.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    print("looking for LiDAR and ESP32 on /dev/ttyUSB*...")
    lidar_port, esp_port = find_ports()
    print(f"LiDAR on {lidar_port}, ESP32 on {esp_port}")
    if not lidar_port or not esp_port:
        print("Could not identify both devices, aborting.")
        return

    esp = serial.Serial()
    esp.port = esp_port
    esp.baudrate = 115200
    esp.timeout = 0.2
    esp.dtr = False
    esp.rts = False
    esp.open()
    time.sleep(1.0)
    # full-rate recording of scans + every ESP32 line + driver input vs ADAS output (pi/drive_log.py)
    from pi.drive_log import DriveLog, LoggedSerial
    dlog = DriveLog("relay", tuning_path=TUNING_PATH)
    esp = LoggedSerial(esp, dlog)
    esp.write(b"A 90 90\nM 0\n")

    clr = Clearance(lidar_port)
    clr.log = dlog
    clr.start()
    time.sleep(2.0)

    gate = SafetyGate()
    intent = IntentTracker()
    adas_override = False
    from pi.relay_assists import RelayAssists
    from adas.assists import deadman_pwm
    assist = RelayAssists(TUNING, WIRE_MOTOR_REVERSED)   # driving assists, all off until toggled on
    from pi.path_gate import PathGate
    from adas.aeb import SpeedEstimator
    pgate = PathGate(assist.p, TUNING.speed_model)      # path-predicted emergency braking + obstacle memory
    vest = SpeedEstimator(TUNING.speed_model)           # speed from what was actually sent to the motor
    last_pkt_t = time.time()
    last_seq, last_seq_t = None, time.time()
    from pi.relay_assists import K_CURV_PER_SERVO_DEG as assist_k
    last_servo_cmd = assist.centre
    gate_delta = 0.0
    follow = FollowController(VP, TUNING.speed_model)   # adaptive cruise / follow-the-leader,
                                                          # opt-in via FOLLOW_ON/FOLLOW_OFF
    logger = DriveLogger(LOG_DIR)
    start_gui_server(clr)

    def _gui_idle_updater():
        while True:
            ft, rt, bmf, bmr = clr.read()
            gui_update_distances(ft, rt, bmf, bmr)
            time.sleep(0.15)
    threading.Thread(target=_gui_idle_updater, daemon=True).start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(0.05)     # short, so a lost driver link can be handled smoothly (dead-man ramp)
    print(f"\nlistening for driver commands on UDP :{UDP_PORT}")
    print("point rc_controller.py's ESP32_IP at this Pi's address to drive.\n")

    last_packet_time = time.time()
    last_status_print = 0.0
    last_steer_offset = 0.0   # persists across ticks that don't include an A line
    last_physical = 0.0       # what was last sent to the motor (physical convention, + = forward)
    last_tick = time.time()
    last_zero_sent = 0.0

    try:
        while True:
            try:
                data, addr = sock.recvfrom(256)
            except ConnectionResetError:
                continue      # Windows only (laptop simulator): a reply went to a sender that already closed
            except socket.timeout:
                # dead-man: the driver link is quiet. Keep the last command alive for 0.5 s (the ESP32's own
                # 500 ms timeout would otherwise cut the motor instantly), then ramp the throttle down smoothly.
                # Anything ahead in the direction of travel -> zero at once.
                now = time.time()
                dt, last_tick = now - last_tick, now
                age = now - last_packet_time
                if last_physical != 0:
                    ft, rt, bmf, bmr = clr.read()
                    gpts, gseq = clr.read_points_seq()
                    if gseq != last_seq:
                        last_seq, last_seq_t = gseq, now
                        pgate.on_scan(RelayAssists.points_vehicle_frame(gpts, assist.p.lidar_x), gseq)
                    ramp = deadman_pwm(last_physical, age, dt)
                    capped, _ = pgate.decide(dt, ramp, gate_delta, vest.v, ft.speed if ramp > 0 else rt.speed)
                    danger = abs(capped) < abs(ramp) - 0.5 or now - last_seq_t > SCAN_LOST_S
                    new = 0.0 if danger else capped
                    vest.update(dt, new)
                    if age > 0.5 and last_physical != 0 and new != last_physical:
                        dlog.event("driver link lost - dead-man ramp" if not danger else "driver link lost - obstacle, stopped",
                                   pwm=new)
                    last_physical = new
                    w = -int(round(new)) if WIRE_MOTOR_REVERSED else int(round(new))
                    esp.write(f"M {w}\n".encode())
                elif age > 0.6 and now - last_zero_sent > 0.5:
                    esp.write(b"M 0\n")
                    last_zero_sent = now
                continue

            last_packet_time = time.time()
            last_tick = last_packet_time
            text = data.decode(errors="ignore")

            parts = text.strip().split()
            if len(parts) == 3 and parts[0] == "ASSIST" and parts[2] in ("ON", "OFF"):
                assist.set(parts[1].lower(), parts[2] == "ON")
                print(f"\nassist {parts[1]} {parts[2]}: now {assist.enabled()}")
                dlog.event("assist toggled", name=parts[1], on=parts[2] == "ON")
                gui_set_assists(assist.enabled())
                sock.sendto(b"OK", addr)
                continue

            if text.strip() == "PING":
                esp.reset_input_buffer()
                esp.write(b"PING\n")
                time.sleep(0.05)
                reply = esp.read(esp.in_waiting or 1)
                sock.sendto(reply or b"PONG", addr)
                continue

            if text.strip() in ("ADAS_OVERRIDE_ON", "ADAS_OVERRIDE_OFF"):
                adas_override = text.strip() == "ADAS_OVERRIDE_ON"
                print(f"\nADAS override {'ENABLED (driver has full control)' if adas_override else 'disabled (safety gate active)'}")
                gui_set_mode("override" if adas_override else "manual")
                sock.sendto(b"OK", addr)
                continue

            if text.strip() in ("FOLLOW_ON", "FOLLOW_OFF"):
                follow.enabled = text.strip() == "FOLLOW_ON"
                print(f"\nFollow-the-leader {'ENABLED' if follow.enabled else 'disabled'}")
                sock.sendto(b"OK", addr)
                continue

            front_track, rear_track, body_min_front, body_min_rear = clr.read()
            tracks_info, moving_contact = clr.read_tracks()
            now = time.time()
            front_blocked = gate.blocked("front", front_track, now)
            rear_blocked = gate.blocked("rear", rear_track, now)
            body_alert_front = gate.body_alert("front", body_min_front, now)
            body_alert_rear = gate.body_alert("rear", body_min_rear, now)

            # driving assists rewrite the driver's steering/throttle first; the safety gate below still has
            # the last word on the throttle
            driver_text = text
            if not adas_override:
                _apts, _aseq = clr.read_points_seq()
                text = "\n".join(assist.process([ln.strip() for ln in text.splitlines() if ln.strip()], _apts, _aseq))
                if assist.assists.evading:
                    # the fixed straight-ahead cone would keep braking for an obstacle the car is steering
                    # around; the evasive planner has checked its own path (full body sweep + margin), so the
                    # cone stands down. The close-range body alert (body_alert_front) still stops the car.
                    front_blocked = False

            pkt_now = time.time()
            dt_pkt, last_pkt_t = min(0.2, max(0.005, pkt_now - last_pkt_t)), pkt_now
            gpts, gseq = clr.read_points_seq()
            if gseq != last_seq:
                last_seq, last_seq_t = gseq, pkt_now
                pgate.on_scan(RelayAssists.points_vehicle_frame(gpts, assist.p.lidar_x), gseq)
            servo_cmd = None
            for ln in text.splitlines():
                q = ln.split()
                if len(q) == 3 and q[0] == "A":
                    try:
                        servo_cmd = (float(q[1]) + float(q[2])) / 2.0
                    except ValueError:
                        pass
            if servo_cmd is not None:
                last_servo_cmd = servo_cmd
            gate_delta = math.atan(-assist_k * (last_servo_cmd - assist.centre) * assist.p.wheelbase)
            gate_info = {}

            out_lines = []
            steer_a1 = steer_a2 = None
            pwm_commanded = pwm_sent = 0
            braking = False
            moving_blocked = False
            final_physical = 0
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("M "):
                    try:
                        wire_val = int(float(line.split()[1]))
                    except (IndexError, ValueError):
                        out_lines.append(line)
                        continue
                    pwm_commanded = wire_val
                    physical = -wire_val if WIRE_MOTOR_REVERSED else wire_val
                    intent.update(physical)
                    braking = False
                    if GATE_MODE == "path":
                        closing = front_track.speed if physical > 0 else rear_track.speed
                        g_phys, g_brake = pgate.decide(dt_pkt, physical, gate_delta, vest.v, closing)
                        gate_info = dict(pgate.info)
                        if pkt_now - last_seq_t > SCAN_LOST_S and physical != 0:
                            g_phys, g_brake = 0, False
                            gate_info["action"] = "LiDAR lost"
                        if not adas_override:
                            physical, braking = g_phys, g_brake
                    elif adas_override:
                        pass   # driver has explicitly taken full control - pass through as-is
                    elif physical > 0 and (front_blocked or body_alert_front):
                        if body_alert_front and not front_blocked:
                            # wide backstop only - no closing-speed data here to size a brake
                            # pulse from, but the same creep-vs-cancel split still applies: a
                            # low-pwm nudge is allowed down to the absolute floor, a high-pwm
                            # command gets cancelled rather than an unconditional hard block
                            if body_min_front is None or body_min_front < CREEP_FLOOR_M:
                                physical = 0   # None means either genuinely clear OR too
                                               # close to see (filtered as self-hit) - the
                                               # hold-after-lost-reading above already covers
                                               # "still recently in-alert"; with no fresh
                                               # positive distance at all, never allow a creep
                            elif abs(physical) > CREEP_PWM_MAX:
                                physical = 0
                            # else: low pwm, still above the floor - let the creep through
                        elif front_track.dist is None:
                            physical = 0   # lost the reading while blocked - no fresh distance
                                           # to safely creep against
                        else:
                            req = front_track.required_margin()
                            deep = front_track.dist < req * BRAKE_ZONE_FRAC
                            high_pwm = abs(physical) > CREEP_PWM_MAX
                            if front_track.dist < CREEP_FLOOR_M or (deep and high_pwm):
                                speed_scale = min(1.0, front_track.speed / max(0.3, MAX_CLOSING_SPEED_M_S * 0.5))
                                intent_scale = INTENT_EASE_SCALE if intent.easing_off("front") else 1.0
                                physical = -int(BRAKE_PWM_MAX * speed_scale * intent_scale)
                                braking = True
                            elif high_pwm:
                                physical = 0   # committing hard while already inside the
                                               # margin - cancelled, not softened to a cap
                            # else: low pwm, still above the floor - let the creep through
                    elif physical < 0 and (rear_blocked or body_alert_rear):
                        if body_alert_rear and not rear_blocked:
                            if body_min_rear is None or body_min_rear < CREEP_FLOOR_M:
                                physical = 0
                            elif abs(physical) > CREEP_PWM_MAX:
                                physical = 0
                            # else: low pwm, still above the floor - let the creep through
                        elif rear_track.dist is None:
                            physical = 0
                        else:
                            req = rear_track.required_margin()
                            deep = rear_track.dist < req * BRAKE_ZONE_FRAC
                            high_pwm = abs(physical) > CREEP_PWM_MAX
                            if rear_track.dist < CREEP_FLOOR_M or (deep and high_pwm):
                                speed_scale = min(1.0, rear_track.speed / max(0.3, MAX_CLOSING_SPEED_M_S * 0.5))
                                intent_scale = INTENT_EASE_SCALE if intent.easing_off("rear") else 1.0
                                physical = int(BRAKE_PWM_MAX * speed_scale * intent_scale)
                                braking = True
                            elif high_pwm:
                                physical = 0
                            # else: low pwm, still above the floor - let the creep through

                    # Moving-object check (adas/tracking.py, ported from the simulator): the
                    # static front/rear/body checks above only see WHERE things are right
                    # now, not where a moving object is headed - something crossing in from
                    # outside the current cone can still meet the car on its predicted path.
                    # Genuinely new capability, not just a tighter version of the existing
                    # checks - it needs actual velocity data, which only the tracker has.
                    if not adas_override and physical != 0:
                        contact_dist, _, contact_id = moving_contact
                        if contact_id is not None:
                            est_v = TUNING.speed_model.speed(physical)
                            needed = BASE_MARGIN_M + abs(est_v) * REACTION_TIME_S + \
                                est_v * est_v / (2.0 * ASSUMED_DECEL)
                            if contact_dist < needed:
                                physical = 0
                                moving_blocked = True

                    # Adaptive cruise / follow-the-leader (adas/acc.py) - opt-in, driver
                    # toggles it on; caps forward throttle to hold a time-gap behind whatever
                    # the tracker sees moving ahead in the car's own lane, never raises it.
                    if not adas_override and physical > 0:
                        raw_tracks = clr.read_raw_tracks()
                        v_ego = TUNING.speed_model.speed(physical)
                        cap_v_pwm = follow.limit(raw_tracks, v_ego, physical)
                        if cap_v_pwm is not None:
                            physical = min(physical, int(cap_v_pwm))

                    wire_out = -physical if WIRE_MOTOR_REVERSED else physical
                    pwm_sent = wire_out
                    final_physical = physical
                    out_lines.append(f"M {wire_out}")
                elif line.startswith("A "):
                    parts = line.split()
                    if len(parts) == 3:
                        try:
                            steer_a1, steer_a2 = float(parts[1]), float(parts[2])
                            last_steer_offset = (steer_a1 + steer_a2) / 2.0 - 90.0
                        except ValueError:
                            pass
                    out_lines.append(line)   # steering is never modified by the safety gate
                else:
                    out_lines.append(line)

            clr.set_motion_state(final_physical, last_steer_offset)
            esp.write(("\n".join(out_lines) + "\n").encode())
            last_physical = float(final_physical)
            # GUI: where the body is heading at the driver's stick/throttle (X = first contact), plus any planned
            # manoeuvre and the original line it returns to
            phys_cmd = -pwm_commanded if WIRE_MOTOR_REVERSED else pwm_commanded
            plan = pgate.overlay(gate_delta, (phys_cmd > 0) - (phys_cmd < 0), vest.v,
                                 TUNING.speed_model.speed(phys_cmd)) if phys_cmd else None
            prev = assist.assists.preview() if assist.assists.evading else None
            if prev is not None:
                plan = plan or {}
                to_polar = lambda xy: [[round(float(-math.degrees(math.atan2(y, x - assist.p.lidar_x))), 1),
                                        round(float(math.hypot(x - assist.p.lidar_x, y)), 3)] for x, y in xy]
                plan["maneuver"], plan["line"] = to_polar(prev[0]), to_polar(prev[1])
            vest.update(dt_pkt, final_physical)
            with GUI_STATE["lock"]:
                GUI_STATE["data"]["assist"] = {"level": assist.level, "info": assist.info, "enabled": assist.enabled(),
                                               "changed": assist.changed}
                GUI_STATE["data"]["gate"] = gate_info
                GUI_STATE["data"]["plan"] = plan
                GUI_STATE["data"]["drive"] = {"v": round(vest.v, 3), "pwm_in": -pwm_commanded if WIRE_MOTOR_REVERSED else pwm_commanded,
                                              "pwm_out": -pwm_sent if WIRE_MOTOR_REVERSED else pwm_sent,
                                              "servo": last_servo_cmd, "centre": assist.centre, "t": time.time()}
            # what the driver asked for vs what the ADAS let through - the raw material for intent learning
            dlog.driver(inp=driver_text.strip(), assisted=text.strip() if assist.changed else None,
                        assist_level=assist.level, assist_info=assist.info or None,
                        gate=gate_info or None, v_est=round(vest.v, 3), servo_cmd=last_servo_cmd,
                        out=out_lines, pwm_in=pwm_commanded, pwm_out=pwm_sent,
                        front=front_track.dist, rear=rear_track.dist, fb=int(front_blocked), rb=int(rear_blocked),
                        braking=int(braking), moving_blocked=int(moving_blocked), override=int(adas_override),
                        follow=int(follow.enabled))

            logger.maybe_log(now, {
                "t": round(now, 3), "steer_a1": steer_a1, "steer_a2": steer_a2,
                "pwm_commanded": pwm_commanded, "pwm_sent": pwm_sent,
                "front_dist": front_track.dist, "front_speed": round(front_track.speed, 3),
                "front_blocked": int(front_blocked),
                "rear_dist": rear_track.dist, "rear_speed": round(rear_track.speed, 3),
                "rear_blocked": int(rear_blocked),
                "body_min_front": body_min_front, "body_min_rear": body_min_rear,
                "body_alert_front": int(body_alert_front), "body_alert_rear": int(body_alert_rear),
                "braking": int(braking), "adas_override": int(adas_override),
                "n_tracks": len(tracks_info), "n_moving": sum(1 for t in tracks_info if t["moving"]),
                "moving_blocked": int(moving_blocked),
            })
            gui_snapshot(front_track, rear_track, body_min_front, body_min_rear, front_blocked,
                         rear_blocked, body_alert_front, body_alert_rear, steer_a1, steer_a2,
                         pwm_sent, "override" if adas_override else "manual", braking,
                         tracks_info, moving_blocked, follow.enabled, follow.lead)

            if now - last_status_print > 0.5:
                last_status_print = now
                ftxt = f"{front_track.dist:.2f}m@{front_track.speed:.1f}m/s" if front_track.dist is not None else "--"
                rtxt = f"{rear_track.dist:.2f}m@{rear_track.speed:.1f}m/s" if rear_track.dist is not None else "--"
                btxt = f"{body_min_front:.2f}/{body_min_rear:.2f}m" if body_min_front is not None and body_min_rear is not None else "--"
                print(f"  front {ftxt:>16s} {'[BLOCKED]' if front_blocked else '         '}   "
                      f"rear {rtxt:>16s} {'[BLOCKED]' if rear_blocked else '         '}   "
                      f"body {btxt:>11s} {'[ALERT]' if (body_alert_front or body_alert_rear) else '       '}   "
                      f"-> {' '.join(out_lines)}", end="\r")

    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        esp.write(b"M 0\nSTOP\n")
        time.sleep(0.1)
        esp.close()
        clr.stop()
        sock.close()
        logger.close()
        dlog.close()
        print("motor stopped, LiDAR stopped, exiting.")


if __name__ == "__main__":
    main()
