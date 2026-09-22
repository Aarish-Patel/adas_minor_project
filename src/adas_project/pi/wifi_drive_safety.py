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
import os
import signal
import socket
import sys
import threading
import time

import serial
from rplidar import RPLidar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adas.config import load_tuning  # noqa: E402

TUNING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning_real_car.json")
TUNING = load_tuning(TUNING_PATH)

UDP_PORT = 4210

FRONT_OFFSET_DEG = TUNING.mount.yaw_offset_deg       # raw LiDAR angle that is the car's straight-ahead
CONE_DEG = 25                                        # +/- around straight ahead / straight behind
                                                     # (widened from 15: gives the closing-speed
                                                     # check more lead time on something crossing
                                                     # in from the side before it's dead ahead)
FRONT_OVERHANG_M = TUNING.mount.front_overhang_m     # LiDAR to front bumper
REAR_OVERHANG_M = TUNING.mount.rear_overhang_m       # LiDAR to rear bumper
MIN_VALID_RANGE_M = TUNING.mount.min_valid_range_m   # ignore raw readings closer than this
                                                     # (self-hits: wires/mount clutter)
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

# The front/rear cones above only cover straight-ahead/straight-behind motion. They are
# blind to something close off to the SIDE that a turning front (or rear) corner can swing
# into - confirmed live: the car's front-right corner clipped an obstacle that never showed
# up in either cone. This is a static, direction-independent backstop: if anything, anywhere
# around the car, gets this close, throttle is blocked in BOTH directions (we don't model the
# exact footprint sweep for the current steering angle, so we can't tell which way is safe -
# see STATUS.md's "no steering-aware curved-path prediction" gap for the real fix).
BODY_HARD_FLOOR_M = 0.30

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


class Clearance:
    """Background thread: keeps the latest min clearance + closing speed, ahead and behind."""

    def __init__(self, port):
        self.lidar = RPLidar(port, baudrate=256000, timeout=3)
        self.front_track = DirectionTrack()
        self.rear_track = DirectionTrack()
        self.body_min = None   # closest point anywhere around the car (360deg), body-adjusted
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        print("LiDAR info:", self.lidar.get_info())
        print("LiDAR health:", self.lidar.get_health())
        self.thread.start()

    def _loop(self):
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=6000, min_len=5):
                if not self.running:
                    break
                best_front = best_rear = best_body = None
                for _, angle, dist in scan:
                    if dist <= 0 or dist / 1000.0 < MIN_VALID_RANGE_M:
                        continue
                    a = (angle - FRONT_OFFSET_DEG) % 360
                    a = a if a <= 180 else a - 360   # -180..180, 0 = car's straight ahead
                    d_m = dist / 1000.0

                    if abs(a) <= CONE_DEG:
                        d = d_m - FRONT_OVERHANG_M
                        if best_front is None or d < best_front:
                            best_front = d

                    ra = a - 180 if a > 0 else a + 180   # angle relative to straight behind
                    if abs(ra) <= CONE_DEG:
                        d = d_m - REAR_OVERHANG_M
                        if best_rear is None or d < best_rear:
                            best_rear = d

                    # full-360, direction-independent: use the smaller overhang so this stays
                    # conservative (an underestimate of true clearance) at any bearing
                    body_d = d_m - min(FRONT_OVERHANG_M, REAR_OVERHANG_M)
                    if best_body is None or body_d < best_body:
                        best_body = body_d

                now = time.time()
                with self.lock:
                    self.front_track.update(best_front, now)
                    self.rear_track.update(best_rear, now)
                    self.body_min = best_body
        except Exception as e:
            print("LiDAR thread stopped:", e)

    def read(self):
        with self.lock:
            return self.front_track, self.rear_track, self.body_min

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
              "body_min", "body_alert"]

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

    def blocked(self, which, track, now):
        if track.dist is not None and track.dist < track.required_margin():
            self.last_blocked[which] = now
            return True
        last = self.last_blocked[which]
        if track.dist is None and last is not None and now - last < HOLD_AFTER_LOST_READING_S:
            return True
        return False


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
    esp.write(b"A 90 90\nM 0\n")

    clr = Clearance(lidar_port)
    clr.start()
    time.sleep(2.0)

    gate = SafetyGate()
    logger = DriveLogger(LOG_DIR)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(0.5)
    print(f"\nlistening for driver commands on UDP :{UDP_PORT}")
    print("point rc_controller.py's ESP32_IP at this Pi's address to drive.\n")

    last_packet_time = time.time()
    last_status_print = 0.0

    try:
        while True:
            try:
                data, addr = sock.recvfrom(256)
            except socket.timeout:
                # failsafe: if the driver link drops, stop the car (belt-and-suspenders;
                # the ESP32 firmware also has its own 500 ms timeout).
                if time.time() - last_packet_time > 0.6:
                    esp.write(b"M 0\n")
                continue

            last_packet_time = time.time()
            text = data.decode(errors="ignore")

            if text.strip() == "PING":
                esp.reset_input_buffer()
                esp.write(b"PING\n")
                time.sleep(0.05)
                reply = esp.read(esp.in_waiting or 1)
                sock.sendto(reply or b"PONG", addr)
                continue

            front_track, rear_track, body_min = clr.read()
            now = time.time()
            front_blocked = gate.blocked("front", front_track, now)
            rear_blocked = gate.blocked("rear", rear_track, now)
            body_alert = body_min is not None and body_min < BODY_HARD_FLOOR_M

            out_lines = []
            steer_a1 = steer_a2 = None
            pwm_commanded = pwm_sent = 0
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
                    if (physical > 0 and (front_blocked or body_alert)) or \
                       (physical < 0 and (rear_blocked or body_alert)):
                        physical = 0
                    wire_out = -physical if WIRE_MOTOR_REVERSED else physical
                    pwm_sent = wire_out
                    out_lines.append(f"M {wire_out}")
                elif line.startswith("A "):
                    parts = line.split()
                    if len(parts) == 3:
                        try:
                            steer_a1, steer_a2 = float(parts[1]), float(parts[2])
                        except ValueError:
                            pass
                    out_lines.append(line)   # steering is never modified by the safety gate
                else:
                    out_lines.append(line)

            esp.write(("\n".join(out_lines) + "\n").encode())

            logger.maybe_log(now, {
                "t": round(now, 3), "steer_a1": steer_a1, "steer_a2": steer_a2,
                "pwm_commanded": pwm_commanded, "pwm_sent": pwm_sent,
                "front_dist": front_track.dist, "front_speed": round(front_track.speed, 3),
                "front_blocked": int(front_blocked),
                "rear_dist": rear_track.dist, "rear_speed": round(rear_track.speed, 3),
                "rear_blocked": int(rear_blocked),
                "body_min": body_min, "body_alert": int(body_alert),
            })

            if now - last_status_print > 0.5:
                last_status_print = now
                ftxt = f"{front_track.dist:.2f}m@{front_track.speed:.1f}m/s" if front_track.dist is not None else "--"
                rtxt = f"{rear_track.dist:.2f}m@{rear_track.speed:.1f}m/s" if rear_track.dist is not None else "--"
                btxt = f"{body_min:.2f}m" if body_min is not None else "--"
                print(f"  front {ftxt:>16s} {'[BLOCKED]' if front_blocked else '         '}   "
                      f"rear {rtxt:>16s} {'[BLOCKED]' if rear_blocked else '         '}   "
                      f"body {btxt:>6s} {'[ALERT]' if body_alert else '       '}   "
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
        print("motor stopped, LiDAR stopped, exiting.")


if __name__ == "__main__":
    main()
