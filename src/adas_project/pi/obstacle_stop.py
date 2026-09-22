"""Minimal live demo: creep forward, stop automatically if something is close ahead.

Auto-detects which /dev/ttyUSB* is the LiDAR (responds to RPLIDAR GET_INFO) and which
is the ESP32 (everything else), so it doesn't matter which port each enumerates on.

Safety: motor PWM is capped low (CREEP_PWM), steering stays centered, and the loop
stops the motor immediately if the closest point within +/-30 deg of straight ahead
is nearer than STOP_DISTANCE_M. Ctrl+C always sends M 0 + STOP before exiting.
"""

import glob
import math
import sys
import threading
import time

import serial
from rplidar import RPLidar

STOP_DISTANCE_M = 0.10    # clearance to keep in front of the BUMPER (not the sensor)
                          # NOTE: must stay above (MIN_VALID_RANGE_M - FRONT_OVERHANG_M) = 0.04 m,
                          # or the car goes blind to the obstacle before it stops and drives into it.
FRONT_OVERHANG_M = 0.16   # LiDAR sits this far behind the front bumper (measured)
CREEP_PWM = 90            # low, just above the motor's dead-band
MOTOR_REVERSED = True     # measured: +PWM drove this car BACKWARD, not forward
FRONT_OFFSET_DEG = 96.5   # raw LiDAR angle that is actually the car's straight-ahead (measured)
FORWARD_CONE_DEG = 15     # +/- around straight ahead
MIN_VALID_RANGE_M = 0.20  # ignore raw readings closer than this (self-hits/clutter near the sensor)
RUN_SECONDS = 25


def find_ports():
    candidates = sorted(glob.glob("/dev/ttyUSB*"))
    lidar_port, esp_port = None, None
    for p in candidates:
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
    return lidar_port, esp_port


class ForwardDistance:
    """Background thread: keeps the latest min distance within the forward cone."""

    def __init__(self, port):
        self.lidar = RPLidar(port, baudrate=256000, timeout=3)
        self.value = None
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
                best = None
                for _, angle, dist in scan:
                    if dist <= 0 or dist / 1000.0 < MIN_VALID_RANGE_M:
                        continue
                    a = (angle - FRONT_OFFSET_DEG) % 360
                    a = a if a <= 180 else a - 360   # -180..180, 0 = car's straight ahead
                    if abs(a) <= FORWARD_CONE_DEG:
                        d = dist / 1000.0 - FRONT_OVERHANG_M     # distance from the BUMPER, not the sensor
                        if best is None or d < best:
                            best = d
                with self.lock:
                    self.value = best
        except Exception as e:
            print("LiDAR thread stopped:", e)

    def read(self):
        with self.lock:
            return self.value

    def stop(self):
        self.running = False
        try:
            self.lidar.stop()
            self.lidar.stop_motor()
            self.lidar.disconnect()
        except Exception:
            pass


def esp_send(ser, line):
    ser.write((line + "\n").encode())


def motor_cmd(pwm):
    """pwm > 0 always means physically FORWARD, regardless of wiring polarity."""
    sent = -pwm if MOTOR_REVERSED else pwm
    return f"M {int(sent)}"


def main():
    print("looking for LiDAR and ESP32 on /dev/ttyUSB*...")
    lidar_port, esp_port = find_ports()
    print(f"LiDAR on {lidar_port}, ESP32 on {esp_port}")
    if not lidar_port or not esp_port:
        print("Could not identify both devices, aborting.")
        sys.exit(1)

    esp = serial.Serial()
    esp.port = esp_port
    esp.baudrate = 115200
    esp.timeout = 0.2
    esp.dtr = False
    esp.rts = False
    esp.open()
    time.sleep(1.0)
    esp_send(esp, "A 90 90")   # centre steering
    esp_send(esp, motor_cmd(0))

    dist = ForwardDistance(lidar_port)
    dist.start()
    time.sleep(2.5)   # let the motor spin up and a few scans accumulate

    print(f"\nrunning for {RUN_SECONDS}s: creep at PWM {CREEP_PWM}, stop under {STOP_DISTANCE_M} m ahead")
    print("Ctrl+C to stop early\n")

    t0 = time.time()
    last_state = None
    last_blocked_at = None
    HOLD_AFTER_LOST_READING_S = 1.0   # a lost reading right after a stop does NOT mean "clear"
    try:
        while time.time() - t0 < RUN_SECONDS:
            d = dist.read()
            now = time.time()
            if d is not None and d < STOP_DISTANCE_M:
                blocked = True
                last_blocked_at = now
            elif d is None and last_blocked_at is not None and now - last_blocked_at < HOLD_AFTER_LOST_READING_S:
                blocked = True   # no reading right after being blocked: stay stopped, don't guess "clear"
            else:
                blocked = False
            pwm = 0 if blocked else CREEP_PWM
            esp_send(esp, motor_cmd(pwm))

            state = "STOP (obstacle)" if blocked else "creep"
            if state != last_state or True:
                dtxt = f"{d:.2f} m (bumper)" if d is not None else "no reading"
                print(f"  t={time.time() - t0:5.1f}s  forward clear: {dtxt:>16s}  ->  {state}")
                last_state = state
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        esp_send(esp, motor_cmd(0))
        esp_send(esp, "STOP")
        time.sleep(0.1)
        esp.close()
        dist.stop()
        print("motor stopped, LiDAR stopped, exiting.")


if __name__ == "__main__":
    main()
