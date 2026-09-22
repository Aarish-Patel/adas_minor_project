"""Calibration procedures: turn a few timed test runs on the car into ADAS settings.

    python -m pi.calibrate --sim      REHEARSAL on the simulator, with hidden "true" values to recover
    python -m pi.calibrate --real     on the car (drive at a wall with clear floor; LiDAR + ESP32 required)

What it measures and why:
    speed model   speed vs PWM  -> SpeedModel.v_max and dead-band. The ADAS has no encoder, so this
                  table is how it knows how fast the car is going.
    braking       deceleration when the motor is cut / reversed -> AEBConfig.decel (with a safety factor)
    latency       time from a command to the car starting to move -> AEBConfig.latency

Both the rehearsal and the real run use the same procedure code; only the Platform differs.
"""

import argparse
import glob
import json
import math
import os
import sys
import threading
import time

import numpy as np


class Platform:
    """What a calibration needs from the car: drive it, watch the distance to a wall, and time."""

    def send_pwm(self, pwm): ...
    def range_ahead(self): ...           # metres from the front bumper to the wall
    def wait(self, seconds): ...
    def rewind(self): ...                # get the car back to the start line (manual on the real car)


def _speed_over(platform, seconds, samples=8):
    """Average speed toward the wall over `seconds` from distance readings."""
    times, dists = [], []
    step = seconds / samples
    for _ in range(samples + 1):
        times.append(platform.now())
        dists.append(platform.range_ahead())
        platform.wait(step)
    slope = np.polyfit(times, dists, 1)[0]
    return -slope


def calibrate_speed(platform, levels=(80, 100, 120, 140, 160, 180, 200, 220, 240, 255), settle=0.8, measure=0.8):
    data = []
    for pwm in levels:
        platform.rewind()
        platform.send_pwm(pwm)
        platform.wait(settle)
        if platform.range_ahead() < 0.7:
            platform.send_pwm(0)
            continue
        v = _speed_over(platform, measure)
        platform.send_pwm(0)
        platform.wait(1.0)
        data.append((pwm, max(0.0, v)))
    moving = [(p, v) for p, v in data if v > 0.04]
    if len(moving) < 3:
        raise RuntimeError("the car barely moved: check the motor and clear floor ahead")
    p = np.array([m[0] for m in moving], dtype=float)
    v = np.array([m[1] for m in moving])
    slope, intercept = np.polyfit(p, v, 1)
    deadband = -intercept / slope
    v_max = slope * (255.0 - deadband)
    return {"v_max": float(v_max), "deadband": float(max(0.0, deadband))}, data


def calibrate_braking(platform, latency, pwm=200):
    """Deceleration from the stopping distance when the motor is simply cut (coasting).

    Coasting is the weakest way this car can stop, so planning with it is conservative;
    active reverse braking then only adds margin.
    """
    platform.rewind()
    platform.send_pwm(pwm)
    platform.wait(1.0)
    v0 = _speed_over(platform, 0.4, 8)
    r0 = platform.range_ahead()
    platform.send_pwm(0)
    platform.wait(3.0)
    rolled = r0 - platform.range_ahead()
    # rolled includes the reaction delay, so v0^2/(2*rolled) under-estimates the real deceleration:
    # deliberately conservative (the delay is added again through AEBConfig.latency).
    decel_distance = max(rolled, 0.03)
    return {"coast": v0 * v0 / (2.0 * decel_distance), "v0": v0, "rolled": rolled}


def calibrate_latency(platform, pwm=220, threshold=0.025, confirm=4):
    """Time from the command until the car has clearly moved (several readings past the threshold)."""
    platform.rewind()
    platform.send_pwm(0)
    platform.wait(0.6)
    d0 = np.mean([platform.range_ahead() for _ in range(5)])
    t0 = platform.now()
    platform.send_pwm(pwm)
    run, first = 0, None
    while platform.now() - t0 < 1.5:
        platform.wait(0.01)
        if d0 - platform.range_ahead() > threshold:
            run += 1
            first = first if first is not None else platform.now()
            if run >= confirm:
                break
        else:
            run, first = 0, None
    latency = (first if first is not None else platform.now()) - t0
    platform.send_pwm(0)
    platform.wait(1.0)
    return latency


def run_calibration(platform):
    speed, table = calibrate_speed(platform)
    latency = calibrate_latency(platform)
    braking = calibrate_braking(platform, latency)
    result = {
        "speed_model": {"v_max": round(speed["v_max"], 3), "deadband": round(speed["deadband"], 1)},
        "aeb": {"decel": round(float(braking["coast"]) * 0.9, 2),                    # 10 % safety factor
                "latency": round(float(latency) + 0.10, 3)},                          # + one scan period
        "measured": {"decel_coast": round(float(braking["coast"]), 2), "response_delay_s": round(float(latency), 3)},
        "speed_table": [(int(p), round(v, 3)) for p, v in table],
    }
    return result


# ---------------------------------------------------------------- simulator rehearsal
class SimPlatform(Platform):
    """The simulated car with values the procedure does not know."""

    def __init__(self, v_max=1.17, deadband=52, tau=0.18, brake=2.2, coast=1.1, link_delay=0.05):
        from adas.aeb import SpeedModel
        from sim.car_sim import Dynamics
        from sim.simulator import Simulator
        from sim.world import Box, World
        self.truth = {"v_max": v_max, "deadband": deadband, "brake": brake, "coast": coast}
        w = World()
        w.add(Box(9.0, 0.0, 0.06, 2.0, 0.0, 0.2))
        dyn = Dynamics(speed_model=SpeedModel(v_max=v_max, deadband=deadband), tau_motor=tau, brake_max=brake,
                       coast_decel=coast)
        self.sim = Simulator(w, (0.0, 0.0, 0.0), adas_on="off", dynamics=dyn, seed=3)
        self.sim.link_delay = link_delay
        self.pwm = 0.0

    def now(self):
        return self.sim.t

    def send_pwm(self, pwm):
        self.pwm = pwm

    def range_ahead(self):
        # what a LiDAR would report: distance to the wall from the bumper, with a little noise
        c = self.sim.car
        return 9.0 - 0.03 - (c.x + self.sim.p.front_x) + np.random.normal(0, 0.004)

    def wait(self, seconds):
        end = self.sim.t + seconds
        while self.sim.t < end:
            self.sim.step(0.0, self.pwm)

    def rewind(self):
        self.sim.car.x, self.sim.car.y, self.sim.car.theta, self.sim.car.v = 0.0, 0.0, 0.0, 0.0
        self.sim.queue.clear()


class RealPlatform(Platform):
    """The real car: ESP32 over serial for motor commands, RPLIDAR forward sector for range.

    SAFETY: this drives the motor repeatedly and automatically once started (that's the
    whole point of the procedure - it needs many timed runs at different PWM levels). Only
    run this with a clear, open floor ahead and someone watching. `rewind()` is a manual
    step: it prints a prompt and waits for Enter, since nothing can push the car back
    for you.
    """

    def __init__(self, wall_hint_m=3.0):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from adas.config import load_tuning
        from rplidar import RPLidar
        import serial

        tuning_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning_real_car.json")
        self.mount = load_tuning(tuning_path).mount
        self.wall_hint_m = wall_hint_m   # used only to sanity-check we're seeing the wall, not noise

        lidar_port, esp_port = self._find_ports()
        if not lidar_port or not esp_port:
            raise RuntimeError(f"could not identify both devices (LiDAR={lidar_port}, ESP32={esp_port})")

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
        self._front = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
        time.sleep(2.0)   # let the motor spin up and a few scans accumulate
        self._t0 = time.time()

    @staticmethod
    def _find_ports(retries=5, delay=1.5):
        import serial
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
                    (lidar_port := p) if resp[:2] == bytes([0xA5, 0x5A]) else (esp_port := p)
                except Exception:
                    pass
            if lidar_port and esp_port:
                return lidar_port, esp_port
            time.sleep(delay)
        return lidar_port, esp_port

    def _scan_loop(self):
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=6000, min_len=5):
                if not self._running:
                    break
                best = None
                for _, angle, dist in scan:
                    if dist <= 0 or dist / 1000.0 < self.mount.min_valid_range_m:
                        continue
                    a = self.mount.to_car_angle(angle)
                    if abs(a) <= 15:
                        d = dist / 1000.0 - self.mount.front_overhang_m
                        if best is None or d < best:
                            best = d
                with self._lock:
                    self._front = best
        except Exception as e:
            print("LiDAR thread stopped:", e)

    def now(self):
        return time.time() - self._t0

    def send_pwm(self, pwm):
        # Raw ESP32 M is reversed on this car (confirmed live: negative wire value drives
        # forward) - same convention as WIRE_MOTOR_REVERSED in wifi_drive_safety.py and
        # MOTOR_REVERSED in rc_controller.py. `pwm` here is always the physical-forward
        # magnitude the calibration procedure wants, so flip it before writing to the ESP32.
        self.esp.write(f"M {-int(pwm)}\n".encode())

    def range_ahead(self):
        with self._lock:
            return self._front if self._front is not None else 0.0

    def wait(self, seconds):
        time.sleep(seconds)

    def rewind(self):
        self.esp.write(b"M 0\n")
        input("  >> move the car back to the start line, keep the path clear, then press Enter... ")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="rehearse on the simulator")
    ap.add_argument("--real", action="store_true", help="run on the car")
    args = ap.parse_args()
    if args.real:
        print("REAL CALIBRATION: the car will drive itself repeatedly at a wall, at increasing")
        print("speed, and coast/reverse to measure braking. Make sure the floor ahead is clear")
        print("and open (several metres), and stay ready to cut power if anything looks wrong.")
        if input("Type 'yes' to continue: ").strip().lower() != "yes":
            print("aborted")
            return
        plat = RealPlatform()
        try:
            result = run_calibration(plat)
        finally:
            plat.close()
        print("measured speed model:", result["speed_model"])
        print("measured braking:", result["measured"])
        print("suggested AEB settings:", result["aeb"])
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning_suggested.json")
        with open(out_path, "w") as f:
            json.dump({k: result[k] for k in ("speed_model", "aeb")}, f, indent=2)
        print(f"wrote {out_path} - merge these numbers into tuning_real_car.json by hand once you've")
        print("sanity-checked them (they overwrite v_max/deadband/decel/latency only).")
        return

    np.random.seed(0)
    plat = SimPlatform()
    result = run_calibration(plat)
    truth = plat.truth
    print("recovered speed model:", result["speed_model"], " (simulator's hidden truth:",
          {"v_max": truth["v_max"], "deadband": truth["deadband"]}, ")")
    print("measured braking:", result["measured"], " (hidden truth: coast", truth["coast"], ", link delay 0.05 s + motor lag)")
    print("suggested AEB settings:", result["aeb"])
    with open("tuning_suggested.json", "w") as f:
        json.dump({k: result[k] for k in ("speed_model", "aeb")}, f, indent=2)
    print("wrote tuning_suggested.json")


if __name__ == "__main__":
    main()
