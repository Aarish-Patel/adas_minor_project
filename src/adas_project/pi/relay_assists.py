"""Runs the driving assists (adas/assists.py) inside the relay, on the driver's live controller packets.

The relay receives "A <servo> <servo>" and "M <wire pwm>" lines from rc_controller.py. Before its own safety
gate runs, this rewrites them with the assists' output:
  servo angle  <->  path curvature  (fitted from the car's drive logs: 0.0656 rad/m per servo degree,
                                     servo above the straight-ahead angle = turning right)
  wire PWM     <->  physical PWM    (motor_reversed: negative wire = forward)
The relay's existing emergency braking still runs afterwards and always has the last word.
Toggle each assist with UDP "ASSIST <name> ON|OFF" or the GUI buttons.
"""
import math
import time

import numpy as np

from adas.aeb import SpeedEstimator
from adas.assists import DrivingAssists
from adas.vehicle_params import VehicleParams, delta_to_steer, steer_to_delta

K_CURV_PER_SERVO_DEG = 0.0656      # fitted from the logging drive (sim/fitted_car.json)
SERVO_TRAVEL = (57.0, 53.0)        # degrees right / left of centre the servo can move


def car_params(mount, wheelbase=0.20, lidar_x=0.12):
    """The real body, from the ruler measurements taken at the LiDAR (tuning mount section)."""
    ratio = math.degrees(math.atan(K_CURV_PER_SERVO_DEG * wheelbase))
    return VehicleParams(wheelbase=wheelbase, pivot_track=0.07,
                         width=mount.left_overhang_m + mount.right_overhang_m,
                         front_overhang=mount.front_overhang_m - (wheelbase - lidar_x),
                         rear_overhang=mount.rear_overhang_m - lidar_x, lidar_x=lidar_x, lidar_y=0.0,
                         max_inner_left_deg=ratio * SERVO_TRAVEL[1], max_inner_right_deg=ratio * SERVO_TRAVEL[0])


class RelayAssists:
    NAMES = ("evasive", "centring", "limiter", "narrow", "proximity")

    def __init__(self, tuning, motor_reversed=True):
        self.p = car_params(tuning.mount)
        self.centre = float(tuning.servo.left_center)
        self.model = tuning.speed_model
        self.assists = DrivingAssists(self.p, self.model)
        self.est = SpeedEstimator(self.model)
        self.reversed = motor_reversed
        self.last_t = None
        self.last_servo = self.centre
        self.info, self.level, self.changed = {}, 0, False

    def enabled(self):
        return {k: bool(v) for k, v in self.assists.enabled.items()}

    def set(self, name, on):
        if name == "all":
            for k in self.NAMES:
                self.assists.enabled[k] = on
        elif name in self.NAMES:
            self.assists.enabled[name] = on

    # --- unit conversions
    def servo_to_stick(self, servo):
        kappa_left = -K_CURV_PER_SERVO_DEG * (servo - self.centre)
        return delta_to_steer(math.atan(kappa_left * self.p.wheelbase), self.p)

    def stick_to_servo(self, stick):
        kappa_left = math.tan(steer_to_delta(stick, self.p)) / self.p.wheelbase
        return self.centre - kappa_left / K_CURV_PER_SERVO_DEG

    @staticmethod
    def points_vehicle_frame(points, lidar_x=0.12):
        """Relay points (car angle deg, clockwise-positive = right; distance from the LiDAR) -> vehicle frame."""
        if not points:
            return np.empty((0, 2))
        a = np.radians(np.array([p[0] for p in points]))
        d = np.array([p[1] for p in points])
        return np.column_stack([lidar_x + d * np.cos(a), -d * np.sin(a)])

    def process(self, lines, points):
        """lines: the driver's packet lines. Returns the lines to hand on to the safety gate."""
        now = time.time()
        dt = 0.05 if self.last_t is None else min(0.2, max(0.005, now - self.last_t))
        self.last_t = now
        servo, wire, i_a, i_m = None, None, None, None
        for i, ln in enumerate(lines):
            p = ln.split()
            try:
                if p[0] == "A" and len(p) == 3:
                    servo, i_a = (float(p[1]) + float(p[2])) / 2.0, i
                elif p[0] == "M" and len(p) == 2:
                    wire, i_m = float(p[1]), i
            except (ValueError, IndexError):
                pass
        if servo is None:
            servo = self.last_servo
        self.last_servo = servo
        physical = 0.0 if wire is None else (-wire if self.reversed else wire)
        stick = self.servo_to_stick(servo)
        self.changed = False
        if not any(self.assists.enabled.values()):
            self.est.update(dt, physical)
            self.info, self.level = {}, 0
            return lines
        pts = self.points_vehicle_frame(points, self.p.lidar_x)
        s_out, p_out, self.level = self.assists.update(dt, pts, stick, physical, self.est.v)
        self.info = dict(self.assists.info)
        out = list(lines)
        if abs(s_out - stick) > 1e-3:
            sv = int(round(max(35.0, min(145.0, self.stick_to_servo(s_out)))))
            line = f"A {sv} {sv}"
            if i_a is None:
                out.append(line)
            else:
                out[i_a] = line
            self.changed = True
        if wire is not None and abs(p_out - physical) > 0.5:
            w = -int(round(p_out)) if self.reversed else int(round(p_out))
            out[i_m] = f"M {w}"
            self.changed = True
            physical = p_out
        self.est.update(dt, physical)
        return out
