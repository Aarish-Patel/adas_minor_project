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
from pi.scanmatch import Odometry, polar_to_xy

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


def apply_car_model(tuning, path=None):
    """Use the car model fitted from drive logs (pi/car_model.json, from sim/log_fit.py) for the speed model.
    Returns a short description, or None if there is no usable file (the tuning file's model is kept)."""
    import json
    import os
    from adas.aeb import SpeedModel
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "car_model.json")
    if not os.path.exists(path):
        return None
    try:
        cm = json.load(open(path))
        tuning.speed_model = SpeedModel(v_max=cm["speed_model"]["v_max"], deadband=cm["speed_model"]["deadband"])
        return f"v_max {tuning.speed_model.v_max} m/s, dead-band {tuning.speed_model.deadband} PWM"
    except Exception as e:                 # a broken file must never stop the relay
        return f"ignored ({e})"


def driver_intent(servo_hist, dt, centre, activity_deg=3.0, attention_s=1.0, window=4):
    """Driver state from the stick (driver-state-adaptive ADAS). None = no stick activity for `attention_s`
    (a lapse: nothing says the driver is handling it). Otherwise the driver is attentive and this returns the rate at
    which they are changing the path curvature (1/m per s, + = turning more to the LEFT; 0.0 = attentive, holding).
    servo_hist: recent servo commands, oldest first, one per control tick of `dt` seconds."""
    n = int(round(attention_s / dt)) + 1
    if len(servo_hist) < max(n, window + 1):
        return None
    if np.ptp(np.asarray(servo_hist[-n:], float)) < activity_deg:
        return None
    ds = servo_hist[-1] - servo_hist[-1 - window]
    return -K_CURV_PER_SERVO_DEG * ds / (window * dt)


class RelayIntent:
    """The learned driver-intent model (adas/intent_net.py, trained in sim/train_intent_net.py) as the relay runs it:
    the stick sampled on the 50 ms clock the model was trained on, features from the live scan, the driver's online
    reaction-distance profile. It decides only whether the evasive steer may take over (sets `intent_hold` and
    `intent_k_rate` on the assists); braking stays pure physics. One class, used by the relay, the Monte Carlo and
    the scenario checks, so the simulator runs exactly the car's logic."""

    def __init__(self, assist, path=None, trust=0.5):
        import os
        from adas.intent_net import DriverProfile, IntentNet
        path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "intent_net.json")
        self.net = IntentNet(path) if os.path.exists(path) else None
        self.profile = DriverProfile()
        self.assist, self.trust_threshold = assist, trust
        self.hist, self.last_t = [], None
        self.p_crash, self.trusted, self.attentive = None, False, False

    def update(self, now, servo, physical, v, points):
        """servo: the driver's servo command, physical: the driver's throttle (+ forward), points: relay format."""
        from adas.intent_net import features
        if self.last_t is not None and now - self.last_t < 0.05 - 1e-6:
            return
        self.last_t = now
        a = self.assist
        self.hist = (self.hist + [servo])[-80:]
        self.p_crash, self.trusted = None, False
        if self.net is not None:
            f = features(self.hist, physical, v, a.points_vehicle_frame(points, a.p.lidar_x), a.p, a.centre,
                         K_CURV_PER_SERVO_DEG, profile=self.profile)
            if f is not None:
                self.profile.update(self.hist, f[len(f) - 5] * 1.5)
                self.p_crash = self.net.crash_probability(f)
                self.trusted = self.p_crash < self.trust_threshold
        k_rate = driver_intent(self.hist, 0.05, a.centre)
        self.attentive = k_rate is not None            # the stick moved in the last second
        a.assists.intent_hold = self.trusted
        a.assists.intent_k_rate = (k_rate or 0.0) if self.trusted else None

    @property
    def gate_trust(self):
        """For the brake gate's later soft cap: the model trusts the driver AND they are active on the stick."""
        return self.trusted and self.attentive

    def gui(self):
        return {"p_crash": None if self.p_crash is None else round(self.p_crash, 3), "trusted": self.trusted,
                "attentive": self.attentive, "reaction_m": round(self.profile.reaction_distance, 2)}


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
        self.odo = None                # scan-matching odometry, only while a manoeuvre runs
        self._odo_seq = None
        from adas.autonav import AutoNav
        self.nav = AutoNav(self.p, self.model)   # click-to-go autonomy (Hybrid A* + pure pursuit)
        self.nav_pose = (0.0, 0.0, 0.0)
        self._nav_stick = 0.0
        self._last_raw = []
        self._nav_was_active = False
        self._hold = False

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

    def _odometry(self, points, seq, stick, now=None):
        """While a manoeuvre runs, track the car by matching each LiDAR scan against the one taken when it began
        (pi/scanmatch.py - ~1 cm on the car in the obstacle-avoidance demo) and hand the pose to the assist."""
        a = self.assists
        if not (a.evading or self.nav.active):
            self.odo = None
            return
        if seq is None or seq == self._odo_seq or not points:
            return
        self._odo_seq = seq
        xy = polar_to_xy(points)                  # car angle clockwise-positive -> pi frame (y right)
        kappa_right = -math.tan(steer_to_delta(stick, self.p)) / self.p.wheelbase
        if self.odo is None:
            self.odo = Odometry()
        pose = self.odo.update(xy, time.time() if now is None else now, self.est.v, kappa_right)
        xl, yl, thl = float(pose[0]), -float(pose[1]), -float(pose[2])       # LiDAR pose, y left
        lx = self.p.lidar_x
        fix = (xl - lx * math.cos(thl) + lx, yl - lx * math.sin(thl), thl)   # rear axle, start frame
        if self.nav.active:
            self.nav_pose = fix
        else:
            a.pose_fix = fix

    # --- click-to-go autonomy
    def goto(self, x, y, points=None):
        """Start driving to (x, y) m in the vehicle frame now (x forward from the rear axle, y left).
        points: the latest scan (relay format); defaults to the last one seen by process()."""
        if self.assists.evading:
            self.assists._stop_evading("autonomy started")
        self.odo, self.nav_pose = None, (0.0, 0.0, 0.0)
        raw = points if points else self._last_raw
        return self.nav.start((float(x), float(y)), self.points_vehicle_frame(raw, self.p.lidar_x))

    def _navigate(self, dt, pts, stick, physical, points, seq, now):
        """One tick of autonomy. Like Smart Summon, the operator holds the throttle as a dead-man switch: held =
        drive at the planner's speed, released = stop and wait, stick or brake = cancel and hand back.
        Returns (stick, physical) to send, or None when the driver has control again."""
        if abs(stick) > self.assists.cfg.evade_driver_override:
            self.nav.cancel("driver steered - handed back")
            return None
        if physical < -20:
            self.nav.cancel("driver braked - handed back")
            return None
        x, y, th = self.nav_pose                  # dead reckoning between scans; the scan match below corrects it
        th += math.tan(steer_to_delta(self._nav_stick, self.p)) / self.p.wheelbase * self.est.v * dt
        self.nav_pose = (x + self.est.v * math.cos(th) * dt, y + self.est.v * math.sin(th) * dt, th)
        self._odometry(points, seq, self._nav_stick, now)
        out = self.nav.step(dt, self.nav_pose, pts, self.est.v)
        if out is None:
            return None
        kappa, v_target = out
        self._nav_stick = delta_to_steer(math.atan(kappa * self.p.wheelbase), self.p)
        if physical <= 20:
            self.info = {"autonomy": "paused - hold the throttle to drive"}
            return self._nav_stick, 0.0
        self.info = {"autonomy": self.nav.msg}
        return self._nav_stick, math.copysign(self.model.pwm_for_speed(abs(v_target)), v_target)

    def _rewrite(self, lines, i_a, i_m, servo, physical, dt):
        """Replace the driver's lines with our own steering (None = keep the driver's) and throttle."""
        w = -int(round(physical)) if self.reversed else int(round(physical))
        drop = (i_a, i_m) if servo is not None else (i_m,)
        out = [ln for i, ln in enumerate(lines) if i not in drop]
        if servo is not None:
            out.append(f"A {servo} {servo}")
        out.append(f"M {w}")
        self.level, self.changed = 2, True
        self.est.update(dt, physical)
        return out

    def process(self, lines, points, seq=None, now=None):
        """lines: the driver's packet lines. Returns the lines to hand on to the safety gate."""
        now = time.time() if now is None else now
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
        if points:
            self._last_raw = points
        if self.nav.active:
            nav = self._navigate(dt, self.points_vehicle_frame(points, self.p.lidar_x), stick, physical,
                                 points, seq, now)
            if nav is not None:
                self._nav_was_active = True
                s_out, p_out = nav
                sv = int(round(max(35.0, min(145.0, self.stick_to_servo(s_out)))))
                return self._rewrite(lines, i_a, i_m, sv, p_out, dt)
        if self._nav_was_active and not self.nav.msg.startswith("driver"):
            self._hold = True       # autonomy ended by itself (arrived / blocked / GUI): stay stopped until the
        self._nav_was_active = False            # operator lets go of the throttle, then hand back (as Summon does)
        if self._hold:
            if physical <= 20:
                self._hold = False
            else:
                self.info = {"autonomy": f"{self.nav.msg} - release the throttle to drive"}
                return self._rewrite(lines, i_a, i_m, None, 0.0, dt)
        if not any(self.assists.enabled.values()):
            self.est.update(dt, physical)
            self.info, self.level = {}, 0
            return lines
        pts = self.points_vehicle_frame(points, self.p.lidar_x)
        self._odometry(points, seq, stick, now)
        s_out, p_out, self.level = self.assists.update(dt, pts, stick, physical, self.est.v)
        self._odometry(points, seq, stick, now)     # a manoeuvre that just began: this scan is its reference
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
