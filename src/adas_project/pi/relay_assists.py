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


class RelaySpeed:
    """The relay's speed and yaw rate (TODO C5; evaluation: `python -m sim.odometry_eval`):
      model   the fitted throttle->speed model with a lag - what the relay used alone before; blind to a sagging
              battery, a carpet or a blocked wheel
      ekf     that model fused with LiDAR range flow (adas/rf2o.py + adas/speed_ekf.py): 3 cm/s RMSE even with the
              car 20 % off its model (the throttle model: 10 cm/s)
    `v` / `w` are the EKF's. For braking, `v_gate` takes the more conservative of the two in the direction of
    travel, so adding the LiDAR estimate can only make the brake earlier, never later. Shared by the relay, the
    Monte Carlo and the scenario checks."""

    def __init__(self, speed_model, lidar_x=0.12, car_model_path=None):
        import json
        import os
        from adas.aeb import SpeedEstimator
        from adas.rf2o import RangeFlow
        from adas.speed_ekf import SpeedEKF
        self.model_est = SpeedEstimator(speed_model)
        cm = {}
        path = car_model_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "car_model.json")
        try:
            cm = json.load(open(path))
        except (OSError, ValueError):
            pass
        self.ekf = SpeedEKF(speed_model.v_max, speed_model.deadband, tau=max(cm.get("tau_motor", 0.1), 0.05),
                            delay=cm.get("delay_s", 0.12), coast_decel=4.0, brake_decel=8.0,   # measured, see sim/hw_sim.py
                            k_curv_per_deg=cm.get("k_curv_per_deg", K_CURV_PER_SERVO_DEG),
                            servo_centre=cm.get("servo_centre", 87.0), lidar_x=lidar_x)
        self.rf = RangeFlow()
        self.seq = None
        self.meas = None

    def command(self, t, dt, physical, servo):
        """What was actually sent: physical throttle (+ forward) and servo degrees, at time t."""
        self.model_est.update(dt, physical)
        self.ekf.command(t, physical, servo)
        self.ekf.predict(t)

    def on_scan(self, points, seq, t):
        """A LiDAR scan in the relay's format (car angle deg clockwise-positive, distance from the LiDAR)."""
        if seq is None or seq == self.seq or not points:
            return
        self.seq = seq
        a = -np.radians(np.array([q[0] for q in points]))
        d = np.array([q[1] for q in points])
        xy = np.column_stack([d * np.cos(a), d * np.sin(a)])
        e = self.ekf
        self.meas = self.rf.update(xy, t, guess=(e.v, e.w * e.lx, e.w))
        if self.meas is not None:
            e.correct(t, self.meas)

    @property
    def v(self):
        return self.ekf.v

    @property
    def w(self):
        return self.ekf.w

    def v_gate(self, direction):
        m, e = self.model_est.v, self.ekf.v
        if direction > 0:
            return max(m, e)
        if direction < 0:
            return min(m, e)
        return e


class ThrottleSmoother:
    """Rate limit on the throttle actually sent (user, 28 Sep: the car jumped forward / backward when a manoeuvre
    re-planned or switched direction). Up gently, down quickly, and through zero when changing direction; a cut
    made for safety (brake gate braking / holding / stopped, health fault) goes straight through."""

    RISE_PWM_S = 600.0     # 0 -> full throttle in ~0.4 s
    FALL_PWM_S = 1500.0    # full -> 0 in ~0.17 s

    def __init__(self):
        self.out = 0.0

    def step(self, target, dt, emergency=False):
        if emergency:
            self.out = float(target)
            return self.out
        cur = self.out
        if cur != 0 and target * cur < 0:            # changing direction: come down to zero first
            target = 0.0
        if abs(target) > abs(cur):
            cur += math.copysign(min(abs(target) - abs(cur), self.RISE_PWM_S * dt), target)
        elif cur:
            cur -= math.copysign(min(abs(cur) - abs(target), self.FALL_PWM_S * dt), cur)
        self.out = cur
        return cur


class RelayIntent:
    """The learned driver-intent model (adas/intent_net.py, trained in sim/train_intent_net.py) as the relay runs it:
    the stick sampled on the 50 ms clock the model was trained on, features from the live scan, the driver's online
    reaction-distance profile. It decides only whether the evasive steer may take over (sets `intent_hold` and
    `intent_k_rate` on the assists); braking stays pure physics. One class, used by the relay, the Monte Carlo and
    the scenario checks, so the simulator runs exactly the car's logic."""

    def __init__(self, assist, path=None, trust=0.5, risk_path=None):
        import os
        from adas.intent_net import DriverProfile, IntentNet
        here = os.path.dirname(os.path.abspath(__file__))
        path = path or os.path.join(here, "intent_net.json")
        self.net = IntentNet(path) if os.path.exists(path) else None
        # Two jobs, two models (Monte Carlo 28 Sep, models/mc_intent_v2_v3.json): the model at `path` (v2) decides
        # whether the evasive steer may take over - it gave the least needless overriding with 0 crashes; the
        # twin-trained v3 (pi/intent_v3.json, sim/train_intent_torch.py) is far more accurate and calibrated
        # (held-out AP 0.42 -> 0.86) and supplies the risk the driver sees and the warnings.
        rp = risk_path or os.path.join(here, "intent_v3.json")
        self.risk_net = IntentNet(rp) if os.path.exists(rp) else None
        self.p_risk = None
        self.profile = DriverProfile()
        self.assist, self.trust_threshold = assist, trust
        self.hist, self.pwm_hist, self.last_t = [], [], None
        self.zhist = []                                 # v3 models: the last 1.6 s of tick vectors
        self.p_crash, self.trusted, self.attentive = None, False, False

    def update(self, now, servo, physical, v, points):
        """servo: the driver's servo command, physical: the driver's throttle (+ forward), points: relay format."""
        from adas.intent_net import features
        if self.last_t is not None and now - self.last_t < 0.05 - 1e-6:
            return
        self.last_t = now
        from adas.intent_net import free_now
        a = self.assist
        self.hist = (self.hist + [servo])[-80:]
        self.pwm_hist = (self.pwm_hist + [physical])[-40:]
        self.p_crash, self.trusted = None, False
        pts_v = a.points_vehicle_frame(points, a.p.lidar_x)
        decides_v3 = self.net is not None and self.net.version == 3
        v3 = self.net if decides_v3 else self.risk_net
        if v3 is not None:
            from adas.intent_net import HORIZON3, WINDOW, Z_FREE_NOW, tick_vector, window_of
            z = tick_vector(servo, physical, v, pts_v, a.p, a.centre, K_CURV_PER_SERVO_DEG,
                            react=self.profile.reaction_distance)
            self.zhist = (self.zhist + [z])[-WINDOW:]
            self.p_risk = v3.risk(window_of(self.zhist))
            if decides_v3:
                self.profile.update(self.hist, z[Z_FREE_NOW] * HORIZON3)
                self.p_crash = self.p_risk
                self.trusted = self.p_crash < self.trust_threshold
        if self.net is not None and not decides_v3:
            f = features(self.hist, physical, v, pts_v, a.p, a.centre,
                         K_CURV_PER_SERVO_DEG, profile=self.profile, pwm_hist=self.pwm_hist)
            if f is not None:
                self.profile.update(self.hist, free_now(f))
                self.p_crash = self.net.crash_probability(f)
                self.trusted = self.p_crash < self.trust_threshold
        k_rate = driver_intent(self.hist, 0.05, a.centre)
        self.attentive = k_rate is not None            # the stick moved in the last second
        a.assists.intent_hold = self.trusted
        a.assists.intent_attentive = self.attentive
        a.assists.intent_k_rate = (k_rate or 0.0) if self.trusted else None

    @property
    def gate_trust(self):
        """For the brake gate's later soft cap: the model trusts the driver AND they are active on the stick."""
        return self.trusted and self.attentive

    @property
    def gate_k_rate(self):
        """The trusted, active driver's curvature rate (1/m per s) for the gate's predicted path, else None."""
        return self.assist.assists.intent_k_rate if self.gate_trust else None

    def gui(self):
        """p_crash: the risk the driver sees (v3 when available); p_decision: the model deciding takeovers."""
        shown = self.p_risk if self.p_risk is not None else self.p_crash
        return {"p_crash": None if shown is None else round(shown, 3),
                "p_decision": None if self.p_crash is None else round(self.p_crash, 3), "trusted": self.trusted,
                "attentive": self.attentive, "reaction_m": round(self.profile.reaction_distance, 2)}


class RelayAssists:
    NAMES = ("evasive", "centring", "limiter", "narrow", "proximity", "nudge")

    def __init__(self, tuning, motor_reversed=True, plan_mode="inline", plan_latency=0.0):
        """plan_mode 'process': path searches run in a worker process (the relay); 'inline' computes them at once,
        released after compute time x plan_latency (simulations: the Pi's delay)."""
        from adas.plan_service import PlanService
        self.p = car_params(tuning.mount)
        self.centre = float(tuning.servo.left_center)
        self.model = tuning.speed_model
        self.assists = DrivingAssists(self.p, self.model)
        self.planner = PlanService(plan_mode, plan_latency).start()
        self.assists.plan_service = self.planner
        self.est = SpeedEstimator(self.model)
        self.speed = None              # a RelaySpeed shared with the brake gate; if set, the assists use its speed
        self.nudge_on = False          # steering correction instead of braking (see nudge())
        self._nudge_note = None        # this tick's correction, for the GUI (process() keeps it)
        self.reversed = motor_reversed
        self.last_t = None
        self.last_servo = self.centre
        self.info, self.level, self.changed = {}, 0, False
        self.odo = None                # scan-matching odometry, only while a manoeuvre runs
        self._odo_seq = None
        from adas.autonav import AutoNav
        self.nav = AutoNav(self.p, self.model, service=self.planner)   # click-to-go (Hybrid A* + pure pursuit)
        self.nav_pose = (0.0, 0.0, 0.0)
        self._nav_stick = 0.0
        self._last_raw = []
        self._nav_was_active = False
        self._hold = False

    def enabled(self):
        out = {k: bool(v) for k, v in self.assists.enabled.items()}
        out["nudge"] = self.nudge_on
        return out

    def nudge(self, gate, lines, v):
        """Steering correction instead of braking (pi/path_gate.steer_correction), when the 'nudge' assist is on:
        the driver's steering is changed by the smallest amount that makes their path safe at this speed. Forward
        only, never during a manoeuvre, not for a driver the intent model trusts who is steering. Call it BEFORE
        process(): the evasive steer then sees the corrected path, so a small correction pre-empts a full swerve.
        The gate must have the latest scan. Returns (lines, corrected steering angle or None)."""
        self._nudge_note = None
        if not self.nudge_on or self.assists.phase is not None or self.nav.active:
            return lines, None
        a = self.assists
        if a.intent_hold and a.intent_attentive:          # the learned intent model: this driver is handling it
            return lines, None
        servo, wire, i_a = None, None, None
        for i, ln in enumerate(lines):
            p = ln.split()
            try:
                if p[0] == "A" and len(p) == 3:
                    servo, i_a = (float(p[1]) + float(p[2])) / 2.0, i
                elif p[0] == "M" and len(p) == 2:
                    wire = float(p[1])
            except (ValueError, IndexError):
                pass
        self._nudge_note = None
        if servo is None or wire is None:
            return lines, None
        physical = -wire if self.reversed else wire
        if physical <= 0:
            return lines, None
        wb = self.p.wheelbase
        delta = math.atan(-K_CURV_PER_SERVO_DEG * (servo - self.centre) * wb)
        d2 = gate.steer_correction(delta, 1, v)
        if d2 is None:
            return lines, None
        sv = self.centre - (math.tan(d2) / wb) / K_CURV_PER_SERVO_DEG
        sv = int(round(max(35.0, min(145.0, sv))))
        out = list(lines)
        out[i_a] = f"A {sv} {sv}"
        self._nudge_note = gate.info.get("nudge", "steering corrected")
        self.info = dict(self.info, nudge=self._nudge_note)
        self.changed = True
        return out, d2

    @property
    def v(self):
        """Speed for the assists: the shared LiDAR+model estimate when the relay provides one."""
        return self.est.v if self.speed is None else self.speed.v

    def set(self, name, on):
        if name == "all":
            for k in self.NAMES:
                self.set(k, on)
        elif name == "nudge":
            self.nudge_on = bool(on)
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
        if not (a.phase is not None or self.nav.active):     # a manoeuvre being planned or driven
            self.odo = None
            return
        if seq is None or seq == self._odo_seq or not points:
            return
        self._odo_seq = seq
        xy = polar_to_xy(points)                  # car angle clockwise-positive -> pi frame (y right)
        kappa_right = -math.tan(steer_to_delta(stick, self.p)) / self.p.wheelbase
        if self.odo is None:
            self.odo = Odometry()
        pose = self.odo.update(xy, time.time() if now is None else now, self.v, kappa_right)
        xl, yl, thl = float(pose[0]), -float(pose[1]), -float(pose[2])       # LiDAR pose, y left
        lx = self.p.lidar_x
        fix = (xl - lx * math.cos(thl) + lx, yl - lx * math.sin(thl), thl)   # rear axle, start frame
        if self.nav.active:
            self.nav_pose = fix
        else:
            a.pose_fix = fix

    # --- click-to-go autonomy
    def goto(self, x, y, points=None, heading_deg=None):
        """Start driving to (x, y) m in the vehicle frame now (x forward from the rear axle, y left), arriving
        pointing heading_deg (0 = the car's heading now, + = left) or any way if None.
        points: the latest scan (relay format); defaults to the last one seen by process()."""
        if self.assists.phase is not None:
            self.assists._stop_evading("autonomy started")
        self.odo, self.nav_pose = None, (0.0, 0.0, 0.0)
        raw = points if points else self._last_raw
        return self.nav.start((float(x), float(y)), self.points_vehicle_frame(raw, self.p.lidar_x),
                              None if heading_deg is None else math.radians(float(heading_deg)))

    def planned_leg(self):
        """The planned path leg being followed right now (click-to-go or evasive manoeuvre), for the brake gate."""
        if self.nav.active:
            return self.nav.leg()
        return self.assists.leg()

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
        v = self.v
        th += math.tan(steer_to_delta(self._nav_stick, self.p)) / self.p.wheelbase * v * dt
        self.nav_pose = (x + v * math.cos(th) * dt, y + v * math.sin(th) * dt, th)
        self._odometry(points, seq, self._nav_stick, now)
        out = self.nav.step(dt, self.nav_pose, pts, v, held=physical > 20)
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
        nudged = {"nudge": self._nudge_note} if self._nudge_note else {}   # a correction made before process()
        self.changed = bool(nudged)
        if not any(self.assists.enabled.values()):
            self.est.update(dt, physical)
            self.info, self.level = dict(nudged), (1 if nudged else 0)
            return lines
        pts = self.points_vehicle_frame(points, self.p.lidar_x)
        self._odometry(points, seq, stick, now)
        s_out, p_out, self.level = self.assists.update(dt, pts, stick, physical, self.v)
        self._odometry(points, seq, stick, now)     # a manoeuvre that just began: this scan is its reference
        self.info = dict(self.assists.info, **nudged)
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
