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
# realistic steering (user, 28 Sep): the car is ~1:14 of a 4.5 m car (body ~0.33 m); lock capped at ~2 wheelbases of
# turning radius as on real cars (0.40 m -> 2.5 1/m, ~26 deg at the wheels), and at speed the full-size lateral
# acceleration kept under ~0.6 g (a typical handling envelope; road cars feel steering limits at speed this way)
CAR_SCALE = 14.0
KAPPA_REAL = 2.5
LAT_ACCEL_REAL = 6.0
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
        from adas.latency import DelayEstimator
        self.latency = DelayEstimator()        # how late the scans are (adas/latency.py); the gate reads .excess
        self.seq = None
        self.meas = None
        # world pose (rear axle; x, y m, heading rad), the frame of the speed zones and the world map: scan matching
        # (pi/scanmatch.Odometry, keyframed every KF_M / KF_DEG) on every scan, dead reckoning in between. Integrating
        # the EKF alone drifted 7-11 % and 13 deg over 2-3 m in the twin.
        self.lidar_x = lidar_x
        self.pose = (0.0, 0.0, 0.0)
        self._kf, self._odo = (0.0, 0.0, 0.0), None

    KF_M, KF_DEG = 1.0, 30.0

    def reset_pose(self):
        self.pose, self._kf, self._odo = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None

    def _scan_pose(self, points, t):
        from pi.scanmatch import Odometry, polar_to_xy
        xy = polar_to_xy(points)
        if len(xy) < 40:
            return
        if self._odo is None:
            self._odo, self._kf = Odometry(), self.pose
        v = self.ekf.v
        k_right = -(self.ekf.w / v) if abs(v) > 0.05 else 0.0
        p = self._odo.update(xy, t, v, k_right)
        xl, yl, thl = float(p[0]), -float(p[1]), -float(p[2])        # the LiDAR in the keyframe, y left
        lx = self.lidar_x
        dx, dy = xl - lx * math.cos(thl) + lx, yl - lx * math.sin(thl)   # the rear axle in the keyframe
        kx, ky, kth = self._kf
        c, sn = math.cos(kth), math.sin(kth)
        self.pose = (kx + c * dx - sn * dy, ky + sn * dx + c * dy, kth + thl)
        if math.hypot(dx, dy) > self.KF_M or abs(thl) > math.radians(self.KF_DEG):
            self._odo = None                      # new keyframe from the next scan

    def command(self, t, dt, physical, servo):
        """What was actually sent: physical throttle (+ forward) and servo degrees, at time t."""
        self.model_est.update(dt, physical)
        self.latency.push_model(t, self.model_est.v)
        self.ekf.command(t, physical, servo)
        self.ekf.predict(t)
        x, y, th = self.pose
        th2 = th + self.ekf.w * dt
        mid = (th + th2) / 2
        self.pose = (x + self.ekf.v * math.cos(mid) * dt, y + self.ekf.v * math.sin(mid) * dt, th2)

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
            self.latency.push_flow(t, self.meas[0])
        self._scan_pose(points, t)

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


def _stick_of(kappa, p):
    """Stick position (-1..1) for a path curvature (1/m, + = left)."""
    return delta_to_steer(math.atan(kappa * p.wheelbase), p)


DRIVE_MODES = {
    # throttle cap (fraction of full), how fast the throttle may rise (PWM / s), label - the safety margins do NOT change
    "eco": {"cap": 0.55, "rise": 320.0, "label": "ECO"},
    "normal": {"cap": 1.00, "rise": 600.0, "label": "NORMAL"},
    "sport": {"cap": 1.00, "rise": 950.0, "label": "SPORT"},
}


def home_goal(pose):
    """The world origin (where the car started / the last ORIGIN reset) as a click-to-go goal in the vehicle frame now:
    (x ahead, y left, heading to arrive with in degrees relative to the car's heading now = the original heading)."""
    px, py, th = pose
    c, s = math.cos(th), math.sin(th)
    x, y = -px, -py
    return (c * x + s * y, -s * x + c * y, math.degrees(math.remainder(-th, 2 * math.pi)))


class ThrottleSmoother:
    """Rate limit on the throttle actually sent (user, 28 Sep: the car jumped forward / backward when a manoeuvre
    re-planned or switched direction). Up gently, down quickly, and through zero when changing direction; a cut
    made for safety (brake gate braking / holding / stopped, health fault) goes straight through."""

    RISE_PWM_S = 600.0     # 0 -> full throttle in ~0.4 s
    FALL_PWM_S = 1500.0    # full -> 0 in ~0.17 s

    def __init__(self):
        self.out = 0.0

    REVERSE_LOCK_V = 0.12  # m/s: no driving the other way while still moving faster than this (as ESCs do)

    def step(self, target, dt, emergency=False, v=None):
        if emergency:
            self.out = float(target)
            return self.out
        cur = self.out
        if cur != 0 and target * cur < 0:            # changing direction (or ending a brake pulse): drop to zero at
            cur = 0.0                                # once - ramping a reverse pulse down would keep driving backwards
        # reverse lockout (drivetrain protection, after the rear shaft broke on 28 Sep): driving against the way the
        # car is still rolling shock-loads the gearbox and shaft - coast down first
        if v is not None and target * v < 0 and abs(v) > self.REVERSE_LOCK_V:
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
        # progress monitor: intent may hold the evasive steer back only while the driver is getting somewhere. A driver the
        # model trusts but who has been pushing the throttle for STALL_S without leaving a ~STALL_M patch is stuck (a corner,
        # a gap that will not fit), and trust ends - the evasive planner then takes over (Monte Carlo 29 Sep: ADAS + intent
        # sat in a corner for 25 s where plain ADAS drove out)
        self.track, self.release_until, self.stalled = [], -1.0, False

    STALL_S, STALL_M, RELEASE_S = 6.0, 0.9, 10.0

    def _progress(self, now, physical):
        sp = getattr(self.assist, "speed", None)
        if sp is None:
            return
        x, y = sp.pose[0], sp.pose[1]
        if not self.track or now - self.track[-1][0] >= 0.25:
            self.track.append((now, x, y, abs(physical) > 40))
        self.track = [r for r in self.track if now - r[0] <= self.STALL_S]
        if len(self.track) < 12 or now - self.track[0][0] < self.STALL_S - 0.6:
            return
        xs, ys = np.array([r[1] for r in self.track]), np.array([r[2] for r in self.track])
        active = np.mean([r[3] for r in self.track])
        spread = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
        if active >= 0.6 and spread < self.STALL_M:
            self.release_until = now + self.RELEASE_S
            self.track = []

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
            z = tick_vector(servo, physical, v, pts_v, a.p, a.centre, a.k,
                            react=self.profile.reaction_distance)
            self.zhist = (self.zhist + [z])[-WINDOW:]
            self.p_risk = v3.risk(window_of(self.zhist))
            if decides_v3:
                self.profile.update(self.hist, z[Z_FREE_NOW] * HORIZON3)
                self.p_crash = self.p_risk
                self.trusted = self.p_crash < self.trust_threshold
        if self.net is not None and not decides_v3:
            f = features(self.hist, physical, v, pts_v, a.p, a.centre,
                         a.k, profile=self.profile, pwm_hist=self.pwm_hist)
            if f is not None:
                self.profile.update(self.hist, free_now(f))
                self.p_crash = self.net.crash_probability(f)
                self.trusted = self.p_crash < self.trust_threshold
        self._progress(now, physical)
        self.stalled = now < self.release_until
        if self.stalled:                                 # stuck: the model's trust is withdrawn for a while
            self.trusted = False
            self.p_crash = None if self.p_crash is None else max(self.p_crash, self.trust_threshold)
        k_rate = driver_intent(self.hist, 0.05, a.centre)
        self.attentive = k_rate is not None            # the stick moved in the last second
        a.assists.intent_hold = self.trusted
        a.assists.intent_stalled = self.stalled
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
                "attentive": self.attentive, "reaction_m": round(self.profile.reaction_distance, 2),
                "stalled": self.stalled}


class RelayAssists:
    NAMES = ("evasive", "centring", "limiter", "narrow", "proximity", "nudge", "moving")

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
        self.memory = None             # the brake gate's obstacle memory (its blind-ring points join the planners)
        from pi.zones import SpeedZones
        self.zones = SpeedZones()      # speed-limit zones in the world frame (pi/zones.py)
        self.zone_kph = None           # the limit applying right now (km/h full-size), for the GUI
        # online steering calibration (adas/online_steering.py): the servo centre and the steering gain the predicted
        # paths use follow what the car really does (scan-matched heading change per distance driven)
        from adas.online_steering import OnlineSteering
        self.k = K_CURV_PER_SERVO_DEG
        self.centre0, self.k0 = self.centre, self.k
        self.steer_est = OnlineSteering(self.centre, self.k, prior_weight=2)
        self.steer_adapt = True
        self.drive_mode = "normal"     # eco / normal / sport (DRIVE_MODES): throttle cap and response only
        self.steer_envelope = True     # realistic, speed-dependent steering limit (steer_limit_kappa)
        self.steer_limited = None      # (asked, allowed) servo offsets when the envelope clipped the steering
        self.nudge_on = False          # steering correction instead of braking (see nudge())
        self.moving_on = False         # predictive speed choice for moving obstacles (adas/crossing.py)
        self._tracks = []              # the relay's moving-object tracks (adas/tracking.py), set every packet
        self.crossing = None           # (action, target m/s, info) of the last decision, for the GUI
        self._xhold, self._xv = None, {}    # a swerve being held (s), smoothed velocities of moving tracks
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
        delta = math.atan(-self.k * (servo - self.centre) * wb)
        d2 = gate.steer_correction(delta, 1, v)
        if d2 is None:
            return lines, None
        sv = self.centre - (math.tan(d2) / wb) / self.k
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
        elif name == "moving":
            self.moving_on = bool(on)
        elif name in self.NAMES:
            self.assists.enabled[name] = on

    # --- unit conversions
    def servo_to_stick(self, servo):
        kappa_left = -self.k * (servo - self.centre)
        return delta_to_steer(math.atan(kappa_left * self.p.wheelbase), self.p)

    def stick_to_servo(self, stick):
        kappa_left = math.tan(steer_to_delta(stick, self.p)) / self.p.wheelbase
        return self.centre - kappa_left / self.k

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
        return self.nav.start((float(x), float(y)), self._planning_points(raw),
                              None if heading_deg is None else math.radians(float(heading_deg)))

    def _planning_points(self, points):
        """The scan in the vehicle frame PLUS the brake gate's remembered points in the LiDAR's blind ring (closer
        than its minimum range, e.g. right at the nose): the planners must avoid what the gate will brake for, or
        they plan through a spot the gate then refuses to drive (seen on the car, 28 Sep)."""
        v = self.points_vehicle_frame(points, self.p.lidar_x)
        if self.memory is not None:
            blind = self.memory.blind_points()
            if len(blind):
                v = np.vstack([v, blind]) if len(v) else blind
        return v

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

    ADAPT_MIN_SAMPLES = 10          # real samples before the estimate is used
    ADAPT_BLEND = 0.05              # per update: the calibration moves slowly, it is a bias not a signal

    def _adapt_steering(self, lines, now):
        """Feed the online steering estimator with the servo the driver / assists asked for and the scan-matched pose,
        and move the centre / gain the predicted paths use toward its estimate."""
        if not self.steer_adapt or self.speed is None:
            return
        for ln in lines:
            q = ln.split()
            if len(q) == 3 and q[0] == "A":
                try:
                    self.steer_est.command(now, (float(q[1]) + float(q[2])) / 2.0)
                except ValueError:
                    pass
        self.steer_est.update(now, self.v, self.speed.pose)
        if self.steer_est.n >= self.ADAPT_MIN_SAMPLES:
            b = self.ADAPT_BLEND
            self.centre += b * (self.steer_est.centre - self.centre)
            self.k += b * (self.steer_est.k - self.k)

    def steer_limit_kappa(self, v):
        """Realistic steering envelope (user, 28 Sep): the tightest path curvature allowed at speed v (m/s).
        Mechanically the car could turn on 0.27 m (1.35 wheelbases); real cars need ~2 wheelbases (e.g. 2.7 m
        wheelbase, 5.3 m radius), so the lock is capped at KAPPA_REAL. At speed, the curvature shrinks so the
        full-size equivalent stays under LAT_ACCEL_REAL (the car is ~1:SCALE, speeds scaled the same way):
        kappa_car = a_real / (SCALE * v_car^2)."""
        v = abs(v)
        k = KAPPA_REAL
        if v > 0.05:
            k = min(k, LAT_ACCEL_REAL / (CAR_SCALE * v * v))
        return k

    def limit_servo(self, servo, v):
        """Clamp a servo command to the steering envelope at speed v."""
        k = self.steer_limit_kappa(v)
        lo, hi = self.centre - k / self.k, self.centre + k / self.k
        return max(lo, min(hi, servo))

    def process(self, lines, points, seq=None, now=None):
        """lines: the driver's packet lines. Returns the lines to hand on to the safety gate, with the steering
        inside the realistic, speed-dependent envelope (steer_limit_kappa)."""
        self._adapt_steering(lines, time.time() if now is None else now)
        out = self._process(lines, points, seq, now)
        out = self._crossing(out, time.time() if now is None else now)
        out = self._zone_cap(out)
        out = self._mode_cap(out)
        if not self.steer_envelope:
            return out
        v = self.v
        res = []
        for ln in out:
            q = ln.split()
            if len(q) == 3 and q[0] == "A":
                try:
                    sv = (float(q[1]) + float(q[2])) / 2.0
                except ValueError:
                    res.append(ln)
                    continue
                lim = self.limit_servo(sv, v)
                if abs(lim - sv) > 0.5:
                    self.steer_limited = (round(sv - self.centre), round(lim - self.centre))
                    ln = f"A {lim:.0f} {lim:.0f}"
                else:
                    self.steer_limited = None
            res.append(ln)
        return res

    def set_mode(self, name):
        if name in DRIVE_MODES:
            self.drive_mode = name
            return True
        return False

    def _mode_cap(self, out):
        """Eco caps the throttle at 55 % of full; normal and sport do not (sport also responds faster - the relay's
        ThrottleSmoother uses DRIVE_MODES[mode]['rise']). Safety margins are identical in every mode."""
        cap = DRIVE_MODES[self.drive_mode]["cap"]
        if cap >= 1.0:
            return out
        res = []
        for ln in out:
            q = ln.split()
            if len(q) == 2 and q[0] == "M":
                try:
                    w = float(q[1])
                    if abs(w) > 255 * cap:
                        ln = f"M {int(math.copysign(255 * cap, w))}"
                        self.info.setdefault("mode", "ECO: throttle limited")
                except ValueError:
                    pass
            res.append(ln)
        return res

    def set_tracks(self, tracks):
        self._tracks = list(tracks or [])

    def _crossing(self, out, now=0.0):
        """Predictive speed choice for MOVING obstacles (user, 28 Sep): the car keeps its path (the driver's steering
        arc, or the planned leg in autonomy) and only its speed changes - speed up to get past first, slow down to
        let it pass, stop, or back away if it is coming at the car (adas/crossing.py). The brake gate still has the
        last word on the throttle."""
        self.crossing = None
        if not self.moving_on or not self._tracks:
            self._xhold, self._xv = None, {}
            return out
        from adas.crossing import CrossingConfig, arc_path, decide
        from adas.geometry import travel_distance_to_contact
        # the tracked velocity is noisy (a straight-line fit over ~0.5 s of a few points): smooth it per track, or
        # the decision flips between yield / pass every tick
        moving, seen = [], {}
        for t in self._tracks:
            # only tracks that have moved consistently and at a real speed: with noisy scans wall points flicker into
            # 'moving' clusters (seen at 20 mm range noise), and reacting to them is worse than ignoring them
            if not t.moving or t.moving_count < 5 or math.hypot(*t.vel_obj) < 0.15 or math.hypot(*t.vel_obj) > 2.0:
                continue
            ov = self._xv.get(t.id, t.vel_obj)
            sv = (0.6 * ov[0] + 0.4 * t.vel_obj[0], 0.6 * ov[1] + 0.4 * t.vel_obj[1])
            seen[t.id] = sv
            moving.append((t.pos[0], t.pos[1], sv[0], sv[1], t.radius))
        self._xv = seen
        if not moving:
            return out
        i_m, wire, servo = None, None, self.centre
        for i, ln in enumerate(out):
            q = ln.split()
            try:
                if q[0] == "M" and len(q) == 2:
                    i_m, wire = i, float(q[1])
                elif q[0] == "A" and len(q) == 3:
                    servo = (float(q[1]) + float(q[2])) / 2.0
            except (ValueError, IndexError):
                pass
        if wire is None:
            return out
        physical = -wire if self.reversed else wire
        if abs(physical) < 20:                     # the driver is not asking the car to move: nothing to time
            return out
        direction = 1 if physical > 0 else -1
        leg = self.nav.leg() if self.nav.active else None
        if leg is not None and leg[1] == direction:
            path = leg[0]
        else:
            path = arc_path(-self.k * (servo - self.centre), direction)
        v_along = max(0.0, self.v * direction)
        v_want = self.model.speed(abs(physical))
        pts = self.points_vehicle_frame(self._last_raw, self.p.lidar_x) if self._last_raw else np.empty((0, 2))
        rear = travel_distance_to_contact(pts, 0.0, -1, self.p, horizon=1.0, margin=0.03) if len(pts) else math.inf
        action, v_t, info = decide(moving, path, v_along, v_want, self.p, CrossingConfig(), rear_free=rear)
        self.crossing = (action, v_t, info)
        if action == "clear" and not (self._xhold is not None and now < self._xhold["until"]):
            return out
        swerve = None
        if self._xhold is not None and now < self._xhold["until"]:
            swerve = self._xhold["swerve"]                       # a swerve in progress is followed, not re-decided
        elif action in ("away", "stop") and not self.nav.active and len(pts):
            # nothing works along the current path (it is coming AT the car): move out of the way - the first
            # steering arc (smallest turn first, either side) that is clear of the static scene and safe at some speed
            near = min(moving, key=lambda m: math.hypot(m[0], m[1]))
            side = -1.0 if near[1] > 0.02 else 1.0            # turn away from the side it is on (ties: left)
            for k in (1.5 * side, 1.0 * side, 1.5 * -side, 1.0 * -side, 0.5 * side, 0.5 * -side):
                a2, v2, i2 = decide(moving, arc_path(k, direction), v_along, v_want, self.p, CrossingConfig(),
                                    rear_free=rear)
                free = travel_distance_to_contact(pts, math.atan(k * self.p.wheelbase), direction, self.p,
                                                  horizon=1.2, margin=0.05)
                if a2 in ("clear", "pass", "yield") and v2 > 0.05 and free > 0.9:
                    swerve = (k, a2, v2, i2)
                    break
        if swerve is not None:
            if self._xhold is None or now >= self._xhold["until"]:
                self._xhold = {"until": now + 1.4, "swerve": swerve}
            k, action, v_t, info = swerve
            info = dict(info, why="moving out of its way")
            servo_new = int(round(max(35.0, min(145.0, self.stick_to_servo(_stick_of(k, self.p))))))
            out = list(out)
            i_a = next((i for i, ln in enumerate(out) if ln.split()[0] == "A"), None)
            line = f"A {servo_new} {servo_new}"
            if i_a is None:
                out.append(line)
            else:
                out[i_a] = line
            new_phys = direction * self.model.pwm_for_speed(max(0.0, v_t))
            w = -int(round(new_phys)) if self.reversed else int(round(new_phys))
            out[i_m] = f"M {w}"
            self.info["moving"] = "swerve: moving out of its way"
            self.level, self.changed = max(self.level, 2), True
            return out
        if action == "away":
            new_phys = -direction * self.model.pwm_for_speed(0.15) if direction > 0 else 0.0
        else:
            new_phys = direction * self.model.pwm_for_speed(max(0.0, v_t)) if v_t > 0 else 0.0
            if action in ("yield", "wait", "stop"):
                new_phys = math.copysign(min(abs(new_phys), abs(physical)), direction) if new_phys else 0.0
        w = -int(round(new_phys)) if self.reversed else int(round(new_phys))
        out = list(out)
        out[i_m] = f"M {w}"
        self.info["moving"] = f"{action}: {info.get('why', '')}".strip(": ")
        self.level, self.changed = max(self.level, 2), True
        return out

    def _zone_cap(self, out):
        """Cap the throttle to the speed-limit zone the car is in or about to enter (pi/zones.py)."""
        self.zone_kph = None
        if not self.zones.zones or self.speed is None:
            return out
        from pi.zones import kph_to_car
        kph = self.zones.limit_ahead(self.speed.pose, self.v)
        self.zone_kph = kph
        if kph is None:
            return out
        cap = self.model.pwm_for_speed(kph_to_car(kph))
        res = []
        for ln in out:
            q = ln.split()
            if len(q) == 2 and q[0] == "M":
                try:
                    w = float(q[1])
                except ValueError:
                    res.append(ln)
                    continue
                if abs(w) > cap:
                    ln = f"M {int(math.copysign(cap, w))}"
                    self.info["zone"] = f"speed limit {kph:.0f} km/h"
            res.append(ln)
        return res

    def _process(self, lines, points, seq=None, now=None):
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
            nav = self._navigate(dt, self._planning_points(points), stick, physical,
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
        pts = self._planning_points(points)
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
