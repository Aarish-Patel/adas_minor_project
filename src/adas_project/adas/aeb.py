"""Collision warning and automatic emergency braking (AEB).

Works on obstacle points in the vehicle frame, the driver's motor command
(PWM -255..255) and an estimate of the car's speed. There is no wheel encoder,
so speed comes from a PWM->speed calibration (SpeedModel), and the stopping
maths includes the delay of the LiDAR scan and the link.
"""

import math
from dataclasses import dataclass

from .geometry import travel_distance_to_contact
from .tracking import moving_object_contact
from .vehicle_params import VehicleParams


@dataclass(frozen=True)
class SpeedModel:
    """Open-loop motor model. CALIBRATE on the real car with a timed run."""
    v_max: float = 1.0        # m/s at PWM 255
    deadband: float = 40.0    # PWM below this does not move the car

    def speed(self, pwm):
        mag = max(0.0, (abs(pwm) - self.deadband) / (255.0 - self.deadband))
        return math.copysign(self.v_max * min(mag, 1.0), pwm) if pwm else 0.0

    def pwm_for_speed(self, v):
        if v <= 0.0:
            return 0.0
        return min(255.0, self.deadband + v / self.v_max * (255.0 - self.deadband))


class SpeedEstimator:
    """Predicts speed from the PWM that was commanded (first-order lag)."""

    def __init__(self, model, tau=0.15):
        self.model = model
        self.tau = tau
        self.v = 0.0

    def update(self, dt, pwm_cmd):
        target = self.model.speed(pwm_cmd)
        self.v += (target - self.v) * min(1.0, dt / self.tau)
        return self.v


@dataclass
class AEBConfig:
    decel: float = 1.5          # m/s^2 the car is assumed able to brake at (conservative)
    latency: float = 0.15       # s: scan age (up to 0.1) + link + processing
    margin: float = 0.05        # m of clearance to keep; must cover the LiDAR blind zone
    footprint_margin: float = 0.02
    horizon: float = 2.0        # m of path to look ahead
    brake_pwm_max: float = 120  # strongest active-braking (reverse) command
    brake_gain: float = 220.0   # PWM per m/s of speed above the safe speed
    max_pwm: float = 255        # top-speed cap: the driver's throttle is limited to this PWM
    scan_timeout: float = 0.4   # s without a LiDAR scan: crawl only
    scan_stop: float = 1.0      # s without a LiDAR scan: stop
    crawl_speed: float = 0.15   # m/s allowed while the LiDAR is silent
    aware_margin: float = 0.20  # m each side: obstacles this close to the path scale the speed down
    aware_decel: float = 0.7    # m/s^2: gentle slow-down profile for that awareness zone
    v_floor: float = 0.30       # m/s: awareness never slows the car below this on its own
    warn_ttc: float = 2.0       # s: caution below this time-to-collision
    alert_ttc: float = 1.0      # s: warning below this

    @classmethod
    def for_vehicle(cls, p, lidar_min_range=0.2, **kw):
        """Margin big enough that the car stops before an obstacle enters the LiDAR blind zone.

        The LiDAR cannot see closer than lidar_min_range from ITSELF; it sits
        behind the bumper, so the last few cm in front of the bumper are unseen.
        """
        blind = max(0.0, lidar_min_range - (p.front_x - p.lidar_x),
                    lidar_min_range - (p.lidar_x - p.rear_x))
        return cls(margin=blind + 0.03, **kw)

# Warning levels
CLEAR, CAUTION, WARNING, BRAKING = 0, 1, 2, 3


class AEB:
    def __init__(self, params=None, config=None, speed_model=None):
        self.p = params or VehicleParams()
        self.cfg = config or AEBConfig()
        self.model = speed_model or SpeedModel()
        self.v_lidar = 0.0            # closing speed measured from the LiDAR itself
        self._prev = None             # (scan_time, D, direction) of the previous scan
        self._last_scan_time = None

    def _update_closing_speed(self, scan_time, D, direction):
        """Closing speed from how fast the distance-to-contact shrinks between scans.

        Independent of the PWM speed calibration, so it catches a car that is
        faster than the model assumes. Conservative for moving obstacles too.
        """
        if scan_time is None or scan_time == self._last_scan_time:
            return
        self._last_scan_time = scan_time

        prev = self._prev
        if math.isfinite(D) and prev and prev[2] == direction and math.isfinite(prev[1]):
            dt = scan_time - prev[0]
            if 0.0 < dt < 0.5:
                closing = max(0.0, min((prev[1] - D) / dt, 3.0))
                self.v_lidar = 0.5 * self.v_lidar + 0.5 * closing
        else:
            self.v_lidar = 0.0
        self._prev = (scan_time, D, direction)

    def step(self, points, delta, driver_pwm, v_est, scan_time=None, tracks=None):
        """Returns (motor_pwm_to_send, level, info).

        scan_time identifies the current LiDAR scan so its range-rate can be used.
        tracks: moving objects from adas.tracking.Tracker, predicted forward in time.
        """
        cfg = self.cfg
        driver_pwm = max(-cfg.max_pwm, min(cfg.max_pwm, driver_pwm))
        moving_est = abs(v_est) > 0.03

        if moving_est:
            direction = 1 if v_est > 0 else -1
        elif driver_pwm != 0:
            direction = 1 if driver_pwm > 0 else -1
        else:
            self.v_lidar = 0.0
            self._prev = None
            return driver_pwm, CLEAR, {"D": math.inf, "v_safe": math.inf, "ttc": math.inf}

        D_static = travel_distance_to_contact(points, delta, direction, self.p,
                                              horizon=cfg.horizon, margin=cfg.footprint_margin)
        self._update_closing_speed(scan_time, D_static, direction)

        v = max(abs(v_est) if moving_est else 0.0, self.v_lidar)
        moving = v > 0.03

        D_moving, tau_moving, mover_id = math.inf, math.inf, None
        if tracks:
            D_moving, tau_moving, mover_id = moving_object_contact(
                tracks, delta, direction, v, self.p, margin=cfg.footprint_margin)
        D = min(D_static, D_moving)

        # Awareness zone: the same sweep with extra room on each side. Anything in
        # it (a box a pedestrian could step out from, a cone we are passing) scales
        # the allowed speed down smoothly, long before it is in the way.
        wide = cfg.footprint_margin + cfg.aware_margin
        D_aware = travel_distance_to_contact(points, delta, direction, self.p,
                                             horizon=cfg.horizon, margin=wide)
        if tracks:
            D_aware = min(D_aware, moving_object_contact(
                tracks, delta, direction, max(v, 0.3), self.p, margin=wide)[0])

        usable = max(0.0, D - cfg.margin) if math.isfinite(D) else math.inf
        if math.isinf(usable):
            v_safe = math.inf
        else:
            a_tl = cfg.decel * cfg.latency
            v_safe = max(0.0, -a_tl + math.sqrt(a_tl * a_tl + 2.0 * cfg.decel * usable))

        v_aware = math.inf
        if math.isfinite(D_aware):
            v_aware = max(cfg.v_floor, math.sqrt(2.0 * cfg.aware_decel * max(0.0, D_aware - cfg.margin)))

        ttc = D / v if (v > 0.05 and math.isfinite(D)) else math.inf
        info = {"D": D, "D_static": D_static, "D_moving": D_moving, "D_aware": D_aware,
                "mover": mover_id, "v_safe": v_safe, "v_aware": v_aware, "ttc": ttc,
                "direction": direction, "v_lidar": self.v_lidar}

        same_direction = driver_pwm == 0 or (driver_pwm > 0) == (direction > 0)
        out = driver_pwm
        level = CLEAR

        if math.isfinite(v_safe) and v > v_safe + 0.02 and moving:
            # Too fast to stop in the space available: brake actively.
            brake = min(cfg.brake_pwm_max, 40.0 + cfg.brake_gain * (v - v_safe))
            out = -direction * brake
            level = BRAKING
        elif driver_pwm != 0 and same_direction and (math.isfinite(v_safe) or math.isfinite(v_aware)):
            v_cap = min(v_safe, v_aware)
            allowed = self.model.pwm_for_speed(v_cap)
            if abs(driver_pwm) > allowed:
                out = direction * allowed
                level = BRAKING if v_safe <= v_aware else CAUTION

        if level not in (BRAKING,):
            d_stop = v * cfg.latency + v * v / (2.0 * cfg.decel)
            if v > 0.05 and (ttc < cfg.alert_ttc or D < 1.5 * d_stop + cfg.margin):
                level = WARNING
            elif level == CLEAR and v > 0.05 and (ttc < cfg.warn_ttc or D < 2.5 * d_stop + cfg.margin):
                level = CAUTION

        return out, level, info
