"""The whole on-car ADAS chain in one object, identical in the simulator and on the Pi.

    pipeline.on_scan(points, t)                 every LiDAR scan (~10 Hz)
    pipeline.on_control(dt, steer, pwm)         every control tick (~50 Hz) -> motor PWM to send
"""

import math

import numpy as np

from .aeb import AEB, AEBConfig, SpeedEstimator, SpeedModel
from .acc import FollowController
from .isa import SpeedAdaptation
from .lane import LaneKeeper, fit_lane
from .memory import ObstacleMemory
from .parking import ParkingController
from .tracking import Tracker
from .vehicle_params import VehicleParams, steer_to_delta


class AdasPipeline:
    def __init__(self, params=None, aeb_config=None, speed_model=None, lidar_min_range=0.2):
        self.p = params or VehicleParams()
        self.model = speed_model or SpeedModel()
        self.aeb = AEB(self.p, aeb_config or AEBConfig.for_vehicle(self.p, lidar_min_range), self.model)
        self.estimator = SpeedEstimator(self.model)
        self.tracker = Tracker()
        self.parking = ParkingController(self.p, self.model)
        self.acc = FollowController(self.p, self.model)
        self.isa = SpeedAdaptation(self.p)
        self.lane = LaneKeeper(self.p)
        self._lane_est = None
        self.t_now = 0.0
        self.marker_obs = []
        self._new_obs = []
        self.steer_out = 0.0
        self.memory = ObstacleMemory(self.p, blind_radius=lidar_min_range + 0.04)

        self.points = np.empty((0, 2))
        self.scan_time = 0.0
        self.scan_age = 0.0
        self.fault = None
        self.last_steer = 0.0
        self.pwm_out = 0.0
        self.level = 0
        self.info = {"D": math.inf, "v_safe": math.inf, "ttc": math.inf}

    @property
    def tracks(self):
        return self.tracker.tracks

    def on_scan(self, points, t):
        self.points = points
        self.scan_time = t
        self.scan_age = 0.0
        delta = steer_to_delta(self.last_steer, self.p)
        omega = self.estimator.v * math.tan(delta) / self.p.wheelbase
        self.tracker.update(points, t, self.estimator.v, omega)
        movers = [(tr.pos[0], tr.pos[1], tr.radius) for tr in self.tracker.tracks if tr.moving]
        self.memory.add_scan(points, movers)

    def on_lane_points(self, points, t):
        """Floor points of lane lines from a camera frame (see adas.lane)."""
        self._lane_est = fit_lane(points, self.lane.lane_width, t=t)

    def on_markers(self, observations, t):
        """A new camera frame's marker detections (list of adas.markers.MarkerObs)."""
        self.marker_obs = observations
        self._new_obs = observations

    def on_control(self, dt, steer, driver_pwm, mode="active"):
        """mode: "active" (may limit / brake), "advisory" (analyse and warn only), "off".

        The steering to send is in self.steer_out (the driver's, unless parking is driving).
        """
        self.t_now += dt
        if not self.parking.active:
            steer, lane_level = self.lane.update(self.t_now, self._lane_est, steer, self.estimator.v)
            self._lane_est = None
        else:
            lane_level = 0
        if self.parking.active:
            front_gap = self.info.get("D_static", math.inf)
            cmd = self.parking.update(dt, self._new_obs, self.estimator.v, self.steer_out, front_gap)
            if cmd is not None:
                steer, driver_pwm = cmd
        # traffic-sign speed adaptation and follow-the-leader cap the driver's throttle
        v_cap = self.isa.update(dt, self._new_obs, self.estimator.v, self.steer_out)
        if driver_pwm > 0 and math.isfinite(v_cap):
            driver_pwm = min(driver_pwm, self.model.pwm_for_speed(v_cap))
        acc_cap = self.acc.limit(self.tracks, self.estimator.v, driver_pwm)
        if acc_cap is not None:
            driver_pwm = min(driver_pwm, acc_cap)

        self._new_obs = []
        self.steer_out = steer
        self.last_steer = steer
        delta = steer_to_delta(steer, self.p)
        self.memory.advance(dt, self.estimator.v, delta)

        if mode == "off":
            self.pwm_out, self.level = driver_pwm, 0
        else:
            blind = self.memory.blind_points()
            pts = np.vstack([self.points, blind]) if len(blind) else self.points
            out, self.level, self.info = self.aeb.step(
                pts, delta, driver_pwm, self.estimator.v, self.scan_time, self.tracks)
            self.pwm_out = out if mode == "active" else driver_pwm

            # Watchdog: without fresh LiDAR scans the ADAS is blind, so crawl and then stop.
            self.scan_age += dt
            cfg = self.aeb.cfg
            self.fault = None
            if self.scan_age > cfg.scan_timeout:
                self.fault = "LiDAR lost"
                self.level = max(self.level, 2)
                if mode == "active":
                    if self.scan_age > cfg.scan_stop:
                        self.pwm_out = 0.0
                        self.level = 3
                    else:
                        cap = self.model.pwm_for_speed(cfg.crawl_speed)
                        self.pwm_out = math.copysign(min(abs(self.pwm_out), cap), self.pwm_out)

        self.level = max(self.level, lane_level) if mode != "off" else self.level
        self.estimator.update(dt, self.pwm_out)
        return self.pwm_out, self.level, self.info
