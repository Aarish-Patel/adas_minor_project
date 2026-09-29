"""Runs the virtual car, LiDAR and the ADAS together, timed like the real system.

  LiDAR scans at 10 Hz. The controller loop runs at 50 Hz using the latest scan,
  the driver's command and the speed estimate, and its output reaches the car
  after a link delay. The ADAS only ever sees what it would see on the real car:
  LiDAR points, its own commands, and a PWM-based speed estimate.
"""

import math
from collections import deque

from adas.lidar_utils import scan_to_points
from adas.pipeline import AdasPipeline
from adas.vehicle_params import VehicleParams

from .car_sim import Dynamics, SimCar
from .lidar_sim import LidarSim
from .lane_sensor import LaneSensor
from .marker_sensor import MarkerSensor

DT = 0.01
CONTROL_PERIOD = 0.02
LINK_DELAY = 0.03


class Simulator:
    def __init__(self, world, start_pose=(0.0, 0.0, 0.0), adas_on=True, params=None,
                 dynamics=None, adas_speed_model=None, aeb_config=None, lidar=None, seed=0,
                 logger=None):
        self.p = params or VehicleParams()
        self.world = world
        self.dyn = dynamics or Dynamics()
        self.car = SimCar(*start_pose, self.p, self.dyn)
        self.lidar = lidar or LidarSim(seed=seed)
        self.adas = AdasPipeline(self.p, aeb_config, adas_speed_model, self.lidar.min_range)
        self.marker_sensor = MarkerSensor(seed=seed)
        self.lane_sensor = LaneSensor(seed=seed)
        self.lidar_enabled = True
        self.camera_enabled = True
        self.adas_mode = adas_on if isinstance(adas_on, str) else ('active' if adas_on else 'off')
        self.logger = logger
        self.link_delay = LINK_DELAY

        self.t = 0.0
        self.next_scan = 0.0
        self.next_control = 0.0
        self.queue = deque()
        self.applied_steer = 0.0
        self.applied_pwm = 0.0
        self.scan_pose = start_pose
        self.scan_id = 0
        self.last_scan = None
        self.marker_serial = 0
        self.last_markers = []
        self.driver_steer = 0.0
        self.driver_pwm = 0.0

        self.min_clearance = math.inf
        self.max_level = 0
        self.brake_time = 0.0
        self.distance = 0.0
        self.log = []

    @property
    def adas_on(self):
        return self.adas_mode == "active"

    @adas_on.setter
    def adas_on(self, value):
        self.adas_mode = "active" if value else "off"

    # views the viewers use
    @property
    def points(self):
        return self.adas.points

    @property
    def tracker(self):
        return self.adas.tracker

    @property
    def level(self):
        return self.adas.level

    @property
    def info(self):
        return self.adas.info

    @property
    def pwm_out(self):
        return self.adas.pwm_out

    @property
    def estimator(self):
        return self.adas.estimator

    def step(self, driver_steer, driver_pwm):
        c = self.car
        self.driver_steer, self.driver_pwm = driver_steer, driver_pwm

        if self.t >= self.next_scan and not c.collided and self.lidar_enabled:
            angles, ranges, valid = self.lidar.scan(self.world, c.x, c.y, c.theta, self.p)
            self.adas.on_scan(scan_to_points(angles, ranges, valid, self.p), self.t)
            self.last_scan = (self.t, angles, ranges, valid)
            self.scan_pose = (c.x, c.y, c.theta)
            self.scan_id += 1
            self.next_scan += self.lidar.period
        elif self.t >= self.next_scan and not self.lidar_enabled:
            self.next_scan += self.lidar.period

        if not c.collided and self.camera_enabled:
            obs = self.marker_sensor.sense(self.world, c, self.t)
            if obs is not None:
                self.adas.on_markers(obs, self.t)
                self.last_markers = obs
                self.marker_serial += 1

        if not c.collided and self.camera_enabled:
            lane_pts = self.lane_sensor.sense(self.world, c, self.t)
            if lane_pts is not None:
                self.adas.on_lane_points(lane_pts, self.t)

        if self.t >= self.next_control:
            pwm_out, level, _ = self.adas.on_control(CONTROL_PERIOD, driver_steer, driver_pwm,
                                                     self.adas_mode)
            self.queue.append((self.t + self.link_delay, self.adas.steer_out, pwm_out))
            self.next_control += CONTROL_PERIOD
            self.max_level = max(self.max_level, level)
            if self.logger:
                self.logger.log(self.t, driver_steer, driver_pwm, self.adas)

        while self.queue and self.queue[0][0] <= self.t:
            _, self.applied_steer, self.applied_pwm = self.queue.popleft()

        c.step(DT, self.applied_steer, self.applied_pwm, self.world)
        self.world.update(DT)
        self.t += DT
        self.distance += abs(c.v) * DT

        self.min_clearance = min(self.min_clearance, c.clearance)
        if self.level == 3:
            self.brake_time += DT
        self.log.append((self.t, c.x, c.y, c.theta, c.v, self.estimator.v,
                         self.level, driver_pwm, self.pwm_out, c.clearance))

    def run(self, driver, duration, stop_when_stopped=True):
        """driver(t, sim) -> (steer, pwm). Returns self for inspection."""
        settled = 0.0
        while self.t < duration:
            steer, pwm = driver(self.t, self)
            self.step(steer, pwm)
            if self.car.collided:
                break
            if stop_when_stopped and self.t > 0.5 and abs(self.car.v) < 0.005 and pwm != 0 \
                    and self.pwm_out == 0:
                settled += DT
                if settled > 0.5:
                    break
            else:
                settled = 0.0
        return self
