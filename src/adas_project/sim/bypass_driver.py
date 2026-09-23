"""Runs the real obstacle-bypass controller (pi/bypass_core.py) and scan-matching odometry
(pi/scanmatch.py) inside the simulator. Same code as on the car: it only sees LiDAR scans."""
import math

import numpy as np

from pi.bypass_core import Bypass
from pi.scanmatch import Odometry, polar_to_xy

from .real_car import curvature_to_steer


class BypassDriver:
    def __init__(self, params, pwm=115):
        self.p = params
        self.ctl = Bypass(pwm=pwm)
        self.odo = Odometry()
        self.pwm = pwm
        self.steer = 0.0
        self.last_scan = -1
        self.state = "CRUISE"
        self.msg = "starting"
        self.pose = (0.0, 0.0, 0.0)
        self.y_ref = 0.0
        self.done = False
        self.trace = []
        self.kappa = 0.0
        self.pwm_cmd = pwm

    def command(self, sim):
        """(steer, pwm) for this simulation step; updates the controller once per new LiDAR scan."""
        if self.done:
            return 0.0, 0.0
        if sim.scan_id != self.last_scan and sim.last_scan is not None:
            self.last_scan = sim.scan_id
            t, angles, ranges, valid = sim.last_scan
            pts = [(math.degrees(a), r) for a, r, ok in zip(angles, ranges, valid) if ok]
            xy = polar_to_xy(pts)
            est = self.odo.update(xy, t, max(sim.car.v, 0.05), self.kappa)
            r = self.ctl.step(est, xy)
            self.pose, self.state, self.msg, self.y_ref = tuple(est), r["state"], r["msg"], r["y_ref"]
            self.trace.append((sim.t, float(est[0]), float(est[1]), float(est[2]), r["state"]))
            if r["pwm"] == 0:
                self.done = True
                self.pwm_cmd = 0.0
                self.kappa = 0.0
                return 0.0, 0.0
            self.kappa = r["kappa"]
            self.steer = curvature_to_steer(r["kappa"], self.p)
            self.pwm_cmd = r["pwm"]
        return self.steer, self.pwm_cmd
