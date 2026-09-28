"""Speed and yaw rate of the car: an extended Kalman filter with the fitted car model as the process and the LiDAR
range flow (adas/rf2o.py) as the measurement - the fusion RESEARCH.md section 3 describes.

State   x = [v, w]   (forward speed of the rear axle m/s, yaw rate rad/s, + = left)
Process v -> first-order approach to the steady speed of the (delayed) throttle, coasting decel with no throttle;
        w -> v * curvature of the (delayed) steering command.  Model error enters as process noise.
Measure the LiDAR sits lidar_x ahead of the rear axle, so range flow sees vx = v, vy = w * lidar_x, w = w.
The prediction runs every control tick (lag-free); each scan corrects it.
"""
import math

import numpy as np


class SpeedEKF:
    def __init__(self, v_max, deadband, tau=0.1, delay=0.12, coast_decel=1.2, k_curv_per_deg=0.0656,
                 servo_centre=87.0, lidar_x=0.12, q_v=0.6, q_w=1.5, r_scale=4.0, r_floor=(0.02, 0.02, 0.05),
                 accel_limits=(-4.0, 3.0)):
        self.v_max, self.deadband, self.tau, self.delay = v_max, deadband, max(tau, 0.03), delay
        self.coast = coast_decel
        self.k, self.centre, self.lx = k_curv_per_deg, servo_centre, lidar_x
        self.q_v, self.q_w = q_v, q_w                  # process noise densities (m/s per sqrt s, rad/s per sqrt s)
        self.r_scale, self.r_floor = r_scale, np.array(r_floor)
        self.a_min, self.a_max = accel_limits          # the motor cannot change speed faster (fitted twin limits)
        self.rejects = 0
        self.x = np.zeros(2)
        self.P = np.diag([0.05, 0.1])
        self.cmds = []                                 # (t, pwm, servo) history for the command delay
        self.t = None
        self.innov = None

    def v_steady(self, pwm):
        mag = max(0.0, (abs(pwm) - self.deadband) / (255.0 - self.deadband))
        return math.copysign(self.v_max * min(mag, 1.0), pwm) if pwm else 0.0

    def command(self, t, pwm, servo):
        """The physical throttle (+ forward) and servo command sent at time t."""
        self.cmds.append((t, float(pwm), float(servo)))
        cut = t - self.delay - 1.0
        while len(self.cmds) > 2 and self.cmds[1][0] < cut:
            self.cmds.pop(0)

    def _cmd_at(self, t):
        u, s = 0.0, self.centre
        for tc, pc, sc in self.cmds:
            if tc <= t:
                u, s = pc, sc
            else:
                break
        return u, s

    def predict(self, t):
        if self.t is None:
            self.t = t
            return self.x
        dt = t - self.t
        if dt <= 0:
            return self.x
        self.t = t
        u, s = self._cmd_at(t - self.delay)
        v, w = self.x
        kappa = -self.k * (s - self.centre)            # servo above centre = turning right
        if u == 0:
            dv = -math.copysign(min(abs(v), self.coast * dt), v)
            a = 1.0
        else:
            a = math.exp(-dt / self.tau)
            dv = (self.v_steady(u) - v) * (1 - a)
            if not self.a_min * dt <= dv <= self.a_max * dt:
                dv = min(self.a_max * dt, max(self.a_min * dt, dv))
                a = 1.0                                # saturated: the change no longer depends on v
        v2 = v + dv
        # yaw rate relaxes to v * kappa at the same rate the speed does
        w2 = w + (v2 * kappa - w) * (1 - math.exp(-dt / 0.08))
        F = np.array([[a, 0.0], [kappa * (1 - math.exp(-dt / 0.08)) * a, math.exp(-dt / 0.08)]])
        self.x = np.array([v2, w2])
        self.P = F @ self.P @ F.T + np.diag([self.q_v ** 2 * dt, self.q_w ** 2 * dt])
        return self.x

    def correct(self, t, meas):
        """meas: (vx, vy, w, cov, n) from RangeFlow.update, the average over the last scan interval."""
        self.predict(t)
        vx, vy, w, cov, _n = meas
        z = np.array([vx, vy, w])
        H = np.array([[1.0, 0.0], [0.0, self.lx], [0.0, 1.0]])
        R = np.diag(np.maximum(np.diag(cov) * self.r_scale, self.r_floor ** 2))
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        # gate outliers (a bad scan match must not throw the speed): Mahalanobis distance, 3 dof, 99.9 %. Two
        # rejections in a row mean the MODEL is what is wrong (e.g. the car pushed, stuck, braking harder or softer
        # than modelled): then accept, with the state uncertainty inflated so the measurement takes over.
        d2 = float(y @ np.linalg.solve(S, y))
        self.innov = d2
        if d2 > 16.3:
            self.rejects += 1
            if self.rejects < 2:
                return self.x
            self.P = self.P + np.diag([0.25, 0.5])
            S = H @ self.P @ H.T + R
        self.rejects = 0
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P
        return self.x

    @property
    def v(self):
        return float(self.x[0])

    @property
    def w(self):
        return float(self.x[1])
