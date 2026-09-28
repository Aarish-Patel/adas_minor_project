"""Online steering calibration (TODO N2): the servo's straight-ahead angle and the steering gain, estimated while driving.

Why: the car's steering is not what a one-off calibration said - the servo centre wanders with the linkage, the
battery and vibration, and the gain differs a little from run to run (the digital twin randomises both). Every
predicted path (the brake gate's, the nudge's, the evasive planner's) is built from "commanded servo -> curvature", so
a 3 deg centre error is 0.2 1/m: 0.4 m sideways after 2 m. The LiDAR measures the real yaw rate (RF2O range flow in the
speed EKF), so the mapping can be re-estimated continuously.

Measurement: the scan-matched pose (pi/relay_assists.RelaySpeed.pose), NOT the filter's yaw rate - the EKF's yaw
is dominated by its own steering model, and raw range-flow yaw carries a bias (measured +1.4 deg/s in the twin).

Model (bicycle car, small angles): the path curvature is kappa = w / v = -k * (servo - centre) = b - k * servo, with
b = k * centre. Recursive least squares (Ljung, System Identification; Astrom & Wittenmark, Adaptive Control) with an
exponential forgetting factor on theta = [b, k] from samples (servo, w / v):
  - a sample counts only when |v| > V_MIN and the steering was steady (it is the servo history the wheels really
    followed: the command from DELAY_S ago, the fitted command delay)
  - a prior (the calibrated centre and gain) enters as a first pseudo-measurement and keeps the estimate from drifting
    when the driver goes straight for a long time (then only the centre is observable and only it moves)
  - the estimate is clipped to +-MAX_CENTRE_DEG / +-MAX_GAIN of the prior and changes slowly (it is a bias, not a signal)
"""
import collections
import math

import numpy as np


class OnlineSteering:
    V_MIN = 0.15
    WINDOW_M = 0.35
    DELAY_S = 0.12
    STEADY_DEG = 4.0
    MAX_CENTRE_DEG = 4.5
    MAX_GAIN = 0.25

    def __init__(self, centre, k, forget=0.995, prior_weight=4.0):
        self.centre0, self.k0 = float(centre), float(k)
        self.forget = forget
        # theta = [b, k]; regressors [1, -servo]: kappa = b - k * servo  ->  use x = [1, -servo], theta = [b, k]
        self.theta = np.array([k * centre, k], float)
        self.P = np.diag([50.0, 50.0]) * 1.0
        self.hist = collections.deque(maxlen=64)         # (t, servo)
        self._anchor = None
        self.n = 0
        self._seed_prior(prior_weight)

    def _seed_prior(self, w):
        """The calibrated values as a few pseudo-measurements at two steering angles."""
        for servo in (self.centre0 - 20.0, self.centre0 + 20.0):
            for _ in range(int(w)):
                self._rls(servo, -self.k0 * (servo - self.centre0), prior=True)

    def _rls(self, servo, kappa, prior=False):
        x = np.array([1.0, -servo])
        Px = self.P @ x
        g = Px / (self.forget + x @ Px)
        self.theta = self.theta + g * (kappa - x @ self.theta)
        self.P = (self.P - np.outer(g, Px)) / self.forget
        if not prior:
            self.n += 1

    def command(self, t, servo):
        self.hist.append((t, float(servo)))

    def _servo_then(self, t):
        for tt, sv in reversed(self.hist):
            if tt <= t:
                return sv
        return None

    def update(self, t, v, pose):
        """One sample per WINDOW_M of travel from the scan-matched world pose (x, y, heading): the curvature the car
        really drove is the heading change over the distance (a differential measurement: constant heading errors and
        the yaw-rate bias of range flow drop out; ICP heading is good to ~2 deg over 3 m), paired with the steering the
        wheels followed (the command from DELAY_S ago, averaged over the window). Only steady steering counts."""
        if abs(v) < self.V_MIN or pose is None:
            return
        x, y, th = pose
        s_now = self._servo_then(t - self.DELAY_S)
        if s_now is None:
            return
        if self._anchor is None:
            self._anchor = (x, y, th, [s_now])
            return
        ax, ay, ath, servos = self._anchor
        servos.append(s_now)
        ds = math.hypot(x - ax, y - ay)
        if ds < self.WINDOW_M:
            return
        self._anchor = (x, y, th, [s_now])
        if max(servos) - min(servos) > self.STEADY_DEG or ds > 3 * self.WINDOW_M:
            return                                       # the wheels were moving, or a gap in the data
        dth = math.remainder(th - ath, 2 * math.pi) * (1 if v > 0 else -1)
        kappa = dth / ds
        if abs(kappa) > 3.0:
            return
        self._rls(float(np.mean(servos)), kappa)

    @property
    def k(self):
        lo, hi = self.k0 * (1 - self.MAX_GAIN), self.k0 * (1 + self.MAX_GAIN)
        return float(min(hi, max(lo, self.theta[1])))

    @property
    def centre(self):
        c = self.theta[0] / max(self.theta[1], 1e-6)
        return float(min(self.centre0 + self.MAX_CENTRE_DEG, max(self.centre0 - self.MAX_CENTRE_DEG, c)))

    def status(self):
        return {"centre": round(self.centre, 2), "k": round(self.k, 4), "samples": self.n,
                "centre_shift_deg": round(self.centre - self.centre0, 2), "gain_shift_pct": round(100 * (self.k / self.k0 - 1), 1)}
