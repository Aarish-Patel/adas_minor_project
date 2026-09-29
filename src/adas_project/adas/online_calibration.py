"""On-the-go adaptation of the crash predictor once a person starts driving (TODO W6).

The intent model is trained offline on a twin (randomised, but still a twin) and then meets a real driver and a real car. Their
base rate of trouble, the car's grip and the LiDAR noise differ from anything in the training data, so the model's probabilities
drift out of calibration (twin ablation: on unseen cars the calibration error was 0.12 vs 0.02) - a "30 % risk" no longer happens
30 % of the time. Two things adapt while driving:

  1. DriverProfile (adas/intent_net.py, existing): the driver's usual reaction distance, learned from when they start steering.
  2. OnlineCalibrator (this file): a two-parameter logistic recalibration  p' = sigmoid(a * logit(p) + b)  (Platt scaling, Platt
     1999), tracked online with a Kalman-filter logistic regression (an extended Kalman filter on (a, b); Kalman-filter online
     logistic regression, e.g. Penny & Roberts 1999; Bayesian online learning, Opper & Winther 1998) with a forgetting term so it
     follows a driver whose behaviour changes.

Feedback the CAR can observe (no ground truth needed): a prediction made at time t is resolved 2 s later - outcome 1 if a physical
safety event happened in between (the path brake latched / held the car, or a near miss was measured by the LiDAR), else 0.
That is what the training label "contact or near miss within 2 s" was for, seen from inside the car. (An intervention that prevents
the event still counts as the event: the risk was real.)

    cal = OnlineCalibrator()
    cal.observe(t, logit)          # every ~0.25 s with the model's raw logit
    cal.event(t)                   # whenever a physical safety event happens
    p = cal.probability(logit)     # the recalibrated risk (identity until enough feedback arrives)
"""
import collections
import math

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


class OnlineCalibrator:
    def __init__(self, horizon_s=2.0, prior_a=(1.0, 0.25), prior_b=(0.0, 0.7), process=(2e-4, 6e-4), min_updates=30):
        self.horizon = horizon_s
        self.theta = np.array([prior_a[0], prior_b[0]], float)
        self.P = np.diag([prior_a[1] ** 2, prior_b[1] ** 2])
        self.Q = np.diag(process)
        self.pending = collections.deque()          # (t, logit)
        self.events = collections.deque()           # times of safety events
        self.n = 0
        self.min_updates = min_updates

    def observe(self, t, logit):
        self.pending.append((t, float(logit)))
        self._resolve(t)

    def event(self, t):
        self.events.append(t)

    def _resolve(self, now):
        while self.pending and self.pending[0][0] <= now - self.horizon:
            t0, x = self.pending.popleft()
            y = 1.0 if any(t0 <= e <= t0 + self.horizon for e in self.events) else 0.0
            self._update(x, y)
        while self.events and self.events[0] < now - 2 * self.horizon:
            self.events.popleft()

    def _update(self, x, y):
        a, b = self.theta
        p = _sigmoid(a * x + b)
        w = max(p * (1.0 - p), 1e-4)
        H = np.array([[x * w, w]])                                   # d p / d theta
        self.P = self.P + self.Q                                     # forgetting: (a, b) may change with the driver
        S = float((H @ self.P @ H.T).item()) + w                              # innovation variance (Bernoulli noise ~ p(1-p))
        K = (self.P @ H.T / S).ravel()
        self.theta = self.theta + K * (y - p)
        self.theta[0] = min(3.0, max(0.3, self.theta[0]))            # keep the slope sane
        self.theta[1] = min(4.0, max(-4.0, self.theta[1]))
        self.P = self.P - np.outer(K, H @ self.P)
        self.n += 1

    @property
    def active(self):
        return self.n >= self.min_updates

    def probability(self, logit):
        """The recalibrated probability; the model's own until `min_updates` outcomes have been seen."""
        if not self.active:
            return _sigmoid(logit)
        return _sigmoid(self.theta[0] * logit + self.theta[1])
