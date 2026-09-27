"""Learned driver intent (RESEARCH.md section 4). Two heads, same features:
  - risk:  P(this driver, left alone, hits something within 2 s) - confidence-aware parallel autonomy (CARPAL)
  - steer: where the stick will be in 0.5 s (kept for the GUI's predicted path)

A small MLP trained on driving data (sim/train_intent_net.py) from what the car can observe:
  stick history      servo offset now and 0.1, 0.2, 0.4, 0.8 s ago, stick activity over the last second
  throttle, speed
  the scene          LiDAR free distance along five candidate steering arcs (servo offsets -20..+20 deg from now)
Output: the change of the servo command over the next 0.5 s. Inference is plain numpy (runs on the Pi).
The prediction is only used for an ATTENTIVE driver (see pi/relay_assists.driver_intent) - a lapsed driver's frozen
stick carries no information, and trusting a prediction for them would be unsafe.
"""
import json
import math

import numpy as np

from .geometry import travel_distance_to_contact

LAGS = (0, 2, 4, 8, 16)            # ticks of 0.05 s
ARCS = (-20, -10, 0, 10, 20)       # servo degrees relative to now
HORIZON_S = 0.5


class DriverProfile:
    """Learns, while the person drives, how close to an obstacle THIS driver usually starts steering away
    (personalised driver model). A driver who always reacts at 0.8 m is not in trouble at 1.0 m; one who always
    reacts at 1.3 m and has not reacted at 1.0 m probably is."""

    def __init__(self, prior=1.0):
        self.onsets = [prior]
        self.quiet_ticks = 0
        self.free_hist = []

    def update(self, servo_hist, free_now):
        self.free_hist = (self.free_hist + [free_now])[-8:]
        moved = len(servo_hist) >= 5 and abs(servo_hist[-1] - servo_hist[-5]) > 4.0
        if moved and self.quiet_ticks >= 12 and self.free_hist[0] < 1.5:
            self.onsets.append(float(self.free_hist[0]))          # free distance just before they reacted
            self.onsets = self.onsets[-15:]
        self.quiet_ticks = 0 if moved else self.quiet_ticks + 1

    @property
    def reaction_distance(self):
        return float(np.median(self.onsets))


def features(servo_hist, pwm, v, pts, params, centre, k_per_deg, dt=0.05, profile=None):
    """Feature vector (or None if the history is too short). servo_hist oldest first, one entry per tick."""
    if len(servo_hist) < LAGS[-1] + 1:
        return None
    s_now = servo_hist[-1]
    f = [(servo_hist[-1 - l] - centre) / 30.0 for l in LAGS]
    f.append(float(np.ptp(servo_hist[-21:])) / 10.0)
    f.append(float(np.ptp(servo_hist[-60:])) / 10.0)              # activity over ~3 s
    quiet = 0
    for a, b in zip(servo_hist[::-1][1:], servo_hist[::-1][:-1]):
        if abs(a - b) > 0.5:
            break
        quiet += 1
    f.append(min(quiet * dt, 3.0) / 3.0)                           # time since the stick last moved
    f += [pwm / 255.0, v]
    for a in ARCS:
        servo = s_now + a
        kappa = -k_per_deg * (servo - centre)
        d = travel_distance_to_contact(pts, math.atan(kappa * params.wheelbase), 1, params, horizon=1.5, margin=0.03) \
            if len(pts) else math.inf
        f.append(min(d, 1.5) / 1.5)
    react = profile.reaction_distance if profile is not None else 1.0
    free_now = f[-3] * 1.5
    f += [react, max(0.0, react - free_now)]                       # this driver's usual reaction distance, overdue
    return np.array(f, float)


class IntentNet:
    def __init__(self, path):
        d = json.load(open(path))
        self.W = [np.array(w) for w in d["W"]]
        self.b = [np.array(b) for b in d["b"]]
        self.mu, self.sd = np.array(d["mu"]), np.array(d["sd"])
        self.act = d.get("activation", "tanh")
        self.kind = d.get("kind", "regressor")
        self.report = d.get("report", {})

    def predict_servo_change(self, x):
        h = (np.asarray(x) - self.mu) / self.sd
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(h) if self.act == "tanh" else np.maximum(h, 0)
        return float(h[0])

    def crash_probability(self, x):
        """Classifier head: probability that the driver, left alone, crashes within 2 s."""
        z = self.predict_servo_change(x)
        return 1.0 / (1.0 + math.exp(-z))

    def kappa_rate(self, x, k_per_deg):
        """Predicted rate of change of path curvature (1/m per s, + = turning more to the left)."""
        return -k_per_deg * self.predict_servo_change(x) / HORIZON_S
