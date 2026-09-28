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
PWM_LAGS = (4, 8, 16)              # throttle 0.2, 0.4, 0.8 s ago (ticks)
HORIZON_S = 0.5
FREE_NOW = len(LAGS) + 3 + 2 + ARCS.index(0)   # index of the free distance on the current arc (/1.5 m)
# the stopping distance used for the physics feature (pi/path_gate.py: BASE + v*REACTION + v^2/2*DECEL)
STOP_BASE, STOP_REACTION, STOP_DECEL = 0.05, 0.20, 4.0


def free_now(f):
    """Free distance (m) along the current arc from a feature vector."""
    return float(f[FREE_NOW]) * 1.5


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


def features(servo_hist, pwm, v, pts, params, centre, k_per_deg, dt=0.05, profile=None, pwm_hist=None):
    """Feature vector (or None if the history is too short). servo_hist / pwm_hist oldest first, one entry per tick.
    v2 adds what shows a driver's awareness besides the stick: the throttle history (easing off before an obstacle)
    and two physics features - time to contact and free distance in stopping distances on the current arc."""
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
    fn = f[FREE_NOW] * 1.5
    f += [react, max(0.0, react - fn)]                             # this driver's usual reaction distance, overdue
    ph = list(pwm_hist) if pwm_hist is not None and len(pwm_hist) > PWM_LAGS[-1] else [pwm] * (PWM_LAGS[-1] + 1)
    f += [ph[-1 - l] / 255.0 for l in PWM_LAGS]
    f.append((pwm - max(ph[-21:])) / 255.0)                        # easing off over the last second (<= 0)
    av = abs(v)
    f.append(min(fn / max(av, 0.05), 3.0) / 3.0)                   # time to contact on the current arc
    stop = STOP_BASE + av * STOP_REACTION + av * av / (2 * STOP_DECEL)
    f.append(min(fn / stop, 5.0) / 5.0)                            # free distance in stopping distances
    return np.array(f, float)


class IntentNet:
    """Loads either model the trainer writes: "classifier" / "regressor" (MLP weights) or "trees" (gradient-boosted
    trees exported from scikit-learn's HistGradientBoostingClassifier, evaluated here in numpy - no sklearn on the
    car). All trees are walked at once, one depth level per step: ~0.05 ms for 100 trees of depth 3."""

    def __init__(self, path):
        d = json.load(open(path))
        self.kind = d.get("kind", "regressor")
        self.report = d.get("report", {})
        if self.kind == "trees":
            self.feat = np.array(d["feature"], int)
            self.thr = np.array(d["threshold"], float)
            self.left = np.array(d["left"], int)
            self.right = np.array(d["right"], int)
            self.leaf = np.array(d["leaf"], bool)
            self.value = np.array(d["value"], float)
            self.baseline = float(d["baseline"])
            self.depth = int(d["max_depth"])
            self._rows = np.arange(len(self.feat))
            return
        self.W = [np.array(w) for w in d["W"]]
        self.b = [np.array(b) for b in d["b"]]
        self.mu, self.sd = np.array(d["mu"]), np.array(d["sd"])
        self.act = d.get("activation", "tanh")

    def _trees_raw(self, x):
        x = np.asarray(x, float)
        node = np.zeros(len(self.feat), int)
        r = self._rows
        for _ in range(self.depth + 1):
            f = self.feat[r, node]
            go_left = x[f] <= self.thr[r, node]
            nxt = np.where(go_left, self.left[r, node], self.right[r, node])
            node = np.where(self.leaf[r, node], node, nxt)
        return self.baseline + float(self.value[r, node].sum())

    def predict_servo_change(self, x):
        if self.kind == "trees":
            return self._trees_raw(x)
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
