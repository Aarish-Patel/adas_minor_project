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


# ---------------------------------------------------------------- features v3 (sim/twin_intent_data.py, sim/train_intent_torch.py)
# One small vector per 50 ms tick; the models see the last WINDOW ticks. v2 looked only 1.5 m ahead - at full speed
# the car covers ~1.7 m in the 2 s the label looks ahead, so a fast approach looked the same as a slow one - and only
# forward. v3 looks 2.5 m ahead, in the direction of travel (reversing too), and keeps a 1.6 s history.
HORIZON3 = 2.5
WINDOW = 32                                    # ticks (1.6 s)
TICK_DIMS = 12
Z_STICK, Z_THR, Z_V, Z_ARC0, Z_BACK, Z_TTC, Z_STOP, Z_REACT = 0, 1, 2, 3, 8, 9, 10, 11
Z_FREE_NOW = Z_ARC0 + ARCS.index(0)            # forward, current arc
FLAT_LAGS = (0, 1, 2, 4, 8, 12, 16, 24, 31)


def stopping_distance(v):
    av = abs(v)
    return STOP_BASE + av * STOP_REACTION + av * av / (2 * STOP_DECEL)


def travel_direction(v, physical):
    if v > 0.05:
        return 1
    if v < -0.05:
        return -1
    return -1 if physical < 0 else 1


def tick_vector(servo, physical, v, pts, params, centre, k_per_deg, react=1.0):
    """What the car can observe this tick: stick, throttle (+ forward), estimated speed, free distance along five
    steering arcs forward and the current arc backward (m, /2.5), time to contact and free distance in stopping
    distances in the direction of travel, and this driver's usual reaction distance."""
    z = np.zeros(TICK_DIMS)
    z[Z_STICK], z[Z_THR], z[Z_V] = (servo - centre) / 30.0, physical / 255.0, v

    def free(offset, direction):
        if not len(pts):
            return HORIZON3
        kappa = -k_per_deg * (servo + offset - centre)
        d = travel_distance_to_contact(pts, math.atan(kappa * params.wheelbase), direction, params,
                                       horizon=HORIZON3, margin=0.03)
        return min(d, HORIZON3)

    for i, a in enumerate(ARCS):
        z[Z_ARC0 + i] = free(a, 1) / HORIZON3
    z[Z_BACK] = free(0, -1) / HORIZON3
    d = travel_direction(v, physical)
    fd = (z[Z_FREE_NOW] if d > 0 else z[Z_BACK]) * HORIZON3
    z[Z_TTC] = min(fd / max(abs(v), 0.05), 4.0) / 4.0
    z[Z_STOP] = min(fd / stopping_distance(v), 5.0) / 5.0
    z[Z_REACT] = react
    return z


def window_of(zhist):
    """Last WINDOW tick vectors, oldest first; a short history is padded with its first entry."""
    zs = list(zhist)[-WINDOW:]
    if not zs:
        return None
    return np.array([zs[0]] * (WINDOW - len(zs)) + zs)


def flat_features(W):
    """Window (WINDOW x TICK_DIMS) -> the tabular vector for the MLP / trees: the tick vector at a few lags plus
    stick activity, time since the stick moved, throttle easing and speed change, and how overdue this driver's
    usual reaction is."""
    zl = [W[-1 - l, :Z_REACT] for l in FLAT_LAGS]
    stick = W[:, Z_STICK] * 30.0
    quiet = 0
    for a, b in zip(stick[::-1][1:], stick[::-1][:-1]):
        if abs(a - b) > 0.5:
            break
        quiet += 1
    react = W[-1, Z_REACT]
    extra = [react, max(0.0, react - W[-1, Z_FREE_NOW] * HORIZON3), np.ptp(stick[-20:]) / 10.0,
             np.ptp(stick) / 10.0, quiet / (WINDOW - 1), W[-1, Z_THR] - W[-20:, Z_THR].max(),
             W[-1, Z_THR] - W[-20:, Z_THR].min(), W[-1, Z_V] - W[-8, Z_V]]
    return np.concatenate(zl + [np.array(extra)])


def flat_features_batch(Wb):
    """flat_features for a batch of windows (B x WINDOW x TICK_DIMS), vectorised (training); same numbers."""
    B = len(Wb)
    zl = [Wb[:, -1 - l, :Z_REACT] for l in FLAT_LAGS]
    stick = Wb[:, :, Z_STICK] * 30.0
    moved = np.abs(np.diff(stick, axis=1))[:, ::-1] > 0.5            # newest pair first
    quiet = np.where(moved.any(axis=1), moved.argmax(axis=1), WINDOW - 1)
    react = Wb[:, -1, Z_REACT]
    extra = np.column_stack([react, np.maximum(0.0, react - Wb[:, -1, Z_FREE_NOW] * HORIZON3),
                             np.ptp(stick[:, -20:], axis=1) / 10.0, np.ptp(stick, axis=1) / 10.0,
                             quiet / (WINDOW - 1), Wb[:, -1, Z_THR] - Wb[:, -20:, Z_THR].max(axis=1),
                             Wb[:, -1, Z_THR] - Wb[:, -20:, Z_THR].min(axis=1), Wb[:, -1, Z_V] - Wb[:, -8, Z_V]])
    return np.concatenate(zl + [extra], axis=1).reshape(B, -1)


def physics_floor_batch(Wb):
    z = Wb[:, -1]
    active = (np.abs(z[:, Z_V]) >= 0.1) & (z[:, Z_STOP] * 5.0 < 1.0)
    still = np.ptp(Wb[:, -8:, Z_STICK], axis=1) * 30.0 < 2.0
    thr = np.abs(Wb[:, -8:, Z_THR])
    not_easing = (np.abs(z[:, Z_THR]) >= thr.max(axis=1) - 0.03) & (np.abs(z[:, Z_THR]) > 0.05)
    return np.where(active & still & not_easing, 0.95, 0.0)


def physics_floor(W):
    """Lower bound on the risk that no learned model may undercut: the driver is not acting (stick still for 0.4 s,
    throttle not eased) while the free way in the direction of travel is already shorter than the stopping
    distance - left alone, this ends in contact. Returns 0.95 or 0."""
    z = W[-1]
    if abs(z[Z_V]) < 0.1 or z[Z_STOP] * 5.0 >= 1.0:
        return 0.0
    still = np.ptp(W[-8:, Z_STICK]) * 30.0 < 2.0
    thr = W[-8:, Z_THR]
    not_easing = abs(z[Z_THR]) >= abs(thr).max() - 0.03 and abs(z[Z_THR]) > 0.05
    return 0.95 if still and not_easing else 0.0


class IntentNet:
    """Loads either model the trainer writes: "classifier" / "regressor" (MLP weights) or "trees" (gradient-boosted
    trees exported from scikit-learn's HistGradientBoostingClassifier, evaluated here in numpy - no sklearn on the
    car). All trees are walked at once, one depth level per step: ~0.05 ms for 100 trees of depth 3.
    v3 models (sim/train_intent_torch.py): "mlp3" (flat_features) and "gru3" (the window), numpy on the car."""

    def __init__(self, path):
        d = json.load(open(path))
        self.kind = d.get("kind", "regressor")
        self.report = d.get("report", {})
        self.version = 3 if self.kind in ("mlp3", "gru3", "trees3") else 2
        self.temperature = float(d.get("temperature", 1.0))
        if self.version == 3 and self.kind != "trees3":
            self.mu, self.sd = np.array(d["mu"]), np.array(d["sd"])
            if self.kind == "mlp3":
                self.W = [np.array(w) for w in d["W"]]
                self.b = [np.array(b) for b in d["b"]]
            else:
                self.Wih, self.Whh = np.array(d["W_ih"]), np.array(d["W_hh"])
                self.bih, self.bhh = np.array(d["b_ih"]), np.array(d["b_hh"])
                self.Wo, self.bo = np.array(d["W_out"]), np.array(d["b_out"])
                self.hidden = self.Whh.shape[1]
            return
        if self.kind in ("trees", "trees3"):
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

    # ---------------------------------------------------------------- v3
    def logit3(self, W):
        """Window (WINDOW x TICK_DIMS) -> calibrated logit."""
        if self.kind == "trees3":
            return self._trees_raw(flat_features(W)) / self.temperature
        if self.kind == "mlp3":
            h = (flat_features(W) - self.mu) / self.sd
            for i, (Wm, b) in enumerate(zip(self.W, self.b)):
                h = h @ Wm + b
                if i < len(self.W) - 1:
                    h = np.maximum(h, 0.0)
            return float(h[0]) / self.temperature
        x = (W - self.mu) / self.sd                      # GRU (PyTorch gate order r, z, n)
        H = self.hidden
        h = np.zeros(H)
        gi_all = x @ self.Wih.T + self.bih
        for gi in gi_all:
            gh = self.Whh @ h + self.bhh
            r = 1.0 / (1.0 + np.exp(-(gi[:H] + gh[:H])))
            u = 1.0 / (1.0 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
            n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
            h = (1.0 - u) * n + u * h
        return float(self.Wo @ h + self.bo) / self.temperature

    def risk(self, W, floor=True):
        """v3: P(the driver, left alone, hits or nearly hits something within 2 s), never below the physics floor."""
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, self.logit3(W)))))
        return max(p, physics_floor(W)) if floor else p

    def kappa_rate(self, x, k_per_deg):
        """Predicted rate of change of path curvature (1/m per s, + = turning more to the left)."""
        return -k_per_deg * self.predict_servo_change(x) / HORIZON_S
