"""Driver-intent features, logging, self-labelling and the model.

Intent = what the driver will do in the next ~0.8 s: keep straight, steer
left, steer right, or lift off / brake. It is inferred from the driver's own
recent inputs plus what the LiDAR sees, and learned from logged driving: the
label for any moment is simply what the driver actually did next, so no
manual labelling is needed.

    IntentLogger   records one row of features per control tick (20 Hz)
    make_labels    adds the future-behaviour label to logged rows
    IntentModel    trains on labelled rows, predicts class probabilities
"""

import csv
import math

import numpy as np

CLASSES = ["straight", "left", "right", "brake"]
N_SECTORS = 12
SECTOR_MAX = 2.5
HORIZON = 0.8

FEATURES = (["steer", "steer_rate", "pwm", "pwm_rate", "v", "D", "ttc", "level",
             "lat_left", "lat_right", "n_movers", "mover_lat_v", "gap_bearing",
             "steer_d2", "steer_d4", "steer_d8", "steer_mean", "pwm_d2", "pwm_d8", "pwm_max", "steer_std", "pwm_std"]
            + [f"s{i}" for i in range(N_SECTORS)])


def sector_ranges(points, p, n=N_SECTORS, max_range=SECTOR_MAX):
    """Minimum LiDAR range in n equal sectors across the front 180 deg, right to left."""
    out = np.full(n, max_range)
    if len(points) == 0:
        return out
    dx, dy = points[:, 0] - p.lidar_x, points[:, 1] - p.lidar_y
    ang = np.arctan2(dy, dx)
    r = np.minimum(np.hypot(dx, dy), max_range)
    m = (ang >= -math.pi / 2) & (ang < math.pi / 2)
    idx = np.minimum(((ang[m] + math.pi / 2) / math.pi * n).astype(int), n - 1)
    np.minimum.at(out, idx, r[m])
    return out


def lateral_clearance(points, p, reach=1.0, cap=1.0):
    """Free space beside the car's body, (left, right), for obstacles within `reach` ahead."""
    left = right = cap
    if len(points):
        m = (points[:, 0] > 0.0) & (points[:, 0] < reach)
        if m.any():
            y = points[m, 1]
            yl, yr = y[y > 0], -y[y < 0]
            if len(yl):
                left = min(cap, max(0.0, float(yl.min()) - p.width / 2))
            if len(yr):
                right = min(cap, max(0.0, float(yr.min()) - p.width / 2))
    return left, right


def extract_features(t, steer, pwm, pipeline, hist):
    """One feature dict. `hist` is a list of past (t, steer, pwm), oldest first (may be empty)."""
    p = pipeline.p
    prev = hist[-1] if hist else None
    dt = (t - prev[0]) if prev else 0.0
    steer_rate = (steer - prev[1]) / dt if dt > 1e-6 else 0.0
    pwm_rate = (pwm - prev[2]) / dt if dt > 1e-6 else 0.0

    def back(n, idx):
        """value n samples ago (falls back to the oldest known)"""
        if not hist:
            return steer if idx == 1 else pwm
        return hist[max(0, len(hist) - n)][idx]

    info = pipeline.info
    D = info.get("D_static", info.get("D", math.inf))
    D = min(D, 3.0) if math.isfinite(D) else 3.0
    ttc = info.get("ttc", math.inf)
    ttc = min(ttc, 5.0) if math.isfinite(ttc) else 5.0

    points = pipeline.points
    sectors = sector_ranges(points, p)
    left, right = lateral_clearance(points, p)

    movers = [tr for tr in pipeline.tracks if tr.moving]
    mover_lat_v = 0.0
    if movers:
        near = min(movers, key=lambda tr: math.hypot(*tr.pos))
        mover_lat_v = near.vel_obj[1]

    centres = (np.arange(N_SECTORS) + 0.5) / N_SECTORS * 2.0 - 1.0     # -1 right .. +1 left
    gap = float(centres[int(np.argmax(sectors))])

    steers = [h[1] for h in hist[-10:]] or [steer]
    pwms = [h[2] for h in hist[-20:]] or [pwm]

    row = {
        "t": round(t, 3), "steer": steer, "steer_rate": steer_rate / 5.0,
        "pwm": pwm / 255.0, "pwm_rate": pwm_rate / 500.0, "v": pipeline.estimator.v,
        "D": D / 3.0, "ttc": ttc / 5.0, "level": pipeline.level / 3.0,
        "lat_left": left, "lat_right": right, "n_movers": float(len(movers)),
        "mover_lat_v": mover_lat_v, "gap_bearing": gap,
        "steer_d2": steer - back(2, 1), "steer_d4": steer - back(4, 1), "steer_d8": steer - back(8, 1),
        "steer_mean": float(np.mean(steers)),
        "pwm_d2": (pwm - back(2, 2)) / 255.0, "pwm_d8": (pwm - back(8, 2)) / 255.0,
        "pwm_max": max(pwms) / 255.0,
        "steer_std": float(np.std(steers)) if len(steers) > 2 else 0.0,
        "pwm_std": float(np.std(pwms)) / 255.0 if len(pwms) > 2 else 0.0,
    }
    for i, sec in enumerate(sectors):
        row[f"s{i}"] = sec / SECTOR_MAX
    return row


class IntentLogger:
    """Records feature rows at `rate_hz`; optionally streams them to a CSV file."""

    def __init__(self, path=None, rate_hz=20.0, keep=True):
        self.path = path
        self.period = 1.0 / rate_hz
        self.keep = keep
        self.rows = []
        self._next = 0.0
        self._hist = []
        self._file = None
        self._writer = None
        self.active = True

    def log(self, t, steer, pwm, pipeline):
        if not self.active or t < self._next:
            return
        self._next = t + self.period
        row = extract_features(t, steer, pwm, pipeline, self._hist)
        row["pwm_out"] = pipeline.pwm_out / 255.0
        self._hist.append((t, steer, pwm))
        del self._hist[:-20]
        if self.keep:
            self.rows.append(row)
        if self.path:
            if self._file is None:
                self._file = open(self.path, "w", newline="", encoding="utf-8")
                self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
                self._writer.writeheader()
            self._writer.writerow(row)

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


def make_labels(rows, horizon=HORIZON, steer_thr=0.25):
    """Label each row with what the driver did over the next `horizon` seconds.

    Rows are dicts in time order. The last `horizon` seconds get label None.
    """
    n = len(rows)
    ts = np.array([r["t"] for r in rows])
    steer = np.array([r["steer"] for r in rows])
    pwm = np.array([r["pwm"] for r in rows])
    labels = [None] * n

    j = 0
    for i in range(n):
        end = ts[i] + horizon
        while j < n and ts[j] <= end:
            j += 1
        if j >= n and ts[-1] < end - 1e-9:
            continue
        lo, hi = i + 1, j
        if hi <= lo:
            continue
        fut_steer = steer[lo:hi].mean()
        fut_pwm_min = pwm[lo:hi].min()
        now = pwm[i]
        if now > 0.24 and (fut_pwm_min < 0.4 * now):
            labels[i] = "brake"
        elif fut_steer > steer_thr:
            labels[i] = "left"
        elif fut_steer < -steer_thr:
            labels[i] = "right"
        else:
            labels[i] = "straight"
    return labels


def rows_to_matrix(rows):
    return np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)


class IntentModel:
    """Gradient-boosted classifier over the feature vector."""

    def __init__(self):
        self.clf = None
        self.classes = CLASSES

    def fit(self, rows, labels, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        keep = [i for i, l in enumerate(labels) if l is not None]
        X = rows_to_matrix([rows[i] for i in keep])
        y = np.array([CLASSES.index(labels[i]) for i in keep])
        self.clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=6,
                                                  class_weight="balanced", random_state=seed)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, row):
        x = np.array([[row[f] for f in FEATURES]], dtype=float)
        pr = self.clf.predict_proba(x)[0]
        out = {c: 0.0 for c in CLASSES}
        for cls_idx, prob in zip(self.clf.classes_, pr):
            out[CLASSES[int(cls_idx)]] = float(prob)
        return out

    def predict_batch(self, rows):
        return self.clf.predict(rows_to_matrix(rows))

    def save(self, path):
        import joblib
        joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path):
        import joblib
        m = cls()
        m.clf = joblib.load(path)
        return m


class IntentEstimator:
    """Runtime wrapper: keeps the input history and returns class probabilities each tick."""

    def __init__(self, model, rate_hz=20.0):
        self.model = model
        self.period = 1.0 / rate_hz
        self._hist = []
        self._next = 0.0
        self.row = None
        self.probs = {"straight": 1.0, "left": 0.0, "right": 0.0, "brake": 0.0}

    def update(self, t, steer, pwm, pipeline):
        if t >= self._next:
            self._next = t + self.period
            row = extract_features(t, steer, pwm, pipeline, self._hist)
            self.row = row
            self._hist.append((t, steer, pwm))
            del self._hist[:-20]
            if self.model is not None and self.model.clf is not None:
                self.probs = self.model.predict_proba(row)
        return self.probs
