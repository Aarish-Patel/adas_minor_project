"""Adaptive warning: the learned hazard predictor.

Takes the physics risk, the intent probabilities and how the driver is behaving,
and returns the probability that a hazard (collision or near miss) is about to
happen. Trained on simulated drives in sim/warning_eval.py.
"""

import math
import os

import numpy as np

from .intent import FEATURES
from .warning import RiskScorer, ttc_for_steer

EXTRA = ["risk_base", "ttc_base", "risk_blend", "ttc_cur", "ttc_str", "ttc_left", "ttc_right",
         "ttc_brake", "p_straight", "p_left", "p_right", "p_brake"]
COLUMNS = FEATURES + EXTRA


def _cap(x, hi=5.0):
    return min(x, hi) if math.isfinite(x) else hi


def warning_row(intent_row, pipeline, steer, pwm, probs, scorer):
    """Feature vector (in COLUMNS order) plus the baseline and blended risks."""
    v = pipeline.estimator.v
    d = 1 if v >= 0 else -1
    speed = abs(v)
    pts, tracks, p = pipeline.points, pipeline.tracks, pipeline.p

    rb, ttc_b = scorer.baseline(pipeline, steer, pwm)
    rblend, _ = scorer.intent_aware(pipeline, steer, pwm, probs)
    ttcs = [_cap(ttc_for_steer(pts, tracks, st, speed, d, p)) for st in
            (steer, 0.0, scorer.class_steer["left"], scorer.class_steer["right"])]
    ttc_brake = _cap(ttc_for_steer(pts, tracks, steer, speed * scorer.brake_speed_factor, d, p))

    row = [intent_row[f] for f in FEATURES]
    row += [rb, _cap(ttc_b), rblend, *ttcs, ttc_brake,
            probs["straight"], probs["left"], probs["right"], probs["brake"]]
    return row, rb, rblend


class AdaptiveWarning:
    def __init__(self, path):
        import joblib
        blob = joblib.load(path)
        self.clf = blob["clf"]
        self.columns = blob["columns"]
        self.combine = blob.get("combine", "adaptive")

    @staticmethod
    def available(path):
        return os.path.exists(path)

    def risk(self, row, physics_risk=None):
        """Warning risk 0..1. Depending on what worked best on validation drives it is the learned
        hazard probability alone, or that probability combined with the physics risk."""
        p = float(self.clf.predict_proba(np.array([row], dtype=float))[0, 1])
        if physics_risk is None or self.combine == "adaptive":
            return p
        rb = min(max(physics_risk, 0.0), 1.0)
        if self.combine == "gated":
            return float(np.sqrt(rb * p))
        if self.combine == "gated_hard":
            return rb * p
        return p
