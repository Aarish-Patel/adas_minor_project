"""Collision-risk scoring, with and without driver intent.

Physics-only (baseline): assume the driver keeps the current steering and speed;
risk rises as the predicted time to collision falls.

Intent-aware: the intent model gives probabilities for what the driver will do
next (keep straight, steer left, steer right, ease off). The path for each of
those is checked separately and the risks are blended by probability:

    risk = p(straight) * risk(steer straight)  + p(left) * risk(steer left)
         + p(right) * risk(steer right)        + p(brake) * risk(current steer, slower)

So it warns EARLY when the likely path leads into something, and stays QUIET
when the driver is clearly already steering or braking away from it.
"""

import math

import numpy as np

from .geometry import travel_distance_to_contact
from .tracking import moving_object_contact
from .vehicle_params import steer_to_delta

# Steering the driver typically holds while turning (measured on the training data)
DEFAULT_CLASS_STEER = {"left": 0.5, "right": -0.5, "straight": 0.0}


def ttc_for_steer(points, tracks, steer, speed, direction, p, horizon=2.0, margin=0.03):
    delta = steer_to_delta(steer, p)
    D = travel_distance_to_contact(points, delta, direction, p, horizon=horizon, margin=margin)
    if tracks:
        Dm = moving_object_contact(tracks, delta, direction, max(speed, 0.05), p, margin=margin)[0]
        D = min(D, Dm)
    if not math.isfinite(D) or speed < 0.05:
        return math.inf
    return D / speed


def risk_from_ttc(ttc, t_warn=1.6, t_crit=0.5):
    """0 when far away in time, 1 when collision is imminent."""
    if not math.isfinite(ttc):
        return 0.0
    return float(min(1.0, max(0.0, (t_warn - ttc) / (t_warn - t_crit))))


class RiskScorer:
    def __init__(self, params, class_steer=None, t_warn=1.6, t_crit=0.5, brake_speed_factor=0.45):
        self.p = params
        self.class_steer = class_steer or DEFAULT_CLASS_STEER
        self.t_warn = t_warn
        self.t_crit = t_crit
        self.brake_speed_factor = brake_speed_factor

    def _direction_speed(self, v_est, driver_pwm):
        if abs(v_est) > 0.05:
            return (1 if v_est > 0 else -1), abs(v_est)
        return (1 if driver_pwm >= 0 else -1), 0.0

    def baseline(self, pipeline, steer, driver_pwm):
        d, v = self._direction_speed(pipeline.estimator.v, driver_pwm)
        pts = pipeline.points
        ttc = ttc_for_steer(pts, pipeline.tracks, steer, v, d, self.p)
        return risk_from_ttc(ttc, self.t_warn, self.t_crit), ttc

    def intent_aware(self, pipeline, steer, driver_pwm, probs):
        d, v = self._direction_speed(pipeline.estimator.v, driver_pwm)
        pts, tracks = pipeline.points, pipeline.tracks

        risk = 0.0
        worst = math.inf
        for cls, prob in probs.items():
            if prob < 0.02:
                continue
            if cls == "brake":
                ttc = ttc_for_steer(pts, tracks, steer, v * self.brake_speed_factor, d, self.p)
            else:
                ttc = ttc_for_steer(pts, tracks, self.class_steer[cls], v, d, self.p)
            risk += prob * risk_from_ttc(ttc, self.t_warn, self.t_crit)
            if prob > 0.25:
                worst = min(worst, ttc)
        return float(risk), worst


def level_from_risk(risk, thr_caution=0.25, thr_warning=0.5):
    if risk >= thr_warning:
        return 2
    if risk >= thr_caution:
        return 1
    return 0
