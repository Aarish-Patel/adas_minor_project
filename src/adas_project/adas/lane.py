"""Lane departure warning and lane-keeping assist from floor lines (tape) seen by the front camera.

detect_lane_points()  real image -> white line pixels mapped onto the floor (vehicle frame, metres)
fit_lane()            floor points -> lane centre offset and heading error
LaneKeeper            warns when the wheels near a line; optionally steers back toward the centre

Vehicle frame: x forward, y left. offset > 0 means the car is LEFT of the lane centre.
"""

import math
from dataclasses import dataclass

import numpy as np

from .vehicle_params import delta_to_steer, steer_to_delta


def pixel_to_ground(cam, u, v):
    """Pixels (arrays) -> floor points (x, y) in the vehicle frame; NaN above the horizon."""
    R = cam.rotation()
    d_cam = np.stack([(np.asarray(u) - cam.width / 2.0) / cam.fx, (np.asarray(v) - cam.height / 2.0) / cam.fx,
                      np.ones_like(np.asarray(u, dtype=float))])
    d_veh = R @ d_cam
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -cam.z / d_veh[2]
    ok = (d_veh[2] < -1e-6) & (t > 0)
    x = np.where(ok, cam.x + t * d_veh[0], np.nan)
    y = np.where(ok, cam.y + t * d_veh[1], np.nan)
    return x, y


def detect_lane_points(frame_bgr, cam, x_range=(0.15, 1.6), y_max=0.7, white_v=190, white_s=70):
    """Bright, low-saturation pixels (white tape) on the floor -> ground points (N, 2)."""
    import cv2
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] > white_v) & (hsv[:, :, 1] < white_s)
    mask[: cam.height // 2, :] = False                       # floor is in the lower half
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    vs, us = np.nonzero(mask)
    if len(us) == 0:
        return np.empty((0, 2))
    if len(us) > 4000:
        idx = np.random.default_rng(0).choice(len(us), 4000, replace=False)
        us, vs = us[idx], vs[idx]
    x, y = pixel_to_ground(cam, us.astype(float), vs.astype(float))
    keep = np.isfinite(x) & (x > x_range[0]) & (x < x_range[1]) & (np.abs(y) < y_max)
    return np.column_stack([x[keep], y[keep]])


@dataclass
class LaneEstimate:
    valid: bool = False
    offset: float = 0.0          # m, + = car left of the lane centre
    heading: float = 0.0         # rad, lane direction relative to the car (+ = lane bends left)
    width: float = 0.0
    lines: int = 0               # 1 or 2 lines seen
    t: float = 0.0


def _fit_line(pts):
    """y = a*x + b by robust least squares."""
    x, y = pts[:, 0], pts[:, 1]
    for _ in range(3):
        A = np.column_stack([x, np.ones_like(x)])
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        res = y - (a * x + b)
        sigma = max(res.std(), 0.004)
        keep = np.abs(res) < 2.0 * sigma
        if keep.sum() < 6 or keep.all():
            break
        x, y = x[keep], y[keep]
    return float(a), float(b)


def fit_lane(points, lane_width=0.45, min_points=12, t=0.0):
    if len(points) < min_points:
        return LaneEstimate(t=t)
    ys = np.sort(points[:, 1])
    gaps = np.diff(ys)
    split = None
    if len(gaps) and gaps.max() > 0.15 and lane_width * 0.6 < ys[-1] - ys[0] < lane_width * 1.6:
        split = 0.5 * (ys[int(np.argmax(gaps))] + ys[int(np.argmax(gaps)) + 1])
    if split is not None:
        left, right = points[points[:, 1] > split], points[points[:, 1] <= split]
        if len(left) >= 6 and len(right) >= 6:
            aL, bL = _fit_line(left)
            aR, bR = _fit_line(right)
            a, b = 0.5 * (aL + aR), 0.5 * (bL + bR)
            return LaneEstimate(True, -b, math.atan(a), bL - bR, 2, t)
    a, b = _fit_line(points)
    # one line only: decide which side it is from where it lies, then step half a lane to the centre
    b_centre = b - lane_width / 2.0 if b > 0 else b + lane_width / 2.0
    return LaneEstimate(True, -b_centre, math.atan(a), 0.0, 1, t)


class LaneKeeper:
    """mode: "off" | "warn" | "assist"."""

    def __init__(self, params, lane_width=0.45, caution=0.075, warning=0.115, hands_on=0.22, k=1.8, max_age=0.6):
        self.p = params
        self.lane_width = lane_width
        self.caution, self.warning = caution, warning
        self.hands_on = hands_on
        self.k = k
        self.max_age = max_age
        self.mode = "off"
        self.level = 0
        self.est = LaneEstimate()
        self.assisting = False

    def update(self, t, est, driver_steer, v):
        """Returns (steer_to_use, warning_level 0-3)."""
        if est is not None:
            self.est = est
        self.level = 0
        self.assisting = False
        e = self.est
        if self.mode == "off" or not e.valid or t - e.t > self.max_age:
            return driver_steer, 0

        off = abs(e.offset)
        if off > self.warning:
            self.level = 2
        elif off > self.caution:
            self.level = 1

        if self.mode == "assist" and abs(driver_steer) < self.hands_on:
            v_s = max(abs(v), 0.25)
            delta = e.heading - math.atan(self.k * e.offset / v_s * 0.6)
            dmax = steer_to_delta(1.0, self.p)
            delta = max(steer_to_delta(-1.0, self.p), min(dmax, delta))
            self.assisting = True
            return delta_to_steer(delta, self.p), self.level
        return driver_steer, self.level
