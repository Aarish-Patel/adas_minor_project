"""Driving assists: the driver drives, these help. LiDAR only (no camera), identical in the simulator and on the car.

    assists = DrivingAssists(params, speed_model)
    steer, pwm, level = assists.update(dt, points, driver_steer, driver_pwm, v_est)

Frames and units: vehicle frame (x forward from the rear axle, y left), steering stick -1..+1 (+ = left),
path curvature kappa in 1/m (+ = left). Every assist is switchable (assists.enabled[name]).

Intent-aware by construction: each assist predicts the path the DRIVER is commanding (their stick and
throttle) and only acts when that path is in trouble. A driver who is already steering away, braking or
easing off is left alone.

  evasive   - the driver's path hits an obstacle but a nearby path is clear: steer onto the clear path
              (the smallest change from what the driver commanded), hand control back once their own path
              is clear again. If nothing is clear it does not steer; emergency braking handles it.
  centring  - between walls (a corridor) the car is held on the centre line, next to one wall at the
              distance it had when the wall appeared; yields as soon as the driver steers deliberately.
  limiter   - caps speed in tight turns (lateral acceleration) and limits how fast the steering may move.
  narrow    - measures the free width along the driver's path: "won't fit" warning + crawl speed, or
              "tight" + reduced speed.
  proximity - side (blind-spot style) and rear alerts, raised when the driver steers or reverses toward it.
  deadman   - see deadman_pwm(): driver input lost -> throttle ramps down, never an instant cut.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import travel_distance_to_contact
from .vehicle_params import delta_to_steer, steer_to_delta


@dataclass
class AssistConfig:
    # evasive steer
    evade_time: float = 2.2          # s: look this far ahead along the driver's path (plus stopping distance)
    evade_min_look: float = 1.1      # m: never less - must react before the relay's front-cone braking (~1.0 m)
    evade_margin: float = 0.07       # m of body clearance an evasive path must keep
    evade_release_s: float = 0.35    # s the driver's own path must stay clear before control is handed back
    evade_candidates: int = 17
    evade_period: float = 0.08       # s between re-planning (the sweep is the expensive part)
    # corridor / wall centring
    centre_gain_look: float = 0.6    # m: aim this far ahead on the centre line
    centre_window: tuple = (0.0, 0.9)   # m (x, from the rear axle) where the walls are measured
    centre_max_wall: float = 0.75    # m: walls further to the side than this are ignored
    centre_driver_yield: float = 0.22   # stick: a driver steering more than this is overriding
    centre_max_assist: float = 0.45  # stick authority of the assist
    # limiter
    lat_accel_max: float = 1.2       # m/s^2 in turns
    steer_rate: float = 3.0          # stick units per second at low speed (full lock in 0.33 s)
    steer_rate_ref_v: float = 0.5    # above this speed the steering rate limit tightens proportionally
    # narrow gap
    gap_look: float = 1.5            # m along the driver's path
    gap_tight: float = 0.16          # m of spare width below which a gap counts as tight
    gap_fit: float = 0.03            # m of spare width below which the car does not fit
    gap_crawl: float = 0.18          # m/s through a tight gap
    # proximity
    side_zone: float = 0.20          # m beside the body
    rear_zone: float = 0.35          # m behind the body when reversing


def _kappa(steer, p):
    return math.tan(steer_to_delta(steer, p)) / p.wheelbase


def _steer(kappa, p):
    return delta_to_steer(math.atan(kappa * p.wheelbase), p)


class DrivingAssists:
    def __init__(self, params, speed_model, cfg=None):
        self.p, self.model = params, speed_model
        self.cfg = cfg or AssistConfig()
        self.enabled = {"evasive": False, "centring": False, "limiter": False, "narrow": False, "proximity": False}
        self.kappa_max = abs(_kappa(1.0, params))
        # state
        self.evading = False
        self.evade_kappa = 0.0
        self.evade_side = 0
        self._clear_for = 0.0
        self._since_plan = 1e9
        self._wall_ref = None          # (side, distance) for single-wall following
        self._steer_prev = 0.0
        self.info = {}

    # ------------------------------------------------------------------ helpers
    def _contact(self, points, kappa, horizon):
        return travel_distance_to_contact(points, math.atan(kappa * self.p.wheelbase), 1, self.p,
                                          horizon=horizon, margin=self.cfg.evade_margin)

    def _look(self, v):
        c = self.cfg
        stop = v * v / 3.0                       # ~1.5 m/s^2 braking
        return max(c.evade_min_look, v * c.evade_time + stop)

    # ------------------------------------------------------------------ evasive steer
    def _evasive(self, dt, pts, steer, pwm, v):
        c = self.cfg
        k_drv = _kappa(steer, self.p)
        if pwm <= 0 or v < 0.08 or len(pts) == 0:
            self.evading = False
            return steer, None
        look = self._look(v)
        d_drv = self._contact(pts, k_drv, look + 0.4)
        if not self.evading:
            if d_drv >= look:
                return steer, None
            self._since_plan = 1e9                # trouble ahead: plan now
        self._since_plan += dt
        if self._since_plan >= c.evade_period:
            self._since_plan = 0.0
            best = None
            # an escape path only has to stay clear until just past the obstacle (not the whole look-ahead,
            # otherwise the side walls of a normal room rule out every swerve)
            need = min(look + 0.3, max(0.6, min(d_drv, look) + 0.45))
            for k in np.linspace(-self.kappa_max, self.kappa_max, c.evade_candidates):
                d = self._contact(pts, k, look + 0.4)
                if d < need:
                    continue                      # this path also runs into something
                side = 1 if k > k_drv else -1
                cost = abs(k - k_drv) + (0.0 if (self.evade_side == 0 or side == self.evade_side) else 0.6)
                if best is None or cost < best[0]:
                    best = (cost, float(k), side)
            if best is None:
                if self.evading:                  # nothing clear any more: stop steering, the AEB brakes
                    self.evading = False
                    self.evade_side = 0
                self.info["evasive"] = "no clear path - braking only"
                return steer, 2
            _, self.evade_kappa, self.evade_side = best
            self.evading = True
        if d_drv >= look:
            self._clear_for += dt
            if self._clear_for >= c.evade_release_s:
                self.evading = False              # the driver's own path is clear: hand back
                self.evade_side = 0
                self._clear_for = 0.0
                self.info["evasive"] = "handed back"
                return steer, None
        else:
            self._clear_for = 0.0
        self.info["evasive"] = f"steering {'left' if self.evade_side > 0 else 'right'} around an obstacle"
        return _steer(self.evade_kappa, self.p), 2

    # ------------------------------------------------------------------ corridor / wall centring
    def _walls(self, pts):
        c = self.cfg
        half = self.p.width / 2
        x0, x1 = c.centre_window
        sel = pts[(pts[:, 0] > x0) & (pts[:, 0] < x1) & (np.abs(pts[:, 1]) < c.centre_max_wall)]
        out = {}
        for side in (1, -1):
            s = sel[sel[:, 1] * side > half]
            if len(s) < 6:
                continue
            A = np.stack([np.ones(len(s)), s[:, 0]], 1)
            (a, b), *_ = np.linalg.lstsq(A, s[:, 1], rcond=None)
            if abs(b) < 0.5 and np.std(s[:, 1] - A @ np.array([a, b])) < 0.03:     # straight, wall-like
                out[side] = (float(a), float(b))
        return out

    def _centring(self, pts, steer, v):
        c = self.cfg
        if abs(steer) > c.centre_driver_yield or len(pts) == 0 or v < 0.05:
            self._wall_ref = None                 # driver is turning on purpose: yield
            return steer, None
        walls = self._walls(pts)
        if not walls:
            self._wall_ref = None
            return steer, None
        if 1 in walls and -1 in walls:
            (al, bl), (ar, br) = walls[1], walls[-1]
            offset, slope = (al + ar) / 2, (bl + br) / 2
            what = "centred in the corridor"
            self._wall_ref = None
        else:
            side = 1 if 1 in walls else -1
            a, b = walls[side]
            if self._wall_ref is None or self._wall_ref[0] != side:
                self._wall_ref = (side, abs(a))   # keep the distance the wall was first seen at
            offset, slope = a - side * self._wall_ref[1], b
            what = f"holding {self._wall_ref[1]:.2f} m from the {'left' if side > 0 else 'right'} wall"
        L = c.centre_gain_look
        yt = offset + slope * L                   # target point on the reference line, L ahead
        kappa = 2 * yt / (L * L + yt * yt)
        s = _steer(kappa, self.p)
        s = max(-c.centre_max_assist, min(c.centre_max_assist, s))
        self.info["centring"] = what
        return s, 1

    # ------------------------------------------------------------------ narrow gap
    def _narrow(self, pts, steer, pwm, v):
        c = self.cfg
        if pwm <= 0 or len(pts) == 0:
            return None, None
        k = _kappa(steer, self.p)
        front = self.p.front_x
        sel = pts[(pts[:, 0] > front) & (pts[:, 0] < front + c.gap_look)]
        if len(sel) == 0:
            return None, None
        x = sel[:, 0]
        yc = 0.5 * k * x * x                      # driver's path centre line (small-angle arc)
        lat = sel[:, 1] - yc
        bins = np.floor((x - front) / 0.1).astype(int)
        spare = None
        for b in np.unique(bins):
            m = bins == b
            left, right = lat[m & (lat > 0)], lat[m & (lat < 0)]
            if len(left) == 0 or len(right) == 0:
                continue
            width = left.min() - right.max()
            if width < 1.2:
                sp = width - self.p.width
                spare = sp if spare is None else min(spare, sp)
        if spare is None:
            return None, None
        if spare < c.gap_fit:
            self.info["narrow"] = f"WON'T FIT - gap {spare + self.p.width:.2f} m for a {self.p.width:.2f} m car"
            return self.model.pwm_for_speed(c.gap_crawl * 0.6), 2
        if spare < c.gap_tight:
            self.info["narrow"] = f"tight gap ({spare * 100:.0f} cm spare) - slowing"
            return self.model.pwm_for_speed(c.gap_crawl), 1
        return None, None

    # ------------------------------------------------------------------ proximity
    def _proximity(self, pts, steer, pwm):
        c = self.cfg
        if len(pts) == 0:
            return None
        half, rx, fx = self.p.width / 2, self.p.rear_x, self.p.front_x
        level = None
        beside = pts[(pts[:, 0] > rx) & (pts[:, 0] < fx)]
        for side, name in ((1, "left"), (-1, "right")):
            s = beside[(beside[:, 1] * side > half) & (beside[:, 1] * side < half + c.side_zone)]
            if len(s) >= 2:
                gap = float((s[:, 1] * side).min() - half)
                toward = steer * side > 0.2
                self.info[f"side_{name}"] = f"object {gap * 100:.0f} cm on the {name}" + (" - you are steering toward it" if toward else "")
                level = max(level or 0, 2 if toward else 1)
        if pwm < 0:
            r = pts[(pts[:, 0] < rx) & (pts[:, 0] > rx - c.rear_zone) & (np.abs(pts[:, 1]) < half + 0.05)]
            if len(r) >= 2:
                self.info["rear"] = f"object {(rx - r[:, 0].max()) * 100:.0f} cm behind"
                level = max(level or 0, 2)
        return level

    # ------------------------------------------------------------------ all together
    def update(self, dt, points, steer, pwm, v):
        """Returns (steer, pwm, level). level: 0 none, 1 assisting/caution, 2 warning."""
        self.info = {}
        pts = points if points is not None else np.empty((0, 2))
        level = 0
        en = self.enabled
        if en.get("evasive"):
            s2, lv = self._evasive(dt, pts, steer, pwm, v)
            if lv:
                steer, level = s2, max(level, lv)
        if en.get("centring") and not self.evading:
            s2, lv = self._centring(pts, steer, v)
            if lv:
                steer, level = s2, max(level, lv)
        if en.get("limiter"):
            c = self.cfg
            rate = c.steer_rate * min(1.0, c.steer_rate_ref_v / max(abs(v), 1e-3))
            if self.evading:
                rate = max(rate, c.steer_rate)    # an evasive move must not be slowed down
            ds = max(-rate * dt, min(rate * dt, steer - self._steer_prev))
            steer = self._steer_prev + ds
            k = abs(_kappa(steer, self.p))
            if k > 1e-3 and pwm > 0:
                v_cap = math.sqrt(c.lat_accel_max / k)
                cap = self.model.pwm_for_speed(v_cap)
                if pwm > cap:
                    pwm = cap
                    self.info["limiter"] = f"speed capped to {v_cap:.2f} m/s in the turn"
                    level = max(level, 1)
        if en.get("narrow"):
            cap, lv = self._narrow(pts, steer, pwm, v)
            if cap is not None and pwm > cap:
                pwm = cap
            if lv:
                level = max(level, lv)
        if en.get("proximity"):
            lv = self._proximity(pts, steer, pwm)
            if lv:
                level = max(level, lv)
        self._steer_prev = steer
        return steer, pwm, level


def deadman_pwm(last_pwm, input_age, dt, timeout=0.5, ramp_pwm_per_s=300.0):
    """Throttle to send when the driver's input may be stale: unchanged while fresh, then ramped to 0 at
    ramp_pwm_per_s (a smooth stop, not a jolt). Used by the relay on the car and by the simulator."""
    if input_age <= timeout or last_pwm == 0:
        return last_pwm
    step = ramp_pwm_per_s * dt
    return math.copysign(max(0.0, abs(last_pwm) - step), last_pwm)
