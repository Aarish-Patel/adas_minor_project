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
    evade_release_s: float = 0.35    # (unused since the return-to-line phase; kept for old configs)
    evade_candidates: int = 17
    evade_period: float = 0.08       # s between re-planning (the sweep is the expensive part)
    evade_ttc: float = 1.6           # s: only step in when contact is this close in time...
    evade_min_v: float = 0.18        # m/s: ...and the car is moving at least this fast (creeping = the driver's call)
    evade_confirm_s: float = 0.15    # s the threat must persist (one noisy scan never swerves the car)
    evade_driver_override: float = 0.45   # stick: a driver steering harder than this takes over at once
    evade_max_s: float = 15.0        # s: a manoeuvre never lasts longer...
    evade_max_past: float = 2.5      # m: ...or goes further than this past the obstacle
    return_look: float = 1.2         # m: aim this far ahead on the original line when rejoining it (gentle)
    pass_margin: float = 0.03        # m body clearance used once alongside the obstacle
    gap_clear: float = 0.06          # m: a side is used only if the car fits past with this much each side
    offset_clear: float = 0.12       # m kept between the obstacle edge and the body while passing
    pass_behind: float = 0.08        # m: return once the rear bumper is this far past the obstacle
    law_len: float = 0.45            # heading law (as in the obstacle-avoidance demo that ran on the car)
    law_tau: float = 0.25
    law_max_deg: float = 28.0
    return_done_y: float = 0.05      # m and ...
    return_done_deg: float = 3.0     # ... deg from the original line = rejoined, control handed back
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
        self.y_target = 0.0
        self.box = None
        self.phase = None               # None, "EVADE" (going around) or "RETURN" (rejoining the original line)
        self.phase_t = 0.0
        self.ex = self.ey = self.eth = 0.0   # dead-reckoned pose since the swerve began
        self._trigger_for = 0.0
        self._pwm_hist = []
        self._wall_ref = None          # (side, distance) for single-wall following
        self._steer_prev = 0.0
        self.info = {}

    # ------------------------------------------------------------------ helpers
    def _contact(self, points, kappa, horizon, margin=None):
        return travel_distance_to_contact(points, math.atan(kappa * self.p.wheelbase), 1, self.p, horizon=horizon,
                                          margin=self.cfg.evade_margin if margin is None else margin)

    def _look(self, v):
        c = self.cfg
        stop = v * v / 3.0                       # ~1.5 m/s^2 braking
        return max(c.evade_min_look, v * c.evade_time + stop)

    # ------------------------------------------------------------------ evasive steer
    def _stop_evading(self, why=None):
        self.evading = False
        self.phase = None
        self.evade_side = 0
        self._trigger_for = 0.0
        if why:
            self.info["evasive"] = why

    def _obstacle(self, pts, k_drv, look):
        """The obstacle on the driver's path (region-grown cluster) and the free room beside it on each side.
        Returns (box [xmin, xmax, ymin, ymax], free_left, free_right) in the vehicle frame, or None."""
        c = self.cfg
        half = self.p.width / 2
        fx = self.p.front_x
        lat = pts[:, 1] - 0.5 * k_drv * pts[:, 0] ** 2
        seed = (pts[:, 0] > fx - 0.05) & (pts[:, 0] < fx + look + 0.3) & (np.abs(lat) < half + c.evade_margin)
        if seed.sum() < 2:
            return None
        near = (pts[:, 0] > fx - 0.3) & (pts[:, 0] < fx + look + 1.2) & (np.abs(pts[:, 1]) < 1.8)
        cand = pts[near]
        inside = np.zeros(len(cand), bool)
        frontier = pts[seed]
        for _ in range(40):                                   # region grow, 12 cm steps
            d = np.sqrt(((cand[:, None, :] - frontier[None, :, :]) ** 2).sum(-1)).min(1)
            grow = (d < 0.12) & ~inside
            if not grow.any():
                break
            inside |= grow
            frontier = cand[grow]
        cl = cand[inside]
        if len(cl) < 2:
            return None
        xmin, xmax, ymin, ymax = cl[:, 0].min(), cl[:, 0].max(), cl[:, 1].min(), cl[:, 1].max()
        others = cand[~inside]
        band = others[(others[:, 0] > xmin - 0.3) & (others[:, 0] < xmax + 0.4)]
        left = band[band[:, 1] > ymax][:, 1]
        right = band[band[:, 1] < ymin][:, 1]
        free_l = float(left.min() - ymax) if len(left) else 1.5
        free_r = float(ymin - right.max()) if len(right) else 1.5
        return [float(xmin), float(xmax), float(ymin), float(ymax)], free_l, free_r

    def _to_start_frame(self, pts):
        c, s = math.cos(self.eth), math.sin(self.eth)
        return np.column_stack([self.ex + c * pts[:, 0] - s * pts[:, 1], self.ey + s * pts[:, 0] + c * pts[:, 1]])

    def _heading_law(self, y_ref):
        """Damped heading control onto the line y = y_ref (the law that rejoined the line on the car in the
        obstacle-avoidance demo): aim back at the line, close the heading error over law_tau metres."""
        c = self.cfg
        th_des = -math.atan((self.ey - y_ref) / c.law_len)
        th_des = max(-math.radians(c.law_max_deg), min(math.radians(c.law_max_deg), th_des))
        k = (th_des - self.eth) / c.law_tau
        return max(-self.kappa_max, min(self.kappa_max, k))

    def _evasive(self, dt, pts, steer, pwm, v):
        c = self.cfg
        k_drv = _kappa(steer, self.p)
        self._pwm_hist = (self._pwm_hist + [pwm])[-12:]
        if self.phase is not None:                # dead-reckon the manoeuvre (frame: where the swerve began)
            k_now = _kappa(self._steer_prev, self.p)
            self.eth += k_now * v * dt
            self.ex += v * math.cos(self.eth) * dt
            self.ey += v * math.sin(self.eth) * dt
            if pwm <= 0:
                self._stop_evading("driver braked - handed back")
                return steer, None
            if abs(steer) > c.evade_driver_override:
                self._stop_evading("driver steered - handed back")
                return steer, None
        if pwm <= 0 or len(pts) == 0:
            self._trigger_for = 0.0
            return steer, None
        look = self._look(v)
        half = self.p.width / 2

        if self.phase is None:
            easing = len(self._pwm_hist) >= 6 and pwm < 0.85 * max(self._pwm_hist[:-2])
            d_drv = self._contact(pts, k_drv, look + 0.4)
            ttc = d_drv / max(v, 1e-3)
            if v < c.evade_min_v or easing or d_drv >= look or ttc > c.evade_ttc:
                self._trigger_for = 0.0
                return steer, None
            self._trigger_for += dt
            if self._trigger_for < c.evade_confirm_s:
                return steer, None
            ob = self._obstacle(pts, k_drv, look)
            if ob is None:
                return steer, None
            box, free_l, free_r = ob
            need = self.p.width + 2 * c.gap_clear
            if max(free_l, free_r) < need:
                self.info["evasive"] = (f"no room either side ({free_l:.2f} / {free_r:.2f} m, car needs {need:.2f} m)"
                                        " - braking only")
                self._trigger_for = 0.0
                return steer, 2
            self.evade_side = 1 if free_l >= free_r else -1
            free = free_l if self.evade_side > 0 else free_r
            clear = min(c.offset_clear, (free - self.p.width) / 2)
            edge = box[3] if self.evade_side > 0 else box[2]
            self.y_target = edge + self.evade_side * (half + clear)
            self.box, self.box0 = box, list(box)
            self.phase, self.evading = "EVADE", True
            self.ex = self.ey = self.eth = 0.0
            self.phase_t = 0.0
            self._since_plan = 1e9
        self.phase_t += dt
        self._since_plan += dt

        # keep the obstacle's extent up to date while beside it (its side becomes visible as the car passes)
        if self.phase in ("EVADE", "PASS") and self._since_plan >= c.evade_period:
            self._since_plan = 0.0
            w = self._to_start_frame(pts)
            xmin, xmax, ymin, ymax = self.box
            m = (w[:, 0] > xmin - 0.08) & (w[:, 0] < xmax + 0.25) & (w[:, 1] > ymin - 0.08) & (w[:, 1] < ymax + 0.08)
            if m.sum() >= 2:
                xmax = max(xmax, float(w[m, 0].max()))        # depth: grows as the side comes into view
                y0, y1 = self.box0[2], self.box0[3]           # width: at most 3 cm beyond the first measurement
                self.box = [xmin, xmax, max(y0 - 0.03, min(ymin, float(w[m, 1].min()))),
                            min(y1 + 0.03, max(ymax, float(w[m, 1].max())))]
                edge = self.box[3] if self.evade_side > 0 else self.box[2]
                target = edge + self.evade_side * (half + c.offset_clear)
                if (target - self.y_target) * self.evade_side > 0:
                    self.y_target = target           # the obstacle turned out wider: move out further, never in

        if self.phase == "EVADE" and abs(self.ey - self.y_target) < 0.05 and abs(self.eth) < math.radians(12):
            self.phase = "PASS"
        rear_bumper = self.ex + self.p.rear_x
        if self.phase in ("EVADE", "PASS") and rear_bumper > self.box[1] + c.pass_behind:
            k_ret = self._heading_law(0.0)
            if self._contact(pts, k_ret, look + 0.4, c.pass_margin) >= min(look, 0.6):
                self.phase = "RETURN"
        if self.phase == "RETURN" and abs(self.ey) < c.return_done_y and abs(self.eth) < math.radians(c.return_done_deg):
            self._stop_evading("back on the original line - handed back")
            return steer, None
        if self.phase_t > c.evade_max_s or self.ex > self.box[1] + c.evade_max_past:
            self._stop_evading("manoeuvre ended (limit) - handed back")
            return steer, None
        self.evade_kappa = self._heading_law(0.0 if self.phase == "RETURN" else self.y_target)
        side = "left" if self.evade_side > 0 else "right"
        self.info["evasive"] = {"EVADE": f"steering {side} around an obstacle",
                                "PASS": f"passing the obstacle ({self.ey * 100:+.0f} cm off the line)",
                                "RETURN": f"obstacle passed - rejoining the line ({self.ey * 100:+.0f} cm, "
                                          f"{math.degrees(self.eth):+.0f} deg)"}[self.phase]
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
