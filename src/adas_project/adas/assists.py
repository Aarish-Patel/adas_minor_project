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
    # swerve trigger times from the relay Monte Carlo (96 paired drives, sim/relay_mc.py): 1.2 s instead of 1.6 s
    # cut needless steering takeovers 34 -> 12 with 0 crashes and the same goals reached; the brake gate still
    # stops the car if a swerve comes too late
    evade_ttc_attentive: float = 0.7  # s: an attentive (actively steering) driver is given more time to act
    trigger_margin: float = 0.03     # m: the driver's path must really touch (+3 cm) to trigger a swerve
    evade_ttc: float = 1.2           # s: only step in when contact is this close in time...
    evade_min_v: float = 0.18        # m/s: ...and the car is moving at least this fast (creeping = the driver's call)
    evade_confirm_s: float = 0.15    # s the threat must persist (one noisy scan never swerves the car)
    # last point to steer (Brannstrom, Coelingh & Sjoberg, IEEE T-ITS 2010): the intent model may hold a swerve back
    # for a driver who is NOT moving the stick only while the swerve could still start before the brake gate has to
    # act - FOS x stopping distance (the same constants as pi/path_gate.py) plus the trigger's confirm time
    lps_fos: float = 1.3
    lps_base: float = 0.05
    lps_reaction: float = 0.20
    lps_decel: float = 4.0
    evade_driver_override: float = 0.45   # stick: a driver steering harder than this takes over at once
    evade_max_s: float = 15.0        # s: a manoeuvre never lasts longer...
    evade_max_past: float = 2.5      # m: ...or goes further than this past the obstacle
    return_look: float = 1.2         # m: aim this far ahead on the original line when rejoining it (gentle)
    pass_margin: float = 0.03        # m body clearance used once alongside the obstacle
    goal_past: float = 0.30          # m past the obstacle + a body length before rejoining the line
    plan_kappa_max: float = 1.5      # 1/m: planner steering limit (measured tightest reliable turn ~0.65 m)
    plan_max_nodes: int = 2500
    track_look: float = 0.30         # m pure-pursuit look-ahead on the planned path
    evade_speed: float = 0.40        # m/s: fastest the manoeuvre is driven (the driver's throttle is capped)
    reverse_speed: float = 0.15      # m/s when backing off
    offset_min: float = 0.16         # m: smallest sideways offset tried (half the car + a little)
    offset_max: float = 0.90         # m: largest
    offset_step: float = 0.04
    return_clear_ahead: float = 0.35 # m of the original line that must be free ahead of the nose to come back
    clear_wish: float = 0.15         # m of clearance wanted; less is allowed (down to pass_margin) but costs
    clear_weight: float = 3.0
    replan_period: float = 0.3       # s between re-plans during the manoeuvre
    replan_window: float = 0.25      # m: re-plans prefer offsets near the current one...
    switch_cost: float = 0.8         # ...and pay this per metre of change (no dithering between sides)
    evade_max_len: float = 6.0       # m: a manoeuvre never goes further than this
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
        self._plan_poses = None
        self._plan_clear = None
        self.pose_fix = None            # (x, y, th) from scan matching, start frame, set from outside
        self.intent_k_rate = None       # driver's curvature rate (1/m/s) when attentive, else None; set from outside
        self.intent_hold = False        # learned intent: an attentive driver who will handle it - no swerve (outside)
        self.intent_attentive = True    # the stick moved recently (outside); False limits the hold (last point to steer)
        self._planner = None
        self._track_i = 0
        self._x_goal_v = 1.0
        self.evade_pwm = None           # throttle the manoeuvre needs (cap, or reverse creep)
        self.replans = 0
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

    # ---- Hybrid A* manoeuvres (adas/hybrid_astar.py; Dolgov et al., IJRR 2010) -------------------------------
    # Frame: the "line frame" - where the manoeuvre began, x along the driver's desired path, y left.

    def _to_start_frame(self, pts):
        c, s = math.cos(self.eth), math.sin(self.eth)
        return np.column_stack([self.ex + c * pts[:, 0] - s * pts[:, 1], self.ey + s * pts[:, 0] + c * pts[:, 1]])

    def _to_vehicle_frame(self, pts_start):
        c, s = math.cos(self.eth), math.sin(self.eth)
        dx, dy = pts_start[:, 0] - self.ex, pts_start[:, 1] - self.ey
        return np.column_stack([c * dx + s * dy, -s * dx + c * dy])

    def _x_goal(self, pts_s):
        """Where the manoeuvre may rejoin the line: past the first blockage on it by a body length + margin."""
        half = self.p.width / 2 + 0.06
        on = pts_s[(np.abs(pts_s[:, 1]) < half) & (pts_s[:, 0] > self.ex + self.p.front_x - 0.05)]
        xb = float(on[:, 0].min()) if len(on) else self.ex + 1.0
        return xb + (self.p.front_x - self.p.rear_x) + self.cfg.goal_past

    def _hastar_plan(self, pts_s, allow_reverse):
        from .hybrid_astar import HybridAStar
        if self._planner is None:
            self._planner = HybridAStar(self.p, kappa_max=min(self.kappa_max, self.cfg.plan_kappa_max))
        near = pts_s[(pts_s[:, 0] > self.ex - 1.0) & (pts_s[:, 0] < self.ex + 5.0) & (np.abs(pts_s[:, 1]) < 2.0)]
        return self._planner.plan(near, (self.ex, self.ey, self.eth), x_goal=self._x_goal_v,
                                  allow_reverse=allow_reverse, max_nodes=self.cfg.plan_max_nodes)

    def _path_still_clear(self, pts_s):
        if self._plan_poses is None or self._planner is None:
            return False
        from .hybrid_astar import Grid
        rest = self._plan_poses[self._track_i:]
        if not len(rest):
            return False
        g = Grid(pts_s[(np.abs(pts_s[:, 1]) < 2.0)], min(self.ex, 0.0) - 1.0, self._x_goal_v + 3.0, -2.0, 2.0)
        return self._planner.clearance(g, rest[::2, :3]) >= 0.01     # the margin was applied when planning

    def _track(self):
        """Pure pursuit on the planned path; returns (curvature, direction of travel)."""
        P = self._plan_poses
        j0 = self._track_i
        d = np.hypot(P[j0:j0 + 60, 0] - self.ex, P[j0:j0 + 60, 1] - self.ey)
        j = j0 + int(np.argmin(d)) if len(d) else j0
        self._track_i = j
        direction = int(P[j, 3])
        k = j
        while k + 1 < len(P) and P[k + 1, 3] == direction and \
                math.hypot(P[k, 0] - P[j, 0], P[k, 1] - P[j, 1]) < self.cfg.track_look:
            k += 1
        xl, yl = self._to_vehicle_frame(P[k:k + 1, :2])[0]
        dd = xl * xl + yl * yl
        kappa = 2.0 * yl / dd if dd > 1e-4 else 0.0
        return max(-self.kappa_max, min(self.kappa_max, kappa)), direction

    def preview(self):
        """For the GUI, in the current vehicle frame: the planned manoeuvre and the original line, or None."""
        if self.phase is None or self._plan_poses is None or not len(self._plan_poses):
            return None
        rest = self._plan_poses[self._track_i:]
        path = self._to_vehicle_frame(rest[:, :2]) if len(rest) else np.empty((0, 2))
        xs = np.linspace(-0.8, self._x_goal_v + 1.5, 30)
        line = self._to_vehicle_frame(np.column_stack([xs, np.zeros_like(xs)]))
        return path, line

    def _last_point_to_steer(self, v):
        """Distance along the driver's path below which a held-back swerve must start: where the brake gate would
        begin to limit (FOS x stopping distance), plus the travel during the trigger's confirm time."""
        c = self.cfg
        v = abs(v)
        stop = c.lps_base + v * c.lps_reaction + v * v / (2 * c.lps_decel)
        return c.lps_fos * stop + v * (c.evade_confirm_s + 0.05)

    def _evasive(self, dt, pts, steer, pwm, v):
        c = self.cfg
        k_drv = _kappa(steer, self.p)
        self._pwm_hist = (self._pwm_hist + [pwm])[-12:]
        self.evade_pwm = None
        if self.phase is not None:                # pose in the frame where the manoeuvre began:
            if self.pose_fix is not None:          # scan matching (the relay feeds it) beats dead reckoning...
                self.ex, self.ey, self.eth = self.pose_fix
                self.pose_fix = None
            else:                                  # ...which fills in between scans
                k_now = _kappa(self._steer_prev, self.p)
                self.eth += k_now * v * dt
                self.ex += v * math.cos(self.eth) * dt
                self.ey += v * math.sin(self.eth) * dt
            if pwm <= 0:
                self._stop_evading("driver let go / braked - handed back")
                return steer, None
            if abs(steer) > c.evade_driver_override:
                self._stop_evading("driver steered - handed back")
                return steer, None
        if pwm <= 0 or len(pts) == 0:
            self._trigger_for = 0.0
            return steer, None
        look = self._look(v)

        if self.phase is None:
            easing = len(self._pwm_hist) >= 6 and pwm < 0.85 * max(self._pwm_hist[:-2])
            if self.intent_k_rate is not None:       # intent-aware: an attentive driver's path keeps curving
                from .geometry import contact_along_changing_curvature
                d_drv = contact_along_changing_curvature(pts, k_drv, self.intent_k_rate, v, 1, self.p,
                                                         horizon=look + 0.4, margin=c.trigger_margin)
            else:
                d_drv = self._contact(pts, k_drv, look + 0.4, c.trigger_margin)   # a real contact course
            ttc = d_drv / max(v, 1e-3)
            stuck = v < 0.05 and d_drv < 0.35             # held at an obstacle with the throttle on
            attentive = self.intent_k_rate is not None and self.intent_attentive
            ttc_limit = c.evade_ttc_attentive if attentive else c.evade_ttc
            if not stuck and (v < c.evade_min_v or easing or d_drv >= look or ttc > ttc_limit):
                self._trigger_for = 0.0
                return steer, None
            self._trigger_for += dt
            if self._trigger_for < c.evade_confirm_s:
                return steer, None
            if self.intent_hold and (self.intent_attentive or d_drv > self._last_point_to_steer(v)):
                # the driver is on it (learned intent) - the brake still watches
                self.info["evasive"] = "driver is avoiding it - not intervening"
                return steer, None
            self.ex = self.ey = self.eth = 0.0      # the line frame: the car now, the driver's path straight ahead
            self._x_goal_v = self._x_goal(pts)
            path = self._hastar_plan(pts, allow_reverse=False)
            if path is None:
                path = self._hastar_plan(pts, allow_reverse=True)      # back off first, then go round
            if path is None:
                self.info["evasive"] = "no safe way around - braking only"
                self._trigger_for = 0.0
                return steer, 2
            self._plan_poses, self._track_i = path, 0
            self.phase, self.evading = "EXECUTE", True
            self.phase_t = 0.0
            self._since_plan = 0.0
        self.phase_t += dt
        self._since_plan += dt

        pts_s = self._to_start_frame(pts)
        if self._since_plan >= c.replan_period:
            self._since_plan = 0.0              # new scans: keep the plan if it is still clear, otherwise re-plan
            if self.phase == "WAIT":
                self._since_plan = -0.3             # searching again: every 0.6 s, not every 0.3 s
            if self.phase == "WAIT" or not self._path_still_clear(pts_s):
                path = self._hastar_plan(pts_s, allow_reverse=False)
                if path is None:
                    path = self._hastar_plan(pts_s, allow_reverse=True)
                if path is None:
                    # keep the original line and keep trying; the path brake holds the car meanwhile
                    self.phase = "WAIT"
                else:
                    self._plan_poses, self._track_i = path, 0
                    self.phase = "EXECUTE"
                    self.replans += 1
        if self.phase == "WAIT":
            if self.phase_t > c.evade_max_s:
                self._stop_evading("no way around found - handed back")
                return steer, 2
            self.evade_pwm = min(pwm, self.model.pwm_for_speed(c.reverse_speed))
            self.info["evasive"] = "looking for a way around (holding the original line)"
            self.evade_kappa = 0.0
            return _steer(0.0, self.p), 2

        P = self._plan_poses
        at_end = self._track_i >= len(P) - 3 or math.hypot(P[-1, 0] - self.ex, P[-1, 1] - self.ey) < 0.06
        if at_end and abs(self.ey) < c.return_done_y + 0.03 and abs(self.eth) < math.radians(c.return_done_deg + 3):
            self._stop_evading("back on the original line - handed back")
            return steer, None
        if self.phase_t > c.evade_max_s:
            self._stop_evading("manoeuvre took too long - handed back")
            return steer, None
        kappa, direction = self._track()
        self.evade_kappa = kappa
        # speed the path can be followed at: the driver's throttle, capped; reversing legs at a creep
        if direction < 0:
            self.evade_pwm = -self.model.pwm_for_speed(c.reverse_speed)
        else:
            self.evade_pwm = min(pwm, self.model.pwm_for_speed(c.evade_speed))
        side = "left" if np.mean(P[:, 1]) > 0 else "right"
        self.info["evasive"] = ("backing off to make room" if direction < 0 else
                                f"going around on the {side} ({self.ey * 100:+.0f} cm off the line, "
                                f"{math.degrees(self.eth):+.0f} deg)")
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
            if self.evading and self.evade_pwm is not None:
                pwm = self.evade_pwm
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
