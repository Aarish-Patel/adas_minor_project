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
    # the trigger also scales with speed by DISTANCE (user, 28 Sep: planning failed at high speed): step in no later
    # than the last point to steer at this speed (below) plus this margin, even if the time to contact is longer
    trigger_lps_margin: float = 0.25
    # intent-aware: hold the swerve back while the brake could still stop the car, whatever the stick is doing (an idle stick
    # is not a threat by itself); the swerve starts only when the driver is in trouble for real (learned intent says so, or the
    # brake's physical envelope is reached). Needless-takeover reduction, see RESEARCH.md section 12.
    intent_defer_to_brake: bool = False
    # intent-aware manoeuvre commitment (production evasive-steer practice: the manoeuvre is not dropped for a lifted pedal or a
    # steering twitch, and once the driver has really overridden it the system stays back for a while)
    evade_release_s: float = 0.9     # s the throttle must stay released before the manoeuvre is handed back
    evade_fight_s: float = 0.35      # s the driver must steer AGAINST the manoeuvre (past the override) to cancel it
    stuck_wait_s: float = 2.0        # s a held car waits before the assist frees it (the driver may sort it out first)
    respect_driver_s: float = 6.0    # s after an override: no new swerve unless the brake's own envelope is reached
    defer_fos: float = 1.15
    # last point to steer (Brannstrom, Coelingh & Sjoberg, IEEE T-ITS 2010): the intent model may hold a swerve back
    # for a driver who is NOT moving the stick only until the last point a swerve can still get round: the car needs
    # sqrt(2 * offset / kappa_max) of travel to move `lps_offset` sideways at its tightest turn, plus what it covers
    # during the trigger's confirmation, the plan and the command delay (lps_latency_s) - or, if later, the
    # brake gate's envelope (the same constants as pi/path_gate.py)
    lps_offset: float = 0.30         # m sideways to clear a typical obstacle (half the car + margin + half a box)
    lps_latency_s: float = 0.45      # s: confirm 0.15 + planning ~0.2 on the Pi + command delay 0.12
    lps_fos: float = 1.3
    lps_base: float = 0.05
    lps_reaction: float = 0.20
    lps_decel: float = 4.0
    plan_lookahead_s: float = 0.15   # s: plans start from where the car will be when they arrive (latency compensation)
    evade_driver_override: float = 0.45   # stick: a driver steering harder than this takes over at once
    evade_max_s: float = 15.0        # s: a manoeuvre never lasts longer...
    evade_max_past: float = 2.5      # m: ...or goes further than this past the obstacle
    return_look: float = 1.2         # m: aim this far ahead on the original line when rejoining it (gentle)
    pass_margin: float = 0.03        # m body clearance used once alongside the obstacle
    goal_past: float = 0.30          # m past the obstacle + a body length before rejoining the line
    plan_kappa_max: float = 1.5      # 1/m: planner steering limit (measured tightest reliable turn ~0.65 m)
    plan_max_nodes: int = 2500
    plan_timeout_s: float = 2.5      # s: no plan by then (both search budgets used) -> give up, the brake stays
    mppi_fallback: bool = True       # no Hybrid A* path: MPPI steers locally while the search is retried
    track_look: float = 0.30         # m pure-pursuit look-ahead on the planned path
    evade_speed: float = 0.40        # m/s: fastest the manoeuvre is driven (the driver's throttle is capped)
    reverse_speed: float = 0.15      # m/s when backing off
    # held at an obstacle with the throttle on and no way round found (user, 28 Sep: "don't just sit there"):
    # back straight off a little and search again from there, a few times, then hand back and pause
    backoff_m: float = 0.30          # m to back off before searching again
    backoff_max_s: float = 2.5       # s a back-off may take
    backoff_rear_clear: float = 0.12 # m that must stay free behind the car while backing off
    backoff_tries: int = 3           # back-offs per obstacle before handing back
    stall_replan_s: float = 0.6      # s stopped on a planned path with the throttle held before searching again
    stall_replans: int = 3           # such searches before backing off
    retry_pause_s: float = 1.5       # s without a new trigger after handing back (no on/off flicker)
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


def past_cusp(P, j, pose, tol=0.03):
    """If the leg containing index j ends at a change of direction (a cusp) and the car at pose has reached or
    passed that cusp (along the leg's own heading, in its direction of travel), the first index of the next leg;
    otherwise j. Without this an overshoot at the end of a reversing leg left the tracker aiming at a point it had
    already passed - the car kept reversing (user, 28 Sep)."""
    d = P[j, 3]
    e = j
    while e + 1 < len(P) and P[e + 1, 3] == d:
        e += 1
    if e + 1 >= len(P):
        return j
    along = (pose[0] - P[e, 0]) * math.cos(P[e, 2]) + (pose[1] - P[e, 1]) * math.sin(P[e, 2])
    return e + 1 if d * along >= -tol else j


def _leg_vehicle(P, i, pose):
    """Rows of path P (x, y, heading, direction) from index i up to the next change of direction, moved into the
    vehicle frame of a car at pose: ((N, 3) poses, direction)."""
    if P is None or i >= len(P):
        return None
    d = P[i, 3]
    j = i
    while j + 1 < len(P) and P[j + 1, 3] == d:
        j += 1
    rest = P[i:j + 1]
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    dx, dy = rest[:, 0] - x, rest[:, 1] - y
    poses = np.column_stack([c * dx + s * dy, -s * dx + c * dy, rest[:, 2] - th])
    return poses, int(d)


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
        self.trigger_reason = None      # 'threat' (a real contact course) or 'stuck' (the car is held / going nowhere): a progress assist
        self.intent_commit = False      # set by RelayIntent: the commitment rules above apply
        self._release_t, self._fight_t, self._respect_t, self._stuck_t = 0.0, 0.0, 0.0, 0.0
        self.intent_stalled = False     # the car is stuck (pi/relay_assists.RelayIntent): the evasive steer runs to the end
        self.intent_attentive = True    # the stick moved recently (outside); False limits the hold (last point to steer)
        self._planner = None
        from .plan_service import PlanService
        self.plan_service = PlanService("inline")   # the relay switches it to a worker process
        self._job = None                # a search in progress
        self._mppi = None               # the local fallback controller (created on first use)
        self._track_i = 0
        self._x_goal_v = 1.0
        self.evade_pwm = None           # throttle the manoeuvre needs (cap, or reverse creep)
        self.replans = 0
        self.phase = None               # None, "EVADE" (going around) or "RETURN" (rejoining the original line)
        self.phase_t = 0.0
        self.ex = self.ey = self.eth = 0.0   # dead-reckoned pose since the swerve began
        self._trigger_for = 0.0
        self._backoffs = 0              # back-offs tried for the current obstacle
        self._backoff_from = None       # x where the current back-off began (line frame)
        self._pause = 0.0               # no new trigger until this runs out (after handing back)
        self._plan_wait = 0.0           # smoothed time from submitting a plan to its result (s)
        self._stall = 0.0               # s stopped on the planned path with the throttle held
        self._stalls = 0
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
        self._job = None                # a search still running is abandoned
        self.evade_side = 0
        self._trigger_for = 0.0
        self._backoffs = 0
        self._stall, self._stalls = 0.0, 0
        if why:
            self.info["evasive"] = why

    def _fail(self, why, v, pwm):
        """No way round found. If the car is held at the obstacle with the throttle still on, back off a little and
        search again from there (up to backoff_tries times) instead of sitting still; otherwise hand back and pause
        new triggers briefly, so the assist does not switch on and off every tick while the brake holds the car."""
        c = self.cfg
        if pwm > 0 and abs(v) < 0.08 and self._backoffs < c.backoff_tries:
            self._backoffs += 1
            self.phase, self.evading = "BACKOFF", True
            self._backoff_from = self.ex
            self.phase_t = 0.0
            self._job = None
            self.info["evasive"] = f"{why} - backing off to try again ({self._backoffs}/{c.backoff_tries})"
            return
        tried = self._backoffs
        self._stop_evading(why + (f" (after {tried} back-off{'s' if tried > 1 else ''})" if tried else ""))
        self._pause = c.retry_pause_s

    def _backoff(self, pts, steer, v):
        """Reverse straight back at a creep until backoff_m is gained (or the space behind runs out), then search
        for a way round again from there, standing still."""
        c = self.cfg
        rear = travel_distance_to_contact(pts, 0.0, -1, self.p, horizon=0.6, margin=0.03) if len(pts) else math.inf
        moved = self._backoff_from - self.ex
        if moved >= c.backoff_m or rear < c.backoff_rear_clear or self.phase_t > c.backoff_max_s:
            self._submit_plan(self._to_start_frame(pts), 0.0, 0.0)
            self.phase, self.evading = "PLANNING", False
            self.phase_t = 0.0
            self.info["evasive"] = f"backed off {moved * 100:.0f} cm - searching again"
            self.evade_pwm = 0.0
            return steer, 1
        self.evade_kappa = 0.0
        self.evade_pwm = -self.model.pwm_for_speed(c.reverse_speed)
        self.info["evasive"] = f"no way round from here - backing off ({moved * 100:.0f} cm)"
        return _steer(0.0, self.p), 2

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

    def _predicted_start(self, pts_s, v, kappa):
        """Where the car will be when the plan arrives: plan_lookahead_s along the arc it is on (latency
        compensation, as receding-horizon planners do). The current pose if that spot is not clear."""
        # how long plans really take here (measured, e.g. slower on the Pi), never less than the configured time
        s = max(0.0, v) * min(0.6, max(self.cfg.plan_lookahead_s, self._plan_wait))
        th = self.eth + kappa * s
        if abs(kappa) < 1e-6:
            x, y = self.ex + s * math.cos(self.eth), self.ey + s * math.sin(self.eth)
        else:
            x = self.ex + (math.sin(th) - math.sin(self.eth)) / kappa
            y = self.ey - (math.cos(th) - math.cos(self.eth)) / kappa
        if s > 0.005 and len(pts_s):
            c, sn = math.cos(th), math.sin(th)
            dx, dy = pts_s[:, 0] - x, pts_s[:, 1] - y
            lx, ly = c * dx + sn * dy, -sn * dx + c * dy
            ex = np.maximum(np.maximum(self.p.rear_x - lx, 0.0), lx - self.p.front_x)
            ey = np.maximum(np.abs(ly) - self.p.width / 2, 0.0)
            if np.hypot(ex, ey).min() > self.cfg.evade_margin:
                return (x, y, th)
        return (self.ex, self.ey, self.eth)

    def _submit_plan(self, pts_s, v=0.0, kappa=0.0):
        """Start a Hybrid A* search (forward, then with reversing) off the control loop, from where the car will be
        when the result arrives."""
        from .hybrid_astar import HybridAStar
        from .plan_service import BACKOFF_BUDGET_S, EVASIVE_BUDGET_S, plan_line_job
        kmax = min(self.kappa_max, self.cfg.plan_kappa_max)
        if self._planner is None:                   # kept here only for the cheap "is the plan still clear" check
            self._planner = HybridAStar(self.p, kappa_max=kmax)
        near = pts_s[(pts_s[:, 0] > self.ex - 1.0) & (pts_s[:, 0] < self.ex + 5.0) & (np.abs(pts_s[:, 1]) < 2.0)]
        svc = self.plan_service
        start = self._predicted_start(near, v, kappa)
        self._job = svc.submit(plan_line_job, self.p, kmax, near, start, self._x_goal_v,
                               self.cfg.plan_max_nodes, 4000, svc.budget(EVASIVE_BUDGET_S), svc.budget(BACKOFF_BUDGET_S))

    def _mppi_step(self, pts_s, v):
        """One MPPI control step toward the original line past the obstacle (adas/mppi.py)."""
        from .hybrid_astar import Grid
        from .mppi import MPPI
        c = self.cfg
        if self._mppi is None:
            self._mppi = MPPI(self.p, kappa_max=min(self.kappa_max, c.plan_kappa_max))
        near = pts_s[(np.abs(pts_s[:, 0] - self.ex) < 3.5) & (np.abs(pts_s[:, 1] - self.ey) < 2.0)]
        grid = Grid(near, self.ex - 1.5, self.ex + 4.0, self.ey - 2.0, self.ey + 2.0)
        return self._mppi.step(grid, (self.ex, self.ey, self.eth), max(0.2, min(abs(v), c.evade_speed)),
                               self._x_goal_v)

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
        j = past_cusp(P, j, (self.ex, self.ey, self.eth))
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

    def leg(self):
        """The part of the planned manoeuvre being driven now - up to the next change of direction - as
        (poses (N, 3) in the vehicle frame, direction), for the brake gate; None when no path is being followed."""
        if self.phase != "EXECUTE" or self._plan_poses is None:
            return None
        return _leg_vehicle(self._plan_poses, self._track_i, (self.ex, self.ey, self.eth))

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
        """Distance along the driver's path below which a held-back swerve must start (see AssistConfig)."""
        c = self.cfg
        v = abs(v)
        steer = v * c.lps_latency_s + math.sqrt(2.0 * c.lps_offset / max(c.plan_kappa_max, 0.1))
        stop = c.lps_fos * (c.lps_base + v * c.lps_reaction + v * v / (2 * c.lps_decel)) + v * (c.evade_confirm_s + 0.05)
        return max(steer, stop)

    def _brake_envelope(self, v):
        c = self.cfg
        v = abs(v)
        return c.defer_fos * (c.lps_base + v * c.lps_reaction + v * v / (2 * c.lps_decel)) + v * (c.evade_confirm_s + 0.05)

    def _defer(self, v, d_drv):
        """True: the intent model trusts this driver and the swerve stays back."""
        if self.cfg.intent_defer_to_brake:
            return d_drv > self._brake_envelope(v)
        return self.intent_attentive or d_drv > self._last_point_to_steer(v)

    def _evasive(self, dt, pts, steer, pwm, v):
        c = self.cfg
        k_drv = _kappa(steer, self.p)
        self._pwm_hist = (self._pwm_hist + [pwm])[-12:]
        self._respect_t = max(0.0, self._respect_t - dt)
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
            if self.intent_commit:
                self._release_t = self._release_t + dt if pwm <= 0 else 0.0
                against = abs(steer) > c.evade_driver_override and k_drv * self.evade_kappa < 0
                self._fight_t = self._fight_t + dt if (against and not self.intent_stalled) else max(0.0, self._fight_t - 2 * dt)
                if self._release_t > c.evade_release_s:
                    self._stop_evading("driver let go - handed back")
                    self._respect_t = c.respect_driver_s
                    return steer, None
                if self._fight_t > c.evade_fight_s:
                    self._stop_evading("driver steered against it - handed back")
                    self._respect_t = c.respect_driver_s
                    return steer, None
            else:
                if pwm <= 0:
                    self._stop_evading("driver let go / braked - handed back")
                    return steer, None
                if abs(steer) > c.evade_driver_override and not self.intent_stalled:
                    # (a stuck car: the driver's stick no longer cancels the way out - only letting go of the throttle does)
                    self._stop_evading("driver steered - handed back")
                    return steer, None
        if pwm <= 0 or len(pts) == 0:
            self._trigger_for = 0.0
            return steer, None
        if self.phase is None and self._pause > 0:          # just handed back after finding no way round
            self._pause -= dt
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
            self._stuck_t = self._stuck_t + dt if stuck else 0.0
            attentive = self.intent_k_rate is not None and self.intent_attentive
            ttc_limit = c.evade_ttc_attentive if attentive else c.evade_ttc
            too_far = ttc > ttc_limit and d_drv > self._last_point_to_steer(v) + c.trigger_lps_margin
            if not stuck and (v < c.evade_min_v or easing or d_drv >= look or too_far):
                self._trigger_for = 0.0
                return steer, None
            self._trigger_for += dt
            if self._trigger_for < c.evade_confirm_s:
                return steer, None
            if stuck and self.intent_commit and self._stuck_t < c.stuck_wait_s:
                self.info["evasive"] = "held at an obstacle - giving the driver a moment"
                return steer, None
            if self.intent_commit and self._respect_t > 0 and not self.intent_stalled and not stuck and d_drv > self._brake_envelope(v):
                self.info["evasive"] = "driver overrode it - staying back"
                return steer, None
            if self.intent_hold and not (stuck and self.intent_commit) and self._defer(v, d_drv):
                # the driver is on it (learned intent) - the brake still watches
                self.info["evasive"] = "driver is avoiding it - not intervening"
                return steer, None
            self.trigger_reason = "stuck" if (stuck or self.intent_stalled) else "threat"
            self.ex = self.ey = self.eth = 0.0      # the line frame: the car now, the driver's path straight ahead
            self._x_goal_v = self._x_goal(pts)
            self._submit_plan(pts, v, k_drv)        # off the control loop (adas/plan_service.py)
            self.phase, self.evading = "PLANNING", False
            self.phase_t = 0.0
            self._since_plan = 0.0
            dt_poll = 0.0
        else:
            dt_poll = dt
        # a search running in the planner: take its result when it is ready
        if self._job is not None and self._job.ready(dt_poll):
            path, self._job = self._job.result(), None
            if self.phase == "PLANNING":
                self._plan_wait = 0.7 * self._plan_wait + 0.3 * self.phase_t
                if path is None and not c.mppi_fallback:
                    self._fail("no safe way around - braking only", v, pwm)
                    return steer, 2
                if path is None:                    # MPPI steers locally (WAIT) while Hybrid A* keeps trying
                    self.phase, self.evading = "WAIT", True
                    if self._mppi is not None:
                        self._mppi.reset()
                else:
                    self._plan_poses, self._track_i = path, 0
                    self.phase, self.evading = "EXECUTE", True
                self.phase_t = 0.0
            elif path is None:
                self.phase = "WAIT"                 # keep the original line and keep trying; the brake holds the car
            else:
                self._plan_poses, self._track_i = path, 0
                self.phase = "EXECUTE"
                self.replans += 1
        self.phase_t += dt
        self._since_plan += dt
        if self.phase == "BACKOFF":
            return self._backoff(pts, steer, v)
        if self.phase == "PLANNING":
            if self.phase_t > c.plan_timeout_s:
                self._fail("planning took too long - braking only", v, pwm)
                return steer, 2
            self.info["evasive"] = "planning a way around"
            # slow to the manoeuvre speed while the plan is made: a way round that works at that speed must not be
            # lost because the driver's full throttle carries the car too close before it arrives (user, 28 Sep)
            self.evade_pwm = min(pwm, self.model.pwm_for_speed(c.evade_speed))
            return steer, 1                         # the driver keeps the steering meanwhile; the brake gate watches

        pts_s = self._to_start_frame(pts)
        if self._since_plan >= c.replan_period and self._job is None:
            self._since_plan = 0.0              # new scans: keep the plan if it is still clear, otherwise re-plan
            if self.phase == "WAIT":
                self._since_plan = -0.3             # searching again: every 0.6 s, not every 0.3 s
            if self.phase == "WAIT" or not self._path_still_clear(pts_s):
                self._submit_plan(pts_s, v if self.phase == "EXECUTE" else 0.0, self.evade_kappa)
                if self._job.ready(0.0):            # inline without latency: take it at once
                    path, self._job = self._job.result(), None
                    if path is None:
                        self.phase = "WAIT"
                    else:
                        self._plan_poses, self._track_i = path, 0
                        self.phase = "EXECUTE"
                        self.replans += 1
        if self.phase == "WAIT":
            if self.phase_t > c.evade_max_s:
                self._fail("no way around found", v, pwm)
                return steer, 2
            if c.mppi_fallback:
                # local fallback while Hybrid A* keeps searching: MPPI (adas/mppi.py) steers past what it can
                if self.ex > self._x_goal_v and abs(self.ey) < c.return_done_y + 0.03 and \
                        abs(self.eth) < math.radians(c.return_done_deg + 3):
                    self._stop_evading("back on the original line - handed back")
                    return steer, None
                kappa, ok = self._mppi_step(pts_s, v)
                if not ok:                          # even the best sampled way touches something: brake only
                    self._fail("no safe way around - braking only", v, pwm)
                    return steer, 2
                self.evade_kappa = kappa
                self.evade_pwm = min(pwm, self.model.pwm_for_speed(c.evade_speed))
                self.info["evasive"] = (f"fallback steering (MPPI) while re-planning ({self.ey * 100:+.0f} cm off "
                                        f"the line)")
                return _steer(kappa, self.p), 2
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
        # stalled on the path with the throttle held (the brake gate holds the car, e.g. the arc it is on still
        # points at the obstacle): search again from exactly where the car stands - that path starts at the car, so
        # the gate can check it - and back off after a few (user, 28 Sep: "no replanning, nothing")
        self._stall = self._stall + dt if abs(v) < 0.03 else 0.0
        if self._stall > c.stall_replan_s and self._job is None:
            self._stall = 0.0
            self._stalls += 1
            if self._stalls > c.stall_replans:
                self._stalls = 0
                self._fail("stuck on the planned path", v, pwm)
                return steer, 2
            self._submit_plan(pts_s, 0.0, 0.0)
            if self._job.ready(0.0):
                path, self._job = self._job.result(), None
                if path is not None:
                    self._plan_poses, self._track_i = path, 0
                    self.replans += 1
            self.info["evasive"] = f"stopped on the path - re-planning from here ({self._stalls}/{c.stall_replans})"
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
            if self.phase is not None and self.evade_pwm is not None:
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
