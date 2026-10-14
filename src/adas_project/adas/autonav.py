"""Autonomous point-to-point driving ("click to go"): Hybrid A* (adas/hybrid_astar.py, position goal) to a point
picked on the GUI, tracked with pure pursuit and a curvature/distance speed profile, re-planned when new scans block
the rest of the path. The path-predicted brake gate still has the last word; the relay hands control back as soon as
the operator steers or brakes (pi/relay_assists.py).

Planning runs off the control loop (adas/plan_service.py: a worker process on the car, as a planner node would in
ROS) so the relay's control loop never waits for it; the car holds still while a plan is being made.

Frame: the "start frame" is the vehicle frame when the goal was picked (x forward from the rear axle, y left); the
relay feeds the car's pose in it from scan matching.
"""
import math

import numpy as np

from .hybrid_astar import Grid, HybridAStar
from .plan_service import GOTO_BUDGET_S, PlanService, plan_point_job


class AutoNav:
    def __init__(self, params, speed_model, kappa_max=1.5, cruise=0.30, look=0.30, arrive=0.08, service=None):
        self.p, self.model = params, speed_model
        self.kappa_max, self.cruise, self.look, self.arrive = kappa_max, cruise, look, arrive
        self.service = service or PlanService("inline")
        self.state = "idle"               # idle | planning | driving
        self.path = None
        self.goal = None
        self.goal_heading = None          # rad in the start frame, or None: arrive pointing any way
        self.reverse_first = False
        self.fine_strokes, self._stroke_from, self._stroke_dir = 0, None, 0
        self._park, self.parking_space, self.parking_slot = False, None, None
        self.heading_tol = math.radians(20.0)
        self.x = self.y = self.th = 0.0
        self.i = 0
        self.msg = ""
        self.replan_t = 0.0
        self.replans = 0
        self.plan_ms = None
        self._job = None                  # the search in progress (plans from a cancelled goal are dropped)
        self._job_t = 0.0
        self.stall_s, self.stalls = 0.0, 0   # stopped on the path with the throttle held -> search again
        self._invalid_n = 0                  # consecutive path-validity checks that found the path ahead blocked
        self.reanchor = False                       # parking re-planned at the slot: the relay restarts its scan matcher's reference
        self._term_replans, self._creeps = 0, 0
        self._creep_dir, self._creep_flips = 0, 0
        self._stall_refine, self._refine_pts = False, None
        self._stall_replans, self._keep_stalls = 0, False
        self.park_request, self._refined = None, False   # parallel: strip + click (start frame) to measure the gap again alongside it
        self.goal_clicked, self.goal_moved = None, False
        self._arrive0, self.fixed = arrive, False   # fixed: a precomputed parking path, followed as it is (start_plan)
        # parallel parking runs in phases (as production park assists): "pull_out" (parked already: leave the gap), "approach" (the
        # click-to-go planner drives to the slot's pull-up pose, re-planning as it goes), then the gap is measured again alongside and
        # the "manoeuvre" is planned from where the car stands; None = not parallel parking
        self.park_phase, self._pull_up, self._tail, self._approach_pending, self._approaches = None, None, None, False, 0
        self._refines, self._stall_replans, self._keep_stalls, self._last_launch = 0, 0, False, None
        self._gaps_planned = (None, None)  # parallel: the latest plan's free distances bumper -> neighbour (rear, front) at the goal

    @property
    def active(self):
        return self.state != "idle"

    # ------------------------------------------------------------------ planning
    KERB_WALL_OUT_M = 0.04    # parking searches: the kerb as a wall this far behind the kerb line (the search keeps 6 cm to it, and its goal
                              # check needs half the width + 6 cm free around a goal 3.5 cm off the kerb: 2 cm made the goal 'unreachable')

    def _kerb_wall(self):
        """While parallel parking: the lane's kerb line (start frame) as obstacle points for the click-to-go planner - a 2D LiDAR does not
        see a kerb, and a search that does not know it plans across the pavement (30 Sep: a re-plan put the car 19 cm onto it)."""
        req = self.park_request or {}
        if not self.park_phase or "strip" not in req:
            return np.empty((0, 2))
        from .park_select import ParkingStrip
        sp, side, depth = req["strip"]
        strip = ParkingStrip(sp, side, depth)
        out = []
        # on the way to the slot the car stays on the road: the wall on the kerb line itself (the search keeps its 6 cm to it); searches
        # INTO the space need the wall further out, or a goal 3.5 cm off the kerb is 'unreachable'
        out_m = 0.0 if self.park_phase == "approach" else self.KERB_WALL_OUT_M
        for s in np.arange(0.0, strip.length + 1e-9, 0.04):
            x, y, th = strip.point_at(s)
            d = depth + out_m
            out.append((x - side * math.sin(th) * d, y + side * math.cos(th) * d))
        return np.array(out)

    def _launch(self, pts_start, start):
        import time
        pts_start = np.asarray(pts_start, float).reshape(-1, 2)
        self._last_launch = (pts_start, tuple(start))
        kerb = self._kerb_wall()
        if len(kerb):
            pts_start = np.vstack([pts_start, kerb])
        self.state = "planning"
        self._job_t = time.perf_counter()
        self._job = self.service.submit(plan_point_job, self.p, self.kappa_max, pts_start, start, self.goal,
                                        self.service.budget(GOTO_BUDGET_S), self.goal_heading, self.reverse_first)
        if self._job.ready(0.0):
            self._collect()

    def _collect(self):
        import time
        job, self._job = self._job, None
        if job is None:
            return
        path = job.result()
        if getattr(self, "_park", False) == "retry_out":  # stuck in the lane: out, then round again (vehicle frame of the launch)
            self._park = False
            request, slot, space = self.park_request, self.parking_slot, self.parking_space
            if path is None or len(path) < 2:              # no way out found: the re-plan from here as before
                self._stall_refine = True
                self.state = "driving"
                self._launch_refine(self.to_vehicle(self._refine_pts))
                return
            lx, ly, lth = self._retry_pose
            c, s = math.cos(lth), math.sin(lth)
            P = np.asarray(path, float)
            O = np.column_stack([lx + c * P[:, 0] - s * P[:, 1], ly + s * P[:, 0] + c * P[:, 1], P[:, 2] + lth, P[:, 3]])
            self._keep_stalls = True
            self.start_plan(O, tuple(O[-1, :2]), math.degrees(O[-1, 2]))
            self._keep_stalls = False
            self.park_request, self.parking_slot, self.parking_space = request, slot, space
            self._refined, self.park_phase = True, "pull_out"
            self.msg = f"pulling out to try {space} again"
            return
        if getattr(self, "_park", False) == "unpark":     # the way out of the parking space (vehicle frame at the go-to request)
            self._park = False
            if path is None or len(path) < 2:
                self._goto_after_unpark(self._unpark[3])   # no exit found: the search from here (it may reverse out)
                return
            P = np.asarray(path, float)
            self.start_plan(P, tuple(P[-1, :2]), math.degrees(P[-1, 2]))
            self.park_phase = "unpark"
            self.msg = "pulling out of the parking space"
            return
        if getattr(self, "_park", False) == "refine":     # the manoeuvre re-planned alongside the gap (vehicle frame of the launch)
            self._park = False
            self.plan_ms = self.service.last_ms if self.service.mode == "inline" else (time.perf_counter() - self._refine_job_t) * 1000
            if path is None or path.planned_path is None:
                if self._stall_refine and self.goal is not None and self._refine_pts is not None:
                    # a re-plan from mid-manoeuvre (the car held still, half in the gap): the gap planner starts from the lane; a
                    # search to the same goal pose with reversing allowed starts from exactly here
                    self._stall_refine, self.fixed, self.reverse_first = False, False, True
                    self.msg = "blocked - searching a way in from here"
                    self._launch(self._refine_pts, (self.x, self.y, self.th))
                    return
                self.state, self.path, self.msg = "idle", None, "no room to park here after all (the gap is shorter than it looked)"
                return
            lx, ly, lth = self._refine_pose              # the plan is in the vehicle frame of the LAUNCH (the car rolled on since)
            c, s = math.cos(lth), math.sin(lth)
            to_s = lambda B: np.column_stack([lx + c * B[:, 0] - s * B[:, 1], ly + s * B[:, 0] + c * B[:, 1]])
            P = np.array(path.planned_path, float)
            Ps = np.column_stack([to_s(P[:, :2]), P[:, 2] + lth, P[:, 3]])
            gx, gy, gh = path.parking_goal
            request, space, refines = self.park_request, path.selected_space_id, self._refines
            self._gaps_planned = tuple((getattr(path, "request", None) or {}).get("gaps_planned", (None, None)))
            slot = None if path.slot is None else to_s(np.asarray(path.slot, float))
            pu = getattr(path, "pull_up", None)
            from_car = getattr(path, "starts_at_car", True)
            far = pu is not None and math.hypot(pu[0], pu[1]) > self.REFINE_FAR_M
            if pu is not None and (not from_car or far):
                if self._approaches < self.APPROACH_MAX:
                    # no way from here in one curve, or another gap far away won: drive to its pull-up pose with the click-to-go planner
                    self.park_request, self.parking_space, self.parking_slot = request, space, slot
                    self._stall_refine = False
                    self._begin_way_to_slot(path, frame=(lx, ly, lth))
                    return
                if not from_car and math.hypot(pu[0], pu[1]) > 0.15 and getattr(self, "_refine_pts_s", None) is not None:
                    # the approaches are used up: the last resort is a search to the parking pose itself, reversing allowed, from exactly
                    # here (never give up while a way exists - user, 30 Sep: "the vehicle never gets permanently stuck")
                    g = to_s(np.array([[gx, gy]]))[0]
                    self.park_request, self.parking_space, self.parking_slot = request, space, slot
                    self.goal, self.goal_heading = (float(g[0]), float(g[1])), math.radians(gh) + lth
                    self.fixed, self.reverse_first, self.park_phase = False, True, "manoeuvre"
                    self.arrive, self.heading_tol = self.PARK_ARRIVE_M * 2, math.radians(self.PARK_HEADING_TOL_DEG)
                    self.msg = "searching a way into the space from here"
                    self._launch(self._refine_pts_s, (self.x, self.y, self.th))
                    return
            self._keep_stalls = True
            self.start_plan(Ps, tuple(to_s(np.array([[gx, gy]]))[0]), gh + math.degrees(lth))
            self.park_request, self.parking_space, self.parking_slot = request, space, slot
            self._keep_stalls, self._stall_refine = False, False
            self._refines = refines + 1
            self._refined = True
            self.park_phase = "manoeuvre"
            self.reanchor = True
            self.msg = f"parking at {space} (gap measured alongside)"
            return
        if getattr(self, "_park", False):                 # a ParkingPlan (adas/park_select.py) instead of a bare path
            self._park = False
            self.plan_ms = self.service.last_ms if self.service.mode == "inline" else (time.perf_counter() - self._job_t) * 1000
            if path is None or path.planned_path is None:
                self.state, self.path, self.msg = "idle", None, "no way to park there"
                return
            self.parking_space = path.selected_space_id
            self.parking_slot = None if getattr(path, "slot", None) is None else np.asarray(path.slot, float)
            gx, gy, gh = path.parking_goal
            if getattr(path, "pull_up", None) is not None and not getattr(path, "starts_at_car", True):
                self._begin_way_to_slot(path)
                return
            self.start_plan(path.planned_path, (gx, gy), gh)
            self.park_request = getattr(path, "request", None)
            self._gaps_planned = tuple((self.park_request or {}).get("gaps_planned", (None, None)))
            self.msg = f"parking in {path.selected_space_id}"
            return
        self.plan_ms = self.service.last_ms if self.service.mode == "inline" else (time.perf_counter() - self._job_t) * 1000
        if path is None and self.park_phase == "approach":
            # no way to the pull-up pose from here (the planner's cover is ~6 cm fatter than the car): plan the manoeuvre from where the
            # car stands (the gap planner's own approach, exact rectangle) - once; then give up with a reason, never wait for ever
            if self._approaches <= self.APPROACH_MAX and self.parking_slot is not None and getattr(self, "_last_launch", None) is not None:
                self._approaches += 1
                self.state = "driving"
                self.msg = "no way to the pull-up point - planning the manoeuvre from here"
                self._launch_refine(self.to_vehicle(self._last_launch[0]))
                return
            self.state, self.park_phase = "idle", None
            self.msg = "no way to the parking space from here"
            return
        if path is None:
            self.state = "idle"
            if not self.goal_moved and self.goal_heading is None and self.goal is not None and self.path is not None:
                # no way on to a clicked point: it may be on an obstacle only now seen from all sides - go as near as the car fits
                pts, start = self._last_launch
                q = self._near_goal(self.goal, pts, start)
                if q is not None and math.hypot(q[0] - start[0], q[1] - start[1]) < self.NEAR_MOVED_GOAL_M:
                    self.msg = "arrived as near as the car fits (the point is on an obstacle)"
                    return
                if q is not None:
                    self.goal, self.goal_moved = q, True
                    self.msg = "the point is on an obstacle - going as near as the car fits"
                    self._launch(pts, start)
                    return
            near = self.goal is not None and math.hypot(self.x - self.goal[0], self.y - self.goal[1]) < self.NEAR_MOVED_GOAL_M
            if self.goal_moved and near:                  # the click was on an obstacle and the car stands next to it: that is arriving
                self.msg = "arrived as near as the car fits (the point is on an obstacle)"
            else:
                self.msg = "no safe path to that point" if self.path is None else "path blocked - stopped"
            return
        self.path, self.i, self.replan_t = path, 0, 0.0
        self.state = "driving"
        if self.park_phase == "approach":
            self.msg = f"driving to {self.parking_space}" if self.replans == 0 else f"driving to {self.parking_space} (re-planned)"
            return
        self.msg = "driving to the goal" if self.replans == 0 else "re-planned around a new obstacle"

    NEAR_MOVED_GOAL_M = 0.35  # ... and stopped this near that point when the way on is blocked counts as arrived
    GOAL_MARGIN_M = 0.08      # a clicked point the car does not fit on (inside a box, against a wall) moves back toward the car until it does

    def _near_goal(self, goal, pts_start, origin):
        """The recovery when no path to a clicked point exists any more (the point turned out to be on an obstacle: a 2D LiDAR sees only
        a box's near face, so its inside looks free from afar): sweep the car's body from `origin` straight toward the point and return
        the last pose where it stays GOAL_MARGIN_M clear (exact rectangle) - the car's own position when it cannot get nearer - or None
        when there is nothing to judge by."""
        from .park_select import _obb_clearance
        pts = np.asarray(pts_start, float).reshape(-1, 2)
        g, o = np.asarray(goal, float), np.asarray(origin[:2], float)
        d = g - o
        L = float(np.hypot(*d))
        if not len(pts) or L < 0.1:
            return None
        th = math.atan2(d[1], d[0])
        steps = np.arange(0.0, L + 1e-9, 0.05)
        Q = np.column_stack([o[0] + np.cos(th) * steps, o[1] + np.sin(th) * steps, np.full(len(steps), th)])
        near = pts[np.hypot(pts[:, 0] - (o[0] + g[0]) / 2, pts[:, 1] - (o[1] + g[1]) / 2) < L / 2 + 0.6]
        ok = _obb_clearance(Q, near, self.p) >= self.GOAL_MARGIN_M
        n = len(ok) if ok.all() else int(np.argmin(ok))
        if n < 3:
            return float(o[0]), float(o[1])                     # it cannot get any nearer than where it stands
        return float(Q[n - 1, 0]), float(Q[n - 1, 1])

    def start(self, goal, pts_vehicle, heading=None, reverse_first=False, pull_out=None):
        """goal: (x, y) m in the vehicle frame now; heading: the direction to arrive in (rad, 0 = the car's heading
        now, + = left), or None for any. pull_out: (job fn, args) planning the way out of the parking space the car stands in (relay
        _pull_out_job) - followed first, then the search to the goal starts from where it ends. Starts planning; returns False if the
        request is refused."""
        self.x = self.y = self.th = 0.0
        self.goal = self.goal_clicked = (float(goal[0]), float(goal[1]))
        self.goal_heading = None if heading is None else float(heading)
        self.reverse_first = bool(reverse_first)          # parking: search with reversing allowed from the start
        self.goal_moved = False
        self.arrive, self.fixed, self.park_request = self._arrive0, False, None
        self.park_phase, self._tail, self._approach_pending = None, None, False
        self.fine_strokes, self._stroke_from, self._stroke_dir = 0, None, 0
        self.heading_tol = math.radians(8.0 if reverse_first else 20.0)   # a bay / slot needs the car straight
        self.path, self.replans = None, 0
        self.stall_s, self.stalls = 0.0, 0
        self.msg = "planning a path"
        pts = np.asarray(pts_vehicle, float).reshape(-1, 2)
        if pull_out is not None:                          # parked: out of the space first (a search from inside a tight gap finds nothing)
            import time
            self._unpark = (self.goal, self.goal_heading, self.reverse_first, pts)
            self._park, self.state, self._job_t = "unpark", "planning", time.perf_counter()
            self.msg = "pulling out of the parking space"
            self._job = self.service.submit(pull_out[0], *pull_out[1])
            if self._job.ready(0.0):
                self._collect()
            return self.state != "idle"
        self._launch(pts, (0.0, 0.0, 0.0))
        return self.state != "idle"

    def _goto_after_unpark(self, pts_start):
        """The pull-out is done (or there is none): the search to the clicked goal from where the car stands."""
        goal, gh, rf, _ = self._unpark
        self.park_phase, self.fixed = None, False
        self.goal, self.goal_heading, self.reverse_first = goal, gh, rf
        self.goal_clicked, self.goal_moved = goal, False
        self.arrive, self.heading_tol = self._arrive0, math.radians(8.0 if rf else 20.0)
        self.fine_strokes, self._stroke_from, self._stroke_dir = 0, None, 0
        self.path, self.replans, self.stall_s, self.stalls = None, 0, 0.0, 0
        self.msg = "planning a path"
        self._launch(pts_start, (self.x, self.y, self.th))

    PARK_HEADING_TOL_DEG = 4.0   # parked straight: within 4 deg of the slot / bay (straightening strokes below 2 deg when possible)
    PARK_ARRIVE_M = 0.03      # a bay / space is entered to a few cm, not to the 8 cm of click-to-go

    def start_plan(self, path, goal, heading_deg):
        """Follow a precomputed parking path (adas/park_select.py: ParkingPlan, vehicle frame now) instead of searching for one:
        goal = (x, y) of the rear axle, heading_deg the heading to end in (0 = the car's heading now). The path is driven as
        planned - no goal-based re-planning (it would call the few cm of wall / neighbour clearance of a parking path
        'blocked'); the brake gate keeps the last word and a car held still on the path gives up after 2 s."""
        self.x = self.y = self.th = 0.0
        self.goal = (float(goal[0]), float(goal[1]))
        self.goal_clicked, self.goal_moved = None, False
        self.goal_heading = math.radians(float(heading_deg))
        self.reverse_first, self.fixed = True, True
        self.arrive = self.PARK_ARRIVE_M
        self.heading_tol = math.radians(self.PARK_HEADING_TOL_DEG)
        self.fine_strokes, self._stroke_from, self._stroke_dir = 0, None, 0
        self.path, self.i, self.replans, self.replan_t = np.asarray(path, float), 0, 0, 0.0
        self.stall_s, self.stalls = 0.0, 0
        self._job = None
        self._term_replans, self._creeps = 0, 0
        self.park_request, self._refined = None, False
        self._wall_shift = 0.0
        self._creep_dir, self._creep_flips = 0, 0
        self._refines, self._shifts = 0, 0
        self._stall_replans = getattr(self, "_stall_replans", 0) if self._keep_stalls else 0
        self.state = "driving"
        self.msg = "parking"
        return True

    def start_parking(self, fn, *args):
        """Plan a parking manoeuvre in the service (fn(*args) -> ParkingPlan, vehicle frame now) and follow it as soon as it is
        ready; the car holds still while it is being made. Returns False when a plan was refused at once."""
        self.x = self.y = self.th = 0.0
        self.path, self.replans, self._park = None, 0, True
        self.goal, self.parking_space, self.parking_slot = None, None, None     # known once the plan is (start_plan)
        self.goal_clicked, self.goal_moved = None, False
        self.fixed = True
        self.park_phase, self._pull_up, self._tail, self._approach_pending, self._approaches = None, None, None, False, 0
        self._refines, self._stall_replans = 0, 0
        self.msg = "planning the parking manoeuvre"
        import time
        self.state = "planning"
        self._job_t = time.perf_counter()
        self._job = self.service.submit(fn, *args)
        if self._job.ready(0.0):
            self._collect()
        return self.state != "idle"

    APPROACH_ARRIVE_M, APPROACH_HEADING_DEG, APPROACH_MAX = 0.10, 6.0, 3   # at the approach point: this near, this straight; approaches per park
    APPROACH_BACK_M = 1.2     # the approach ends this far before the pull-up pose: beside the gap (measured there at short range), with room
                              # for the gap planner's own curve + straight run-in that brings the car to the pull-up pose exactly aligned
                              # (0.8 m left the curve 0.45 m: a 6 cm side step + 4.5 deg became a 356 deg Dubins loop - refused)

    def _begin_way_to_slot(self, plan, frame=None):
        """A parallel ParkingPlan whose manoeuvre starts at a pull-up pose away from the car (in the vehicle frame of `frame`, a pose in the
        start frame; None = the first plan, made in the start frame): keep the manoeuvre for the map, pull out first when the plan says the
        car stands parked, then drive to the pull-up pose."""
        fx, fy, fth = (0.0, 0.0, 0.0) if frame is None else frame
        c, s = math.cos(fth), math.sin(fth)
        to_s = lambda B: np.column_stack([fx + c * B[:, 0] - s * B[:, 1], fy + s * B[:, 0] + c * B[:, 1]])
        P = np.asarray(plan.planned_path, float)
        pu = np.asarray(plan.pull_up, float)
        if getattr(plan, "starts_at_car", True):         # (a path from the car: the manoeuvre is what follows the pull-up pose)
            P = P[int(np.argmin(np.hypot(P[:, 0] - pu[0], P[:, 1] - pu[1]))):]
        self._tail = np.column_stack([to_s(P[:, :2]), P[:, 2] + fth, P[:, 3]])
        self._pull_up = (*to_s(pu[None, :2])[0], pu[2] + fth)
        if frame is None:
            self.parking_slot = None if plan.slot is None else to_s(np.asarray(plan.slot, float))
            self.park_request = plan.request
        self._gaps_planned = tuple((getattr(plan, "request", None) or {}).get("gaps_planned", (None, None)))
        request, slot, space = self.park_request, self.parking_slot, self.parking_space
        out = getattr(plan, "pull_out", None)
        if out is not None and self.park_phase is None:
            O = np.asarray(out, float)
            O = np.column_stack([to_s(O[:, :2]), O[:, 2] + fth, O[:, 3]])
            self.start_plan(O, tuple(O[-1, :2]), math.degrees(O[-1, 2]))
            self.park_request, self.parking_slot, self.parking_space = request, slot, space
            self._refined, self.park_phase = True, "pull_out"
            self.msg = f"pulling out to park at {space}"
            return
        self.park_phase, self._approach_pending = "approach", True
        self.state, self._job, self.path = "planning", None, None
        self.msg = f"planning the way to {space}"

    def _start_approach(self, pts_vehicle):
        """Drive to the pull-up pose with the click-to-go planner (Hybrid A*, re-planned when the way is blocked, stall recovery) - or,
        standing there already, measure the gap and plan the manoeuvre at once."""
        self._approach_pending = False
        self._approaches += 1                           # (bounds the approach -> measure -> approach cycle: APPROACH_MAX)
        px, py, pth = self._pull_up
        c, s = math.cos(pth), math.sin(pth)
        along, lat = c * (self.x - px) + s * (self.y - py), -s * (self.x - px) + c * (self.y - py)
        head = abs(math.remainder(self.th - pth, 2 * math.pi))
        if -self.APPROACH_BACK_M - 0.4 <= along <= 0.05 and abs(lat) < 0.06 and head < math.radians(self.APPROACH_HEADING_DEG):
            return self._at_pull_up(pts_vehicle)        # alongside already, in line and straight: measure and plan from here
        self.park_phase = "approach"
        self.goal = (float(px - self.APPROACH_BACK_M * c), float(py - self.APPROACH_BACK_M * s))
        self.goal_heading = float(pth)
        self.goal_clicked, self.goal_moved = None, False
        self.fixed, self.reverse_first = False, False
        self.arrive, self.heading_tol = self.APPROACH_ARRIVE_M, math.radians(self.APPROACH_HEADING_DEG)
        self.fine_strokes, self._stroke_from, self._stroke_dir = 0, None, 0
        self.stall_s, self.stalls, self.replans = 0.0, 0, 0
        self.msg = f"planning the way to {self.parking_space}"
        self._launch(self.to_start(np.asarray(pts_vehicle, float).reshape(-1, 2)), (self.x, self.y, self.th))

    def _at_pull_up(self, pts_vehicle):
        """At the pull-up pose: the gap alongside is measured at short range and the manoeuvre planned from exactly here."""
        self.park_phase = "manoeuvre"
        self._refined = False
        self.state = "driving"
        self.msg = "measuring the gap"
        self._launch_refine(pts_vehicle)
        return (0.0, 0.0) if self.state == "planning" else self.step(0.0, (self.x, self.y, self.th), pts_vehicle, 0.0)

    REFINE_NEAR_M = 0.9       # parallel: the gap is measured again when the car is this close to the reversing point
    REFINE_FAR_M, REFINE_MAX = 2.0, 3
    PARK_STALL_S, PARK_STALL_REPLANS = 2.0, 3   # held still this long on a parking path -> plan again from here, at most this often   # ... and once more if that re-selection moved the slot further than this away (at most 3 times)

    def _cusp_index(self):
        """Index of the first reverse pose after the forward approach, or None when the path reverses from the start / never reverses."""
        if self.path is None:
            return None
        d = self.path[:, 3]
        j = np.where((d[1:] < 0) & (d[:-1] > 0))[0]
        return int(j[0]) + 1 if len(j) else None

    WALL_NEAR_M, WALL_BAND_M, WALL_BLEND, WALL_FREEZE_M = 0.8, 0.25, 0.4, 0.04   # bay: measure the rear wall within this distance of the goal, points
                                                            # within this of the expected wall line; share of a new measurement taken

    def _wall_stop(self, pts_vehicle):
        """Bay (perpendicular) parking: on the last reverse leg the LiDAR sees the rear wall - move the goal and the rest of the leg along
        the bay axis so the rear bumper stops `stop_gap` from the wall as MEASURED, not as the camera's range and the odometry put it
        (parking sensors do the same on a car). Only the returns straight behind the car (within its width + 5 cm of the centre line)
        count - not a neighbour's wall."""
        req = self.park_request
        if req is None or "wall" not in req or self.path is None or not len(pts_vehicle) or self.i >= len(self.path):
            return
        if self.path[self.i, 3] >= 0 or (self.path[self.i:, 3] > 0).any():
            return                                          # not on the last (reverse) leg yet
        dist = math.hypot(self.x - self.goal[0], self.y - self.goal[1])
        if dist > self.WALL_NEAR_M or dist < self.WALL_FREEZE_M:
            return                                          # (at the stop: no more chasing the LiDAR's noise)
        m, n = np.asarray(req["wall"][0], float), np.asarray(req["wall"][1], float)
        t = np.array([-n[1], n[0]])
        P = self.to_start(np.asarray(pts_vehicle, float).reshape(-1, 2))
        gl = np.array(self.goal)
        d = (P - m) @ n                                      # + : off the wall toward the bay's mouth
        lat = (P - gl) @ t
        sel = (np.abs(d - self._wall_shift) < self.WALL_BAND_M) & (np.abs(lat) < self.p.width / 2 + 0.05)
        if int(sel.sum()) < 6:
            return
        wall = float(np.median(d[sel]))                     # where the wall is, measured, relative to where the plan put it
        step = self.WALL_BLEND * (wall - self._wall_shift)
        self._wall_shift += step
        move = step * n
        self.goal = (self.goal[0] + move[0], self.goal[1] + move[1])
        self.path[self.i:, :2] += move

    def _refine_due(self):
        """Parallel parking, not re-planned yet, the car near the reversing point (or the path reverses at once): time to measure the gap."""
        if self.park_request is None or "strip" not in self.park_request or self._refined or not self.fixed or self.path is None \
                or self.park_phase in ("pull_out", "approach"):
            return False
        cusp = self._cusp_index()
        ref = self.path[cusp, :2] if cusp is not None else self.path[0, :2]
        return math.hypot(self.x - ref[0], self.y - ref[1]) <= self.REFINE_NEAR_M

    def _launch_refine(self, pts_vehicle):
        """Submit the re-plan alongside the gap: strip, click and the planned slot moved into the vehicle frame now."""
        from .plan_service import plan_refine_job
        self._refined = True
        if self.parking_slot is None:
            return False
        c, s = math.cos(self.th), math.sin(self.th)

        def to_v(P):
            P = np.asarray(P, float).reshape(-1, 2)
            dx, dy = P[:, 0] - self.x, P[:, 1] - self.y
            return np.column_stack([c * dx + s * dy, -s * dx + c * dy])
        strip_pts, side, depth = self.park_request["strip"]
        import time
        self._refine_job_t = time.perf_counter()
        self._refine_pose = (self.x, self.y, self.th)
        self._refine_pts_s = self.to_start(np.asarray(pts_vehicle, float).reshape(-1, 2))    # (the last resort searches with these)
        self._park = "refine"
        self.state = "planning"
        self._job = self.service.submit(plan_refine_job, self.p, to_v(strip_pts), side, depth, tuple(to_v(self.park_request["click"])[0]),
                                        to_v(self.parking_slot), np.asarray(pts_vehicle, float).reshape(-1, 2))
        if self._job.ready(0.0):
            self._collect()
        return True

    def cancel(self, why="cancelled"):
        self._job = None
        self.state = "idle"
        self.msg = why

    # ------------------------------------------------------------------ frames
    def to_vehicle(self, pts_start):
        c, s = math.cos(self.th), math.sin(self.th)
        dx, dy = pts_start[:, 0] - self.x, pts_start[:, 1] - self.y
        return np.column_stack([c * dx + s * dy, -s * dx + c * dy])

    def to_start(self, pts_vehicle):
        c, s = math.cos(self.th), math.sin(self.th)
        return np.column_stack([self.x + c * pts_vehicle[:, 0] - s * pts_vehicle[:, 1],
                                self.y + s * pts_vehicle[:, 0] + c * pts_vehicle[:, 1]])

    # ------------------------------------------------------------------ driving
    def step(self, dt, pose, pts_vehicle, v, held=True):
        """pose: the car in the start frame. Returns (curvature, target speed m/s, + forward), or None when
        finished or stopped. While a plan is being made it returns (0, 0): hold still."""
        if self.state == "planning" and self._job is not None and self._job.ready(dt):
            self._collect()
        if self.state == "idle":
            return None
        self.x, self.y, self.th = pose
        if self._approach_pending and len(pts_vehicle):
            out = self._start_approach(pts_vehicle)
            if out is not None or self.state == "idle":
                return out
        if self.state == "planning":
            return 0.0, 0.0
        if self.park_phase == "unpark" and self.path is not None and len(pts_vehicle) and (
                self.i >= len(self.path) - 3 or math.hypot(self.goal[0] - self.x, self.goal[1] - self.y) < 0.05
                or self.stall_s > self.PARK_STALL_S):
            # out on the road (or held while pulling out): on to the clicked goal, searched from exactly here
            self._goto_after_unpark(self.to_start(np.asarray(pts_vehicle, float).reshape(-1, 2)))
            return (0.0, 0.0) if self.state == "planning" else (None if self.state == "idle" else self.step(0.0, pose, pts_vehicle, v, held))
        if self.park_phase == "pull_out" and self.path is not None and (
                self.i >= len(self.path) - 3 or math.hypot(self.goal[0] - self.x, self.goal[1] - self.y) < 0.05):
            self.park_phase, self._approach_pending = "approach", True     # out on the road: on to the pull-up pose
            return (0.0, 0.0) if not len(pts_vehicle) else (self._start_approach(pts_vehicle) or (0.0, 0.0))
        self._wall_stop(pts_vehicle)
        if self._refine_due() and len(pts_vehicle):
            if self._launch_refine(pts_vehicle):            # alongside the gap: measure it again at short range and re-plan
                self.msg = "measuring the gap"
                return (0.0, 0.0) if self.state == "planning" else self.step(dt, pose, pts_vehicle, v, held)
        gx, gy = self.goal
        dist_goal = math.hypot(gx - self.x, gy - self.y)
        head_err = 0.0 if self.goal_heading is None else             abs((self.th - self.goal_heading + math.pi) % (2 * math.pi) - math.pi)
        if self.park_phase == "approach" and len(pts_vehicle):
            at_end = self.path is not None and self.i >= len(self.path) - 2 and dist_goal < 2 * self.arrive
            if (dist_goal < self.arrive or at_end) and head_err < self.heading_tol:
                return self._at_pull_up(pts_vehicle)
            if at_end and self.fine_strokes < 12:
                # at the approach point but crooked (the steering never follows a curve exactly): straighten first - the gap planner's
                # way in from here is one curve and a run-in (30 Sep: arriving 18 deg off gave no way in, three approaches, then stopped)
                out = self._straighten(head_err, pose, pts_vehicle)
                if out is not None:
                    return out
            if at_end:
                return self._at_pull_up(pts_vehicle)
        if self.fixed and self.park_phase == "manoeuvre" and self.goal_heading is not None and self.path is not None \
                and (self.i >= len(self.path) - 3 or self._next_leg() is None) and dist_goal < self.CREEP_M \
                and head_err < math.radians(8) and len(pts_vehicle):
            meas = self._neighbours_here(pts_vehicle)
            if meas is not None:
                # in the slot between two neighbours: where to stop is measured, as parking sensors do - centred between them (plus the
                # slot's planned offset in a long gap) and in line with their road sides - not taken from the odometry since the gap was
                # measured (along a street it drifts: 9 cm in one validation run), and before 'arrived' is decided
                c, s = math.cos(self.goal_heading), math.sin(self.goal_heading)
                ea = c * (gx - self.x) + s * (gy - self.y) if meas[0] is None else meas[0]
                el = -s * (gx - self.x) + c * (gy - self.y) if meas[1] is None else meas[1]
                self.goal = gx, gy = (self.x + c * ea - s * el, self.y + s * ea + c * el)
                dist_goal = math.hypot(gx - self.x, gy - self.y)
        if dist_goal < self.arrive and head_err < self.heading_tol:
            self.cancel("arrived as near as the car fits (the point is on an obstacle)" if self.goal_moved else "arrived")
            return None
        in_line = False
        if self.fixed and self.goal_heading is not None and self.i >= len(self.path) - 3:
            cg, sg = math.cos(self.goal_heading), math.sin(self.goal_heading)
            in_line = abs(-sg * (gx - self.x) + cg * (gy - self.y)) < self.LAT_MAX_M and abs(cg * (gx - self.x) + sg * (gy - self.y)) < 1.0 \
                and head_err < math.radians(20)
        if self.fixed and self.goal_heading is not None and self.i >= len(self.path) - 3 and (dist_goal < self.CREEP_M or in_line):
            # (in line with the slot but further along it than CREEP_M - a long slot: creeping along is the move, not a search from
            # here, which cannot even take a goal 6 cm off a wall - 30 Sep: 'path blocked - stopped' with the car on its slot)
            # at the end of the parking path: straighten first, then take the along error out by creeping in line (a lateral error that
            # is left cannot be driven out straight - it is reported), as a driver centres the car between the others
            c, s = math.cos(self.goal_heading), math.sin(self.goal_heading)
            e_along = c * (gx - self.x) + s * (gy - self.y)             # (the goal corrected by the neighbours above when they are seen)
            e_lat = -s * (gx - self.x) + c * (gy - self.y)
            if head_err > 0.5 * self.heading_tol and self.fine_strokes < 10:
                out = self._straighten(head_err, pose, pts_vehicle)
                if out is not None:
                    return out
            if self.LAT_TOL_M < abs(e_lat) < self.LAT_MAX_M and self._shifts < self.SHIFT_MAX and len(pts_vehicle):
                P = self._lateral_shift(e_lat, pts_vehicle)
                if P is not None:               # off to the side (the S never ends exactly where planned with real steering): shift over
                    self._shifts += 1
                    self.path, self.i, self._creeps, self._creep_flips, self._creep_dir = P, 0, 0, 0, 0
                    self.msg = f"correcting the side offset ({abs(e_lat) * 100:.0f} cm)"
                    return 0.0, 0.0
            if abs(e_along) > self.PARK_ALONG_TOL_M and abs(e_lat) <= self.CREEP_LATERAL_M and self._creeps < self.CREEP_MAX                     and self._creep_flips < self.CREEP_FLIPS:
                self._creeps += 1
                return self._creep_to_goal(head_err)
            self.cancel("arrived" if head_err < self.heading_tol else f"arrived ({math.degrees(head_err):.0f} deg off the chosen heading)")
            return None
        if self.fixed and self.goal_heading is not None and self.i >= len(self.path) - 3 and dist_goal >= 2 * self.arrive \
                and self._term_replans < self.TERM_REPLANS and len(pts_vehicle):
            # the end of the parking path is reached but the car is not where the plan ends (the steering needs time to swap the arcs
            # of the S): search a short correction from exactly here (the scan shows the neighbours), as a driver pulls out and in again
            self._term_replans += 1
            self.fixed = False
            self.replans += 1
            self.msg = f"correcting the last metres ({self._term_replans}/{self.TERM_REPLANS})"
            self._launch(self.to_start(pts_vehicle), (self.x, self.y, self.th))
            return (0.0, 0.0) if self.state == "planning" else None
        if self.goal_heading is not None and self.i >= len(self.path) - 2 and dist_goal < 2 * self.arrive:
            if self.reverse_first and head_err > 0.5 * self.heading_tol and self.fine_strokes < 10:
                out = self._straighten(head_err, pose, pts_vehicle)
                if out is not None:
                    return out
            self.cancel("arrived" if head_err < self.heading_tol else f"arrived ({math.degrees(head_err):.0f} deg off the chosen heading)")
            return None
        P = self.path
        self.replan_t += dt
        if self.replan_t > self.VALID_EVERY_S and len(pts_vehicle) and not self.fixed:   # new scans: keep the plan while it is clear
            # re-plan only when the path ahead is really invalid (Nav2 "replan only if path becomes invalid"): the stretch from
            # VALID_FROM_M to VALID_TO_M ahead (the first part is the brake gate's business - near a wall the car's own tracking error
            # makes it look blocked), blocked on VALID_N checks in a row (one noisy scan is not a blockage); meanwhile slow down
            self.replan_t = 0.0
            pts_s = self.to_start(pts_vehicle)
            ahead = self._path_ahead(P)
            blocked = clear_block = False
            if ahead is not None:
                ha = HybridAStar(self.p, kappa_max=self.kappa_max)
                g = Grid(pts_s, min(self.x, gx) - 1.5, max(self.x, gx) + 1.5, min(self.y, gy) - 1.5, max(self.y, gy) + 1.5)
                c_ahead = ha.clearance(g, ahead)
                blocked = c_ahead < 0.01
                clear_block = c_ahead < self.HARD_BLOCK_M      # an obstacle squarely on the path: no waiting
            self._invalid_n = self._invalid_n + 1 if blocked else 0
            if self._invalid_n >= self.VALID_N or clear_block:
                self._invalid_n = 0
                self.replans += 1
                self.msg = "path blocked - re-planning"
                self._launch(pts_s, (self.x, self.y, self.th))
                if self.state != "driving":
                    return (0.0, 0.0) if self.state == "planning" else None
                P = self.path
        # stopped on the path although the operator holds the throttle (the brake gate holds the car): search
        # again from exactly here - that path starts at the car - and give up after a few (user, 28 Sep)
        # stalled = held but no progress: moved less than STALL_MOVE_M since the timer started (a car the brake gate lets creep and
        # stops again, over and over, is stalled as well - 30 Sep: 'stuck' for minutes at 0-6 cm/s behind a parked car)
        # (letting go of the throttle pauses the timer, it does not reset it: a driver who pulses RT - 2 s on, 1 s off - must not keep a
        # held car from ever counting as stalled; 30 Sep: stuck mid-S for a minute that way)
        ref = getattr(self, "_stall_ref", None)
        if ref is None or math.hypot(self.x - ref[0], self.y - ref[1]) > self.STALL_MOVE_M:
            self._stall_ref, self.stall_s = (self.x, self.y), 0.0
        elif held:
            self.stall_s += dt
        if self.fixed and self.stall_s > self.STROKE_END_S and self._leg_left() < self.STROKE_END_M and self._next_leg() is not None:
            # held by the gate just short of the end of a stroke (it keeps CREEP_MIN_M + its margin to what the stroke ends at): the
            # stroke is done - on with the next one from here
            self.i, self.stall_s, self._stall_ref = self._next_leg(), 0.0, None
            self.msg = "next stroke"
        if self.park_phase == "pull_out" and self.stall_s > self.PARK_STALL_S:
            # held while pulling out: the click-to-go planner takes over from where the car stands (it may reverse)
            self.stall_s, self._stall_ref = 0.0, None
            self.park_phase, self._approach_pending = "approach", True
            return (0.0, 0.0) if not len(pts_vehicle) else (self._start_approach(pts_vehicle) or (0.0, 0.0))
        if self.fixed and self.stall_s > self.PARK_STALL_S and self.park_phase != "unpark":
            # held on the parking path (the brake gate saw something closer than the plan expected): plan again from exactly here -
            # parallel: measure the gap again and re-plan; otherwise a search to the same goal pose with reversing allowed - and give
            # up only after a few tries, never wait for ever (user, 30 Sep: "the vehicle never gets permanently stuck")
            self.stall_s = 0.0
            self._stall_replans += 1
            if self._stall_replans > self.PARK_STALL_REPLANS or not len(pts_vehicle):
                self.cancel("stopped on the parking path - blocked")
                return None
            self.replans += 1
            self.msg = f"blocked - planning again from here ({self._stall_replans}/{self.PARK_STALL_REPLANS})"
            if self.park_request is not None and self.parking_slot is not None and self._in_lane() and self.park_phase == "manoeuvre":
                # stuck part-way into the lane (a tight gap, the gate holding the car): pull out from exactly here and go round again - the
                # exit strokes work from inside a gap, a search from there does not (30 Sep: tight gap and RT-pulse runs ended stuck)
                self._retry_by_pull_out(pts_vehicle)
            elif self.park_request is not None and self.parking_slot is not None:
                self._stall_refine, self._refine_pts = True, self.to_start(np.asarray(pts_vehicle, float).reshape(-1, 2))
                self._launch_refine(pts_vehicle)
            else:
                self.fixed, self.reverse_first = False, True
                self._launch(self.to_start(pts_vehicle), (self.x, self.y, self.th))
            return (0.0, 0.0) if self.state == "planning" else None
        if self.stall_s > 0.8 and len(pts_vehicle) and not self.fixed:
            self.stall_s = 0.0
            self.stalls += 1
            if self.stalls > 3:
                self.cancel("stuck - no way through from here")
                return None
            self.replans += 1
            self.msg = f"stopped on the path - re-planning from here ({self.stalls}/3)"
            self._launch(self.to_start(pts_vehicle), (self.x, self.y, self.th))
            if self.state != "driving":
                return (0.0, 0.0) if self.state == "planning" else None
            P = self.path
        from .assists import past_cusp
        e = len(P) - 1
        if self.fixed:                               # a parking path retraces itself after a cusp (in / out / in strokes): look for the
            e = self.i                               # nearest point on the current leg only, or the tracker skips into the next leg
            while e + 1 < len(P) and P[e + 1, 3] == P[self.i, 3]:
                e += 1
        d = np.hypot(P[self.i:min(self.i + 60, e + 1), 0] - self.x, P[self.i:min(self.i + 60, e + 1), 1] - self.y)
        self.i += int(np.argmin(d)) if len(d) else 0
        self.i = past_cusp(P, self.i, (self.x, self.y, self.th))
        direction = int(P[self.i, 3])
        k = self.i
        creep = self.fixed and direction < 0            # the reverse stretch of a parking path: slow and a short look-ahead, the steering
        look = self.PARK_LOOK_M if creep else self.look  # servo needs time to swap the two arcs of the S
        while k + 1 < len(P) and P[k + 1, 3] == direction and \
                math.hypot(P[k, 0] - P[self.i, 0], P[k, 1] - P[self.i, 1]) < look:
            k += 1
        # the last stretch: aim at the goal point itself - unless a heading was chosen, then follow the planned
        # path to its end (the Dubins approach arrives pointing the right way)
        tgt = P[k:k + 1, :2] if k < len(P) - 1 or direction < 0 or self.goal_heading is not None             else np.array([[gx, gy]])
        xl, yl = self.to_vehicle(tgt)[0]
        dd = xl * xl + yl * yl
        kappa = max(-self.kappa_max, min(self.kappa_max, 2.0 * yl / dd if dd > 1e-4 else 0.0))
        # slow for tight curves (lateral acceleration ~0.12 g) and approaching the goal / a change of direction
        to_stop = dist_goal if k == len(P) - 1 else math.hypot(P[k, 0] - self.x, P[k, 1] - self.y)
        speed = min(self.PARK_REVERSE_V if creep else self.cruise, math.sqrt(1.2 / max(abs(kappa), 1e-3)), 0.12 + 0.6 * to_stop)
        if self._invalid_n and not self.fixed:
            speed = min(speed, self.PENDING_V)          # a blockage ahead is being confirmed: slow, do not stop
        return kappa, direction * speed

    VALID_EVERY_S, VALID_N = 0.25, 2           # path validity: checked this often, invalid after this many blocked checks in a row
    VALID_FROM_M, VALID_TO_M = 0.30, 2.5       # ... on this stretch ahead of the car
    PENDING_V = 0.15                           # m/s while a blockage ahead is being confirmed
    HARD_BLOCK_M = -0.02                       # clearance below this on the stretch ahead: re-plan at once (only marginal ones wait)

    def _path_ahead(self, P):
        """Poses of the current path from VALID_FROM_M to VALID_TO_M ahead of the car (along the path), every other one."""
        if P is None or len(P) - self.i <= 2:
            return None
        seg = np.asarray(P, float)[self.i:, :3]
        s_along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(seg[:, :2], axis=0).T))])
        m = (s_along >= self.VALID_FROM_M) & (s_along <= self.VALID_TO_M)
        return seg[m][::2] if m.sum() >= 2 else None

    STALL_MOVE_M = 0.015      # progress: less than this in the stall time = stalled
    STROKE_END_S, STROKE_END_M = 0.6, 0.08            # held this long within this of the end of a parking stroke = the stroke is done

    def _leg_end(self):
        """Index of the last pose of the leg being driven (up to the next change of direction)."""
        P, e = self.path, self.i
        while e + 1 < len(P) and P[e + 1, 3] == P[self.i, 3]:
            e += 1
        return e

    def _leg_left(self):
        """Path length left on the leg being driven (m)."""
        if self.path is None or self.i >= len(self.path):
            return 0.0
        e = self._leg_end()
        seg = np.asarray(self.path, float)[self.i:e + 1, :2]
        return float(np.hypot(*np.diff(seg, axis=0).T).sum()) + math.hypot(seg[0, 0] - self.x, seg[0, 1] - self.y) if len(seg) else 0.0

    def _next_leg(self):
        """Index of the first pose of the next leg, or None on the last one."""
        if self.path is None:
            return None
        e = self._leg_end()
        return e + 1 if e + 1 < len(self.path) else None

    TERM_REPLANS = 2                                  # searches from the pose reached when a parking path ended short
    PARK_REVERSE_V, PARK_LOOK_M = 0.14, 0.15          # m/s and m while reversing along a parking path

    CREEP_LATERAL_M = 0.05
    CREEP_FLIPS = 4
    PARK_ALONG_TOL_M = 0.015  # parking: along the slot to 1.5 cm
    CREEP_M, CREEP_MAX, CREEP_V = 0.45, 600, 0.08     # finish within this distance by creeping (ticks are capped: never a loop) - the goal
                                                      # measured from the neighbours can move ~20 cm along the slot; a search from inside the
                                                      # gap (TERM_REPLANS) is for a car further off than this

    def _creep_to_goal(self, head_err):
        """The last few cm: creep along the goal heading toward the goal - forward or in reverse, whichever side it lies - steering only
        to keep the heading. Returns (curvature, signed speed)."""
        c, s = math.cos(self.goal_heading), math.sin(self.goal_heading)
        e_along = c * (self.goal[0] - self.x) + s * (self.goal[1] - self.y)
        direction = 1 if e_along >= 0 else -1
        if self._creep_dir and direction != self._creep_dir:
            self._creep_flips += 1                          # over and back: the goal is within the noise - stop after a few
        self._creep_dir = direction
        e = (self.th - self.goal_heading + math.pi) % (2 * math.pi) - math.pi     # + : pointing left of the goal heading
        kappa = -direction * 3.0 * e                                               # turn the nose toward the goal heading
        self.msg = f"centring in the slot ({abs(e_along) * 100:.0f} cm)"
        return max(-self.kappa_max, min(self.kappa_max, kappa)), direction * self.CREEP_V

    NEIGHBOUR_REACH_M = 1.0   # parking: neighbours within this of the bumpers are measured for the final position (_neighbours_here)

    def _neighbours_here(self, pts_vehicle):
        """Parallel parking, the car in the slot and about straight: the neighbours' end faces ahead and behind in the scan (vehicle frame).
        Returns (e_along, e_lat): the move that centres the car between them (+ = forward; from the free distances ahead of the front
        bumper and behind the rear one, low / high percentiles so the LiDAR's noise does not bias them) and the side move that puts the
        car's road side in line with their road-side corners (+ = left); None for either when it cannot be measured - then None
        altogether unless both neighbours are there. Never nearer the believed kerb than park_select.KERB_MIN_M."""
        req = self.park_request or {}
        if "strip" not in req:
            return None
        from .park_select import KERB_FAR_M, KERB_MIN_M, ParkingStrip
        P = np.asarray(pts_vehicle, float).reshape(-1, 2)
        f, r, w = self.p.front_x, self.p.rear_x, self.p.width
        R = self.NEIGHBOUR_REACH_M
        ahead = P[(P[:, 0] > f - 0.02) & (P[:, 0] < f + R) & (np.abs(P[:, 1]) < w / 2 + 0.15)]
        behind = P[(P[:, 0] < r + 0.02) & (P[:, 0] > r - R) & (np.abs(P[:, 1]) < w / 2 + 0.15)]
        core_a = ahead[np.abs(ahead[:, 1]) < w / 2 - 0.02]
        core_b = behind[np.abs(behind[:, 1]) < w / 2 - 0.02]
        # along: each neighbour seen gives the move that makes its free distance the planned one (both: their mean - centred in a gap the
        # slot fills, the planned offset in a long gap); none: no measurement
        d_rear_p, d_front_p = self._gaps_planned
        moves = []
        if len(core_a) >= 4 and d_front_p is not None:
            moves.append(float(np.percentile(core_a[:, 0], 10)) - f - d_front_p)
        if len(core_b) >= 4 and d_rear_p is not None:
            moves.append(d_rear_p - (r - float(np.percentile(core_b[:, 0], 90))))
        if not moves:
            return None
        e_along = float(np.mean(moves))
        # which way the kerb is, in the vehicle frame (the strip is in the start frame)
        sp, side, depth = req["strip"]
        strip = ParkingStrip(sp, side, depth)
        s_car, y_car = strip.to_strip(np.array([[self.x, self.y]]))
        th = strip.point_at(float(s_car[0]))[2]
        nk = np.array([-side * math.sin(th), side * math.cos(th)])              # toward the kerb, start frame
        ks = 1.0 if (-math.sin(self.th) * nk[0] + math.cos(self.th) * nk[1]) > 0 else -1.0
        corners = []
        for grp, near in ((ahead, lambda q: q[:, 0] < q[:, 0].min() + 0.04), (behind, lambda q: q[:, 0] > q[:, 0].max() - 0.04)):
            grp = grp[-ks * grp[:, 1] > -w / 2]              # (not beyond the car's kerb side: a kerb the LiDAR sees is no neighbour)
            if len(grp) < 4:
                continue
            face = grp[near(grp)]
            if len(face) >= 4:
                corners.append(float(np.percentile(-ks * face[:, 1], 90)))            # the end face's road-side corner (toward the road +)
        e_lat = None
        if corners and (len(corners) == 1 or abs(corners[0] - corners[1]) < 0.05):
            out = float(np.mean(corners)) - w / 2                                     # how much further toward the road they stand than we do
            # the believed kerb: the body's kerb side never nearer than KERB_MIN_M (a neighbour standing on the kerb is not followed)
            # between KERB_MIN_M and KERB_FAR_M off the believed kerb (standing nearer / further already: back inside that band)
            gap_now = depth - (float(y_car[0]) + w / 2)
            e_lat = ks * float(np.clip(-out, gap_now - KERB_FAR_M, gap_now - KERB_MIN_M))
        return e_along, e_lat

    LAT_TOL_M, LAT_MAX_M, SHIFT_MAX = 0.015, 0.12, 4     # parking: a side offset beyond this is shifted out (at most four passes)
    SHIFT_CLEAR_M = 0.04      # (outside the brake gate's crawl envelope, 3.2 cm, or it holds the correction)

    def _lateral_shift(self, e_lat, pts_vehicle):
        """A side shift of e_lat (m, + = left of the goal heading) at the goal: two opposite arcs (lateral move 2 (1 - cos kh) / k) and a
        straight move back along the goal heading to the goal - as a driver corrects a car standing too far from the kerb. Moving toward
        the kerb it reverses first (the short rear overhang then swings toward the kerb, the long front away from it); the arcs are as long
        as the room allows. Checked with the exact rectangle against the scan (SHIFT_CLEAR_M) and against the kerb line of the parking
        strip (park_select.KERB_CLEAR_M). Returns the path (N, 4) in the start frame, or None."""
        from .park_select import KERB_CLEAR_M, ParkingStrip, _arc, _obb_clearance
        gx, gy = self.goal
        gh = self.goal_heading
        c, s = math.cos(gh), math.sin(gh)
        pts = self.to_start(np.asarray(pts_vehicle, float).reshape(-1, 2))
        pts = pts[np.hypot(pts[:, 0] - self.x, pts[:, 1] - self.y) < 1.2]
        strip = None
        req = self.park_request or {}
        if "strip" in req:
            sp, side, depth = req["strip"]
            strip = ParkingStrip(sp, side, depth)
            kerb_y = lambda P: strip.to_strip(P)[1]
        toward_kerb = strip is not None and (strip.to_strip(np.array([[self.x - s * e_lat, self.y + c * e_lat]]))[1][0]
                                             > strip.to_strip(np.array([[self.x, self.y]]))[1][0])
        e = abs(e_lat)
        kmax = 0.8 * self.kappa_max

        def k_for(shift, h):
            """Curvature whose two opposite arcs of length h move the car `shift` sideways: 2 (1 - cos k h) / k = shift (bisection)."""
            lo, hi = 0.0, kmax
            if 2.0 * (1.0 - math.cos(hi * h)) / hi < shift:
                return None
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if 2.0 * (1.0 - math.cos(mid * h)) / mid < shift else (lo, mid)
            return math.copysign(hi, e_lat)

        def s_move(pose, k, d, h):
            a1 = _arc(pose, k, d * h)
            a2 = _arc(tuple(a1[-1]), -k, d * h)
            return np.vstack([a1, a2])

        def kerb_most(P):
            cs = [P[:, :2] + np.column_stack([np.cos(P[:, 2]) * lx - np.sin(P[:, 2]) * ly, np.sin(P[:, 2]) * lx + np.cos(P[:, 2]) * ly])
                  for lx in (self.p.front_x, self.p.rear_x) for ly in (-self.p.width / 2, self.p.width / 2)]
            return max(float(kerb_y(q).max()) for q in cs)

        start = (self.x, self.y, self.th)
        here = np.array([start])
        # the car may already be closer than the margins (that is what is being corrected): never closer than it is now
        clear_min = self.SHIFT_CLEAR_M if not len(pts) else min(self.SHIFT_CLEAR_M, float(_obb_clearance(here, pts, self.p)[0]) - 0.005)
        # (a car already on the kerb line may swing a corner up to 1 cm further while it moves off it: the rear corner swings toward the
        # kerb when the nose turns away from it)
        kerb_max = None if strip is None else max(strip.depth - KERB_CLEAR_M, kerb_most(here) + 0.003)

        def ok(P):
            if len(pts) and (_obb_clearance(P, pts, self.p) < clear_min).any():
                return False
            return strip is None or kerb_most(P) <= kerb_max
        order = (-1, 1) if toward_kerb else (1, -1)
        for partial in (False, True):
            # the whole offset first; when no move of the room there is takes it all out (a tight gap), the largest part that fits - the
            # next pass at the end of the path takes the rest (SHIFT_MAX passes)
            for h in (0.25, 0.20, 0.15, 0.12, 0.09):
                smax = 2.0 * (1.0 - math.cos(kmax * h)) / kmax - 1e-4
                # one S and a straight move back along the goal heading; or - less room along the slot - two S moves, forward and back,
                # each taking half the offset (the classic shuffle: the backward S with the same curvature sign moves the same way)
                for d in order:
                    k = k_for(min(e, smax) if partial else e, h)
                    if k is not None:
                        S = s_move(start, k, d, h)
                        back = c * (gx - S[-1, 0]) + s * (gy - S[-1, 1])
                        B = _arc(tuple(S[-1]), 0.0, back) if abs(back) > 0.01 else np.empty((0, 3))
                        P = np.vstack([q for q in (S, B) if len(q)])
                        if ok(P):
                            return np.column_stack([P, np.concatenate([np.full(len(S), d), np.full(len(B), 1 if back > 0 else -1)])])
                for d in order:
                    k = k_for(min(e / 2, smax) if partial else e / 2, h)
                    if k is not None:
                        S1 = s_move(start, k, d, h)
                        S2 = s_move(tuple(S1[-1]), k, -d, h)
                        P = np.vstack([S1, S2])
                        if ok(P):
                            return np.column_stack([P, np.concatenate([np.full(len(S1), d), np.full(len(S2), -d)])])
        return None

    STROKE_M, STROKE_V = 0.07, 0.10

    def _in_lane(self):
        """Parallel parking: the body's centre is inside the parking lane (past the road edge)."""
        strip = self._kerb_strip()
        if strip is None:
            return False
        c = 0.5 * (self.p.front_x + self.p.rear_x)
        _, y = strip.to_strip(np.array([[self.x + c * math.cos(self.th), self.y + c * math.sin(self.th)]]))
        return bool(y[0] > 0.0)

    def _retry_by_pull_out(self, pts_vehicle):
        """Plan the way out of the lane from here (plan service); _collect then follows it and approaches the slot again."""
        import time
        from .plan_service import plan_pull_out_job
        sp, side, depth = self.park_request["strip"]
        c, s = math.cos(self.th), math.sin(self.th)
        P = np.asarray(sp, float).reshape(-1, 2)
        dx, dy = P[:, 0] - self.x, P[:, 1] - self.y
        strip_v = np.column_stack([c * dx + s * dy, -s * dx + c * dy])
        pts = np.asarray(pts_vehicle, float).reshape(-1, 2)
        self._retry_pose, self._refine_pts = (self.x, self.y, self.th), self.to_start(pts)
        self._park, self.state, self._job_t = "retry_out", "planning", time.perf_counter()
        self.msg = "stuck - pulling out to try again"
        self._job = self.service.submit(plan_pull_out_job, self.p, strip_v, side, depth, pts)
        if self._job.ready(0.0):
            self._collect()

    def _kerb_strip(self):
        req = self.park_request or {}
        if "strip" not in req:
            return None
        from .park_select import ParkingStrip
        sp, side, depth = req["strip"]
        return ParkingStrip(sp, side, depth)

    def _kerb_most(self, P, strip):
        """The body's kerb-most point over poses P (start frame), as y from the road edge toward the kerb."""
        P = np.atleast_2d(P)
        cs = [P[:, :2] + np.column_stack([np.cos(P[:, 2]) * lx - np.sin(P[:, 2]) * ly, np.sin(P[:, 2]) * lx + np.cos(P[:, 2]) * ly])
              for lx in (self.p.front_x, self.p.rear_x) for ly in (-self.p.width / 2, self.p.width / 2)]
        return max(float(strip.to_strip(q)[1].max()) for q in cs)

    def _move_ok(self, P, pts_vehicle):
        """A short parking move (poses P, start frame) keeps park_select.KERB_CLEAR_M off the kerb line and 3 cm off anything scanned -
        or, standing closer already, never comes closer than it is (3 mm)."""
        from .park_select import KERB_CLEAR_M, _obb_clearance
        here = np.array([[self.x, self.y, self.th]])
        pts = self.to_start(np.asarray(pts_vehicle, float).reshape(-1, 2)) if pts_vehicle is not None and len(pts_vehicle) else np.empty((0, 2))
        pts = pts[np.hypot(pts[:, 0] - self.x, pts[:, 1] - self.y) < 1.0] if len(pts) else pts
        if len(pts):
            c0 = float(_obb_clearance(here, pts, self.p)[0])
            if (_obb_clearance(np.atleast_2d(P), pts, self.p) < min(0.04, c0 - 0.003)).any():
                return False
        strip = self._kerb_strip()
        if strip is not None and self._kerb_most(P, strip) > max(strip.depth - KERB_CLEAR_M, self._kerb_most(here, strip) + 0.003):
            return False
        return True

    def _straighten(self, head_err, pose, pts_vehicle=None):
        """Parking, at the spot but not straight: short forward and reverse strokes with opposite steering (each pair turns the car a
        few degrees and brings it back), as a driver does the last correction in a tight bay. Each stroke is checked before it starts
        (_move_ok: full lock swings a corner toward the kerb - 30 Sep: 'during the parking it does slightly clip the kerb'); the other
        direction is taken when the first is not clear. Returns (curvature, signed speed), or None when no stroke is clear."""
        from .park_select import _arc
        e = (self.th - self.goal_heading + math.pi) % (2 * math.pi) - math.pi          # + : the car points left of the goal heading
        k = self.kappa_max
        kap = lambda d: -k * math.copysign(1.0, e) if d > 0 else k * math.copysign(1.0, e)
        if self._stroke_from is None or math.hypot(self.x - self._stroke_from[0], self.y - self._stroke_from[1]) >= self.STROKE_M:
            first = -self._stroke_dir if self._stroke_dir else 1
            for d in (first, -first):
                if pts_vehicle is None or self._move_ok(_arc((self.x, self.y, self.th), kap(d), d * self.STROKE_M), pts_vehicle):
                    break
            else:
                return None
            self._stroke_dir, self._stroke_from = d, (self.x, self.y)
            self.fine_strokes += 1
        self.msg = f"straightening ({math.degrees(abs(e)):.0f} deg off)"
        return kap(self._stroke_dir), self._stroke_dir * self.STROKE_V

    def leg(self):
        """The leg of the path being driven (up to the next change of direction) in the vehicle frame, for the
        brake gate: ((N, 3) poses, direction), or None."""
        if self.state != "driving" or self.path is None:
            return None
        from .assists import _leg_vehicle
        return _leg_vehicle(self.path, self.i, (self.x, self.y, self.th))

    def preview(self):
        """The rest of the planned path in the current vehicle frame, for the GUI: while parallel parking also the manoeuvre that follows
        the way to the slot; kept on the map while a re-plan / the gap measurement runs (the car holds still then)."""
        if self.state == "idle":
            return None
        parts = []
        if self.path is not None and self.i < len(self.path):
            parts.append(np.asarray(self.path, float)[self.i:, :2])
        if self.park_phase in ("pull_out", "approach") and self._tail is not None:
            parts.append(self._tail[:, :2])
        if not parts:
            return None
        return self.to_vehicle(np.vstack(parts))
