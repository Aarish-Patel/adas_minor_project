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

    @property
    def active(self):
        return self.state != "idle"

    # ------------------------------------------------------------------ planning
    def _launch(self, pts_start, start):
        import time
        self.state = "planning"
        self._job_t = time.perf_counter()
        self._job = self.service.submit(plan_point_job, self.p, self.kappa_max, pts_start, start, self.goal,
                                        self.service.budget(GOTO_BUDGET_S), self.goal_heading)
        if self._job.ready(0.0):
            self._collect()

    def _collect(self):
        import time
        job, self._job = self._job, None
        if job is None:
            return
        path = job.result()
        self.plan_ms = self.service.last_ms if self.service.mode == "inline" else (time.perf_counter() - self._job_t) * 1000
        if path is None:
            self.state = "idle"
            self.msg = "no safe path to that point" if self.path is None else "path blocked - stopped"
            return
        self.path, self.i, self.replan_t = path, 0, 0.0
        self.state = "driving"
        self.msg = "driving to the goal" if self.replans == 0 else "re-planned around a new obstacle"

    def start(self, goal, pts_vehicle, heading=None):
        """goal: (x, y) m in the vehicle frame now; heading: the direction to arrive in (rad, 0 = the car's heading
        now, + = left), or None for any. Starts planning; returns False if the request is refused."""
        self.x = self.y = self.th = 0.0
        self.goal = (float(goal[0]), float(goal[1]))
        self.goal_heading = None if heading is None else float(heading)
        self.path, self.replans = None, 0
        self.stall_s, self.stalls = 0.0, 0
        self.msg = "planning a path"
        self._launch(np.asarray(pts_vehicle, float).reshape(-1, 2), (0.0, 0.0, 0.0))
        return self.state != "idle"

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
        if self.state == "planning":
            return 0.0, 0.0
        gx, gy = self.goal
        dist_goal = math.hypot(gx - self.x, gy - self.y)
        head_err = 0.0 if self.goal_heading is None else             abs((self.th - self.goal_heading + math.pi) % (2 * math.pi) - math.pi)
        if dist_goal < self.arrive and head_err < self.heading_tol:
            self.cancel("arrived")
            return None
        if self.goal_heading is not None and self.i >= len(self.path) - 2 and dist_goal < 2 * self.arrive:
            self.cancel(f"arrived ({math.degrees(head_err):.0f} deg off the chosen heading)")
            return None
        P = self.path
        self.replan_t += dt
        if self.replan_t > 0.5 and len(pts_vehicle):      # new scans: keep the plan while the rest of it is clear
            self.replan_t = 0.0
            pts_s = self.to_start(pts_vehicle)
            ha = HybridAStar(self.p, kappa_max=self.kappa_max)
            g = Grid(pts_s, min(self.x, gx) - 1.5, max(self.x, gx) + 1.5, min(self.y, gy) - 1.5, max(self.y, gy) + 1.5)
            if len(P) - self.i > 2 and ha.clearance(g, P[self.i::2, :3]) < 0.01:
                self.replans += 1
                self.msg = "path blocked - re-planning"
                self._launch(pts_s, (self.x, self.y, self.th))
                if self.state != "driving":
                    return (0.0, 0.0) if self.state == "planning" else None
                P = self.path
        # stopped on the path although the operator holds the throttle (the brake gate holds the car): search
        # again from exactly here - that path starts at the car - and give up after a few (user, 28 Sep)
        self.stall_s = self.stall_s + dt if held and abs(v) < 0.03 else 0.0
        if self.stall_s > 0.8 and len(pts_vehicle):
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
        d = np.hypot(P[self.i:self.i + 60, 0] - self.x, P[self.i:self.i + 60, 1] - self.y)
        self.i += int(np.argmin(d)) if len(d) else 0
        self.i = past_cusp(P, self.i, (self.x, self.y, self.th))
        direction = int(P[self.i, 3])
        k = self.i
        while k + 1 < len(P) and P[k + 1, 3] == direction and \
                math.hypot(P[k, 0] - P[self.i, 0], P[k, 1] - P[self.i, 1]) < self.look:
            k += 1
        # the last stretch: aim at the goal point itself - unless a heading was chosen, then follow the planned
        # path to its end (the Dubins approach arrives pointing the right way)
        tgt = P[k:k + 1, :2] if k < len(P) - 1 or direction < 0 or self.goal_heading is not None             else np.array([[gx, gy]])
        xl, yl = self.to_vehicle(tgt)[0]
        dd = xl * xl + yl * yl
        kappa = max(-self.kappa_max, min(self.kappa_max, 2.0 * yl / dd if dd > 1e-4 else 0.0))
        # slow for tight curves (lateral acceleration ~0.12 g) and approaching the goal / a change of direction
        to_stop = dist_goal if k == len(P) - 1 else math.hypot(P[k, 0] - self.x, P[k, 1] - self.y)
        speed = min(self.cruise, math.sqrt(1.2 / max(abs(kappa), 1e-3)), 0.12 + 0.6 * to_stop)
        return kappa, direction * speed

    def leg(self):
        """The leg of the path being driven (up to the next change of direction) in the vehicle frame, for the
        brake gate: ((N, 3) poses, direction), or None."""
        if self.state != "driving" or self.path is None:
            return None
        from .assists import _leg_vehicle
        return _leg_vehicle(self.path, self.i, (self.x, self.y, self.th))

    def preview(self):
        """The rest of the planned path in the current vehicle frame, for the GUI."""
        if self.state != "driving" or self.path is None:
            return None
        return self.to_vehicle(self.path[self.i:, :2])
