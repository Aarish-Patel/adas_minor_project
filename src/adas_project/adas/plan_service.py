"""Path planning off the control loop (TODO K2).

A Hybrid A* search takes 5-50 ms on a laptop and occasionally seconds in a cluttered scene; on the Pi 5 that is
~3.5x longer. Run inside the relay's 20 Hz control loop it would freeze the braking and steering, and a Python
thread would not help either - it holds the interpreter lock while it computes. So, as a planner is a separate
node in ROS, the relay runs the searches in ONE worker process: the control loop only submits a job and polls it
each tick; the brake gate keeps protecting the car while a plan is being made.

Modes:
  process   a worker process (the relay on the car / in the laptop simulator)
  inline    computed at once in the caller (simulations, tests) - but the result is only released after the time
            the Pi would have needed, so simulations see the delay: with a latency factor, the work done (search nodes,
            grid cells) at the Pi's measured rates, and search budgets in nodes - deterministic, whatever the laptop is doing
"""
import time


# Computing-time budgets per search, in seconds ON THE PI (simulations scale them by the Pi's slowness): a swerve
# that cannot be found this quickly is not wanted - the brake gate handles that case.
EVASIVE_BUDGET_S = 0.20           # forward swerve search: the car is moving, it must be quick (typically ~0.05 s)
BACKOFF_BUDGET_S = 2.0            # with reversing: only needed close to an obstacle, where the brake holds the car
GOTO_BUDGET_S = 2.0               # click-to-go: the car waits while it plans
# The twin's model of the Pi's planning time (plan delay and node budgets). Least-squares fit of measured laptop time x PI_COMPUTE_FACTOR
# (3.5) on 127 jobs of every kind (evasive, click-to-go, parking), single-threaded: per-kind median ratio measured / model 0.8-1.0.
PI_NODES_PER_S = 1400.0           # Hybrid A* expansions (with their collision checks and analytic tries) per second on the Pi 5
PI_GRID_CELLS_PER_S = 8.0e5       # planning-grid cells (occupancy + distance transform) per second
PI_POSE_POINTS_PER_S = 1.3e7      # exact body-clearance evaluations (pose x point) per second
PI_JOB_BASE_S = 0.005             # a job's fixed cost (hand-over, set-up)


# ------------------------------------------------------------------ the jobs (module level: they run in the worker)
def plan_line_job(params, kappa_max, pts, start, x_goal, max_nodes, reverse_nodes, budget_s=None, reverse_budget_s=None):
    """Evasive manoeuvre: back to the driver's line y = 0 beyond x_goal; forward first, then with reversing."""
    from .hybrid_astar import HybridAStar
    ha = HybridAStar(params, kappa_max=kappa_max)
    path = ha.plan(pts, start, x_goal=x_goal, allow_reverse=False, max_nodes=max_nodes, budget_s=budget_s)
    if path is None and ha.gave_up != "goal unreachable":
        path = ha.plan(pts, start, x_goal=x_goal, allow_reverse=True, max_nodes=reverse_nodes,
                       budget_s=reverse_budget_s)
    return path


GOTO_W_REVERSE = 6.0              # click-to-go: a metre driven backwards costs as much as 7 m forwards


COMFORT_MARGIN_M = 0.10       # click-to-go plans keep this clearance when they can (brake field ~3 cm + tracking error); the planner's
                              # own 6 cm margin only where no path with this much room exists (clearance hierarchy, AK2)


def plan_point_job(params, kappa_max, pts, start, goal, budget_s=None, goal_heading=None, reverse_first=False):
    # (budget_s applies to each of the two searches)
    """Click-to-go to a goal position (and heading, if given): forward-only first - also for goals behind the car
    (a forward U-turn when there is room) - and only if no forward path exists, a search that may reverse, where
    reversing is expensive, so it backs up only as much as the goal heading or the room needs.
    Clearance hierarchy (research, docs/research_lidar_navigation.md section 3): a first search with COMFORT_MARGIN_M, so the path keeps
    the brake gate's protective field plus the follower's tracking error away from obstacles and the gate does not brake on it; only if
    that finds nothing, the planner's normal margin (narrow doorways, a car parked close)."""
    from .hybrid_astar import HybridAStar
    if not reverse_first:
        ha = HybridAStar(params, kappa_max=kappa_max, margin=COMFORT_MARGIN_M)
        path = ha.plan_to_point(pts, start, goal, max_nodes=2500, budget_s=budget_s, goal_heading=goal_heading,
                                coarse=goal_heading is not None)
        if path is not None:
            return path
    ha = HybridAStar(params, kappa_max=kappa_max)
    if reverse_first:                      # parking: the way in is a reverse S-curve; the forward-only stage would only burn the budget
        return ha.plan_to_point(pts, start, goal, allow_reverse=True, max_nodes=12000,
                                budget_s=None if budget_s is None else 4.0 * budget_s,      # parking plans while standing: more time is fine
                                goal_heading=goal_heading, w_reverse=GOTO_W_REVERSE)
    # a goal pose needs more search than a point: the forward stage then uses the coarse lattice too (the fine one
    # ran out of nodes on a doorway-then-turn goal that the coarse one solves in a fraction of the time)
    path = ha.plan_to_point(pts, start, goal, max_nodes=2500, budget_s=budget_s, goal_heading=goal_heading,
                            coarse=goal_heading is not None)
    if path is not None or ha.gave_up == "goal unreachable":
        return path
    return ha.plan_to_point(pts, start, goal, allow_reverse=True, max_nodes=6000, budget_s=budget_s,
                            goal_heading=goal_heading, w_reverse=GOTO_W_REVERSE)


def plan_parallel_job(params, strip_pts, side, depth, click, pts):
    """Parking: find the free gaps on the strip from the scan, take the viable one nearest to the click and plan the reverse manoeuvre
    from its pull-up pose (adas/park_select.py); the car drives to that pose with the click-to-go planner (AutoNav), after pulling out of
    the gap it stands in if it is parked. Everything in the vehicle frame now. Returns a ParkingPlan."""
    from .park_select import ParkingStrip, plan_parallel, plan_pull_out
    strip = ParkingStrip(strip_pts, side, depth)
    plan = plan_parallel(strip, click, params, pose=(0.0, 0.0, 0.0), points=pts, approach=False)
    if plan.planned_path is not None:
        out = plan_pull_out(strip, params, pts)
        if out is not None:
            plan = plan._replace(pull_out=out)
    return plan


def plan_pull_out_job(params, strip_pts, side, depth, pts):
    """A go-to from a car parked in the lane: the way out onto the road first (adas/park_select.plan_pull_out, vehicle frame now).
    Returns poses (N, 4) or None."""
    from .park_select import ParkingStrip, plan_pull_out
    return plan_pull_out(ParkingStrip(strip_pts, side, depth), params, pts)


def plan_refine_job(params, strip_pts, side, depth, click, slot, pts):
    """Parking, alongside the chosen slot: the gap measured again in the close-range scan and the manoeuvre re-planned (vehicle frame
    now). Returns a ParkingPlan."""
    from .park_select import ParkingStrip, refine_parallel
    return refine_parallel(ParkingStrip(strip_pts, side, depth), click, slot, params, points=pts)


def plan_perpendicular_job(params, marker, pts):
    """Parking: the bay behind the rear-wall ArUco marker (MarkerObs in the vehicle frame). Returns a ParkingPlan."""
    from .park_select import plan_perpendicular
    return plan_perpendicular(marker, params, pose=(0.0, 0.0, 0.0), points=pts)


# ------------------------------------------------------------------ the service
class Job:
    def __init__(self, future=None, result=None, delay=0.0):
        self.future, self._result, self.delay = future, result, delay

    def ready(self, dt=0.0):
        """Advance by dt (inline mode's simulated compute time) and say whether the result can be taken."""
        if self.future is not None:
            return self.future.done()
        self.delay -= dt
        return self.delay <= 0.0

    def result(self):
        if self.future is not None:
            try:
                return self.future.result()
            except Exception:                  # a failed search must never take the relay down - but it is reported, not hidden
                import traceback
                traceback.print_exc()
                return None
        return self._result


class PlanService:
    def __init__(self, mode="inline", latency_factor=0.0):
        self.mode, self.latency_factor = mode, latency_factor
        self._pool = None
        self.last_ms = None

    def start(self):
        """Start the worker now (the relay does this at launch) so the first swerve does not pay for it."""
        if self.mode == "process":
            self._executor()
        return self

    def _executor(self):
        if self._pool is None:
            from concurrent.futures import ProcessPoolExecutor
            self._pool = ProcessPoolExecutor(max_workers=1)
            self._pool.submit(int, 0).result()          # wait until the worker process is up
        return self._pool

    def budget(self, pi_seconds):
        """A computing budget given in Pi seconds, for the machine the search will run on: on the Pi (process
        mode) as is; inline with a latency factor (a simulation of the Pi) as the number of nodes the Pi expands in that time
        (hybrid_astar.NodeBudget) - deterministic, whatever else the laptop is doing."""
        if self.mode == "inline" and self.latency_factor > 0:
            from .hybrid_astar import NodeBudget
            return NodeBudget(pi_seconds * PI_NODES_PER_S)
        return pi_seconds

    def submit(self, fn, *args):
        if self.mode == "process":
            return Job(future=self._executor().submit(fn, *args))
        from .hybrid_astar import WORK
        n0, c0, p0 = WORK["nodes"], WORK["grid_cells"], WORK["pose_points"]
        t0 = time.perf_counter()
        res = fn(*args)
        elapsed = time.perf_counter() - t0
        if self.latency_factor > 0:
            # a simulation of the Pi: the delay from the WORK done (search nodes, grid cells) at the Pi's rates, not from how long the
            # laptop happened to take - the same drive then gives the same result every time
            delay = PI_JOB_BASE_S + (WORK["nodes"] - n0) / PI_NODES_PER_S + (WORK["grid_cells"] - c0) / PI_GRID_CELLS_PER_S                 + (WORK["pose_points"] - p0) / PI_POSE_POINTS_PER_S
        else:
            delay = 0.0
        self.last_ms = (delay if self.latency_factor > 0 else elapsed) * 1000
        return Job(result=res, delay=delay)

    def shutdown(self, wait=True):
        """Stop the worker process (waits for it by default, so it can never outlive the relay)."""
        if self._pool is not None:
            self._pool.shutdown(wait=wait, cancel_futures=True)
            self._pool = None
