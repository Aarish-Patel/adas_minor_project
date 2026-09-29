"""Path planning off the control loop (TODO K2).

A Hybrid A* search takes 5-50 ms on a laptop and occasionally seconds in a cluttered scene; on the Pi 5 that is
~3.5x longer. Run inside the relay's 20 Hz control loop it would freeze the braking and steering, and a Python
thread would not help either - it holds the interpreter lock while it computes. So, as a planner is a separate
node in ROS, the relay runs the searches in ONE worker process: the control loop only submits a job and polls it
each tick; the brake gate keeps protecting the car while a plan is being made.

Modes:
  process   a worker process (the relay on the car / in the laptop simulator)
  inline    computed at once in the caller (simulations, tests) - but the result is only released after the time
            the Pi would have needed (measured compute time x latency_factor), so simulations see the delay
"""
import time


# Computing-time budgets per search, in seconds ON THE PI (simulations scale them by the Pi's slowness): a swerve
# that cannot be found this quickly is not wanted - the brake gate handles that case.
EVASIVE_BUDGET_S = 0.20           # forward swerve search: the car is moving, it must be quick (typically ~0.05 s)
BACKOFF_BUDGET_S = 2.0            # with reversing: only needed close to an obstacle, where the brake holds the car
GOTO_BUDGET_S = 2.0               # click-to-go: the car waits while it plans


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


def plan_point_job(params, kappa_max, pts, start, goal, budget_s=None, goal_heading=None, reverse_first=False):
    # (budget_s applies to each of the two searches)
    """Click-to-go to a goal position (and heading, if given): forward-only first - also for goals behind the car
    (a forward U-turn when there is room) - and only if no forward path exists, a search that may reverse, where
    reversing is expensive, so it backs up only as much as the goal heading or the room needs."""
    from .hybrid_astar import HybridAStar
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
            except Exception:                  # a failed search must never take the relay down
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
        mode) as is; inline with a latency factor (a simulation of the Pi) shrunk by that factor."""
        if self.mode == "inline" and self.latency_factor > 0:
            return pi_seconds / self.latency_factor
        return pi_seconds

    def submit(self, fn, *args):
        if self.mode == "process":
            return Job(future=self._executor().submit(fn, *args))
        t0 = time.perf_counter()
        res = fn(*args)
        elapsed = time.perf_counter() - t0
        self.last_ms = elapsed * 1000
        return Job(result=res, delay=elapsed * self.latency_factor)

    def shutdown(self, wait=True):
        """Stop the worker process (waits for it by default, so it can never outlive the relay)."""
        if self._pool is not None:
            self._pool.shutdown(wait=wait, cancel_futures=True)
            self._pool = None
