"""Scripted virtual drivers.

HumanLikeDriver behaves like a person steering with a controller: it wants to
reach goal points, steers around what it sees, slows for close obstacles, reacts
after a delay, has some hand noise, and now and then looks away (an attention
lapse) and keeps doing what it was doing. Lapses are what create genuine
hazards for the warning system to catch.
"""

import math
from collections import deque

import numpy as np

from adas.geometry import path_pose, travel_distance_to_contact
from adas.lidar_utils import scan_to_points
from adas.vehicle_params import steer_to_delta

from .lidar_sim import LidarSim


class HumanLikeDriver:
    def __init__(self, seed=0, cruise_pwm=None, reaction=0.25, lapse_rate=0.12,
                 arena=(-0.5, 5.5, -1.6, 1.6), lapses=True, margin=0.07):
        self.rng = np.random.default_rng(seed)
        self.cruise = cruise_pwm if cruise_pwm is not None else int(self.rng.integers(120, 230))
        self.reaction = reaction
        self.lapse_rate = lapse_rate if lapses else 0.0
        self.arena = arena
        self.margin = margin
        self.eye = LidarSim(n_points=480, noise_std=0.0, dropout=0.0, min_range=0.02, max_range=4.0)

        self.goal = None
        self.steer = 0.0
        self.pwm = 0.0
        self.noise = 0.0
        self.hist = deque()                 # (t, points) delayed by the reaction time
        self.next_look = 0.0
        self.lapse_until = -1.0
        self.in_lapse = False
        self.stuck_since = None
        self.reverse_until = -1.0
        self._next_plan = 0.0
        self._plan = (0.0, 1.6)

    def _new_goal(self, sim):
        x0, x1, y0, y1 = self.arena
        for _ in range(20):
            g = (self.rng.uniform(x0 + 0.4, x1 - 0.4), self.rng.uniform(y0 + 0.4, y1 - 0.4))
            if math.hypot(g[0] - sim.car.x, g[1] - sim.car.y) > 1.2:
                break
        self.goal = g

    def _delayed_points(self, t):
        while len(self.hist) > 1 and self.hist[1][0] <= t - self.reaction:
            self.hist.popleft()
        return self.hist[0][1] if self.hist else np.empty((0, 2))

    def __call__(self, t, sim):
        c, p = sim.car, sim.p

        if t >= self.next_look:
            self.next_look = t + 0.1
            a, r, v = self.eye.scan(sim.world, c.x, c.y, c.theta, p)
            self.hist.append((t, scan_to_points(a, r, v, p)))

        if self.goal is None or math.hypot(self.goal[0] - c.x, self.goal[1] - c.y) < 0.35:
            self._new_goal(sim)

        # attention lapses
        if not self.in_lapse and self.rng.random() < self.lapse_rate * 0.02:
            self.in_lapse = True
            self.lapse_until = t + self.rng.uniform(0.6, 1.4)
        if self.in_lapse and t >= self.lapse_until:
            self.in_lapse = False

        self.noise += (-self.noise * 2.0 * 0.02 + self.rng.normal(0, 0.05) * math.sqrt(0.02))

        if t < self.reverse_until:
            return 0.0, -140.0

        if self.in_lapse:
            return float(np.clip(self.steer + self.noise, -1, 1)), self.pwm

        if t >= self._next_plan:
            self._next_plan = t + 0.1
            pts = self._delayed_points(t)
            goal_bearing = math.atan2(self.goal[1] - c.y, self.goal[0] - c.x) - c.theta
            goal_bearing = math.atan2(math.sin(goal_bearing), math.cos(goal_bearing))

            best, best_cost, best_D = 0.0, 1e9, 0.0
            for s in np.linspace(-1, 1, 21):
                delta = steer_to_delta(s, p)
                D = travel_distance_to_contact(pts, delta, 1, p, horizon=1.6, margin=self.margin)
                Dc = min(D, 1.6) if math.isfinite(D) else 1.6
                _, _, psi = path_pose(delta, 0.5, p)
                heading_err = abs(math.atan2(math.sin(goal_bearing - float(psi)),
                                             math.cos(goal_bearing - float(psi))))
                cost = 1.6 * (1.0 - Dc / 1.6) + 0.9 * heading_err / math.pi + 0.15 * abs(s - self.steer)
                if cost < best_cost:
                    best, best_cost, best_D = float(s), cost, Dc
            self._plan = (best, best_D)
        best, best_D = self._plan

        self.steer += float(np.clip(best - self.steer, -3.0 * 0.02, 3.0 * 0.02))
        speed_factor = float(np.clip(best_D / 1.1, 0.35, 1.0))
        target = self.cruise * speed_factor
        if best_D < 0.30:
            target = -110.0
        self.pwm += float(np.clip(target - self.pwm, -900 * 0.02, 500 * 0.02))

        if abs(c.v) < 0.05 and self.pwm > 60:
            self.stuck_since = self.stuck_since if self.stuck_since is not None else t
            if t - self.stuck_since > 1.5:
                self.reverse_until = t + 0.9
                self.stuck_since = None
                self.goal = None
        else:
            self.stuck_since = None

        return float(np.clip(self.steer + self.noise, -1, 1)), self.pwm
