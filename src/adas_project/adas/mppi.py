"""MPPI - Model Predictive Path Integral control (Williams, Aldrich, Theodorou, ICRA 2016; Williams et al., "Information
theoretic MPC for model-based reinforcement learning", ICRA 2017 / T-RO 2018). The evasive steer's local fallback when
Hybrid A* finds no way round (or while it searches again): sample N steering-curvature sequences around the current
plan, roll each out with the car's kinematic model, score it (collision, low clearance, not getting back to the
driver's line, jerky steering), and take the exponentially weighted average - every control tick, warm-started from
the last one. Nav2 ships the same idea as its MPPI controller. Vectorised numpy; ~1-2 ms per tick on a laptop.

Frame: the evasive manoeuvre's "line frame" - x along the driver's desired path, y left.
"""
import math

import numpy as np


class MPPI:
    def __init__(self, params, kappa_max=1.5, n=200, steps=25, dt=0.1, sigma=0.6, lam=0.4, margin=0.04, seed=0):
        self.p = params
        self.kmax, self.n, self.T, self.dt = kappa_max, n, steps, dt
        self.sigma, self.lam, self.margin = sigma, lam, margin
        self.U = np.zeros(steps)                      # the mean curvature sequence (warm start)
        self.rng = np.random.default_rng(seed)
        L = params.front_x - params.rear_x
        seg = L / 3.0
        self.circle_x = np.array([params.rear_x + seg / 2, params.rear_x + 1.5 * seg, params.rear_x + 2.5 * seg])
        self.circle_r = math.hypot(seg / 2, params.width / 2)
        self.best = None                              # the best rollout of the last step (for the GUI)

    def reset(self):
        self.U[:] = 0.0

    def step(self, grid, pose, v, x_goal, w_line=4.0, w_head=0.8, w_progress=1.0, w_clear=6.0, w_smooth=0.05):
        """One control step. grid: adas.hybrid_astar.Grid of the obstacles (distance transform), pose (x, y, th) in
        the line frame, v the speed the rollouts drive at (m/s). Returns (curvature to steer now, ok): ok is False
        when even the best sampled rollout touches something - then only the brake can help."""
        x0, y0, th0 = pose
        v = max(0.15, abs(v))
        K = np.clip(self.U[None, :] + self.rng.normal(0.0, self.sigma, (self.n, self.T)), -self.kmax, self.kmax)
        K[0] = self.U                                  # always score the current plan itself
        ds = v * self.dt
        th = th0 + np.cumsum(K * ds, axis=1)
        thm = th - K * ds / 2                          # mid-step heading
        x = x0 + np.cumsum(ds * np.cos(thm), axis=1)
        y = y0 + np.cumsum(ds * np.sin(thm), axis=1)
        cx = x[..., None] + np.cos(th)[..., None] * self.circle_x
        cy = y[..., None] + np.sin(th)[..., None] * self.circle_x
        clear = (grid.lookup(cx, cy) - self.circle_r).min(axis=2)       # (n, T)
        hit = clear < self.margin
        # once a rollout touches something, the rest of it does not count as progress
        alive = np.cumprod(~hit, axis=1).astype(bool)
        cost = 1e4 * hit.any(axis=1) \
            + w_clear * np.maximum(0.0, 0.15 - clear).sum(axis=1) * self.dt \
            + w_line * np.abs(y[:, -1]) + w_head * np.abs(np.arctan2(np.sin(th[:, -1]), np.cos(th[:, -1]))) \
            - w_progress * np.where(alive[:, -1], np.minimum(x[:, -1], x_goal + 1.0) - x0, 0.0) \
            + w_smooth * (np.diff(K, axis=1) ** 2).sum(axis=1)
        c0 = cost.min()
        i = int(np.argmin(cost))
        w = np.exp(-(cost - c0) / self.lam)
        # commit to one side: average only the samples that pass on the same side as the best one (plain MPPI
        # averages "round the left" and "round the right" into "straight on" when both are equally good)
        side = np.sign(y[:, -1] - y0)
        same = side == side[i]
        if same.any():
            w = w * same
        w /= w.sum()
        self.U = (w[:, None] * K).sum(axis=0)
        self.best = np.column_stack([x[i], y[i], th[i]])
        kappa = float(np.clip(self.U[0], -self.kmax, self.kmax))
        self.U = np.concatenate([self.U[1:], self.U[-1:]])   # shift for the next tick (warm start)
        return kappa, bool(c0 < 1e4)
