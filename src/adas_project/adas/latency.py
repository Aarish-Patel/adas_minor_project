"""Online estimate of how late the LiDAR picture is (TODO N4; found by fault injection on 29 Sep: the brake gate is safe to
~0.1 s of extra scan latency at full throttle and touches the wall from ~0.15 s).

The relay has two independent views of the car's speed: the throttle model (advanced every command, so it is current) and the
LiDAR range flow (measured from scans, so it describes the car as it was when the scan was taken). If the scans arrive late,
the flow speed is a delayed copy of the model speed. The delay is found by the same time-alignment used to synchronise
sensors in state estimation (cross-correlation / least-squares time-delay estimation, Knapp & Carter 1976; online temporal
calibration of sensors, e.g. Kelly & Sukhatme 2011; Qin & Shen 2018): over a sliding window, the d that minimises

    sum_i ( v_flow(t_i) - v_model(t_i - d) )^2

Only trusted while the speed actually changed in the window (a constant speed carries no timing information), and the last
good estimate is held and decays slowly. `excess` = the estimate above the delay the gate already budgets for, capped -
the gate adds `speed * excess` to its stopping distance, i.e. it becomes more careful exactly when the picture is stale.
"""
import collections

import numpy as np


class DelayEstimator:
    def __init__(self, nominal_s=0.16, window_s=1.4, max_delay_s=0.5, step_s=0.02, min_swing=0.12, min_samples=8, cap_s=0.4):
        self.nominal, self.window, self.max_d, self.step = nominal_s, window_s, max_delay_s, step_s
        self.min_swing, self.min_samples, self.cap = min_swing, min_samples, cap_s
        self.model = collections.deque()             # (t, v_model)
        self.flow = collections.deque()              # (t, v_flow)
        self.delay = nominal_s                       # current estimate (s)
        self.last_good = None
        self.tested = 0

    def push_model(self, t, v):
        self.model.append((t, float(v)))
        while self.model and self.model[0][0] < t - self.window - self.max_d - 0.5:
            self.model.popleft()

    def push_flow(self, t, v):
        self.flow.append((t, float(v)))
        while self.flow and self.flow[0][0] < t - self.window:
            self.flow.popleft()
        if len(self.flow) >= self.min_samples and len(self.model) >= 2:
            self._estimate(t)

    def _estimate(self, now):
        mt = np.array([m[0] for m in self.model])
        mv = np.array([m[1] for m in self.model])
        ft = np.array([f[0] for f in self.flow])
        fv = np.array([f[1] for f in self.flow])
        if ft[-1] - ft[0] < 0.8 * self.window or fv.max() - fv.min() < self.min_swing:
            return                                    # too short, or no speed change in the window: no timing information
        ds = np.arange(0.0, self.max_d + 1e-9, self.step)
        cost = np.array([np.mean((fv - np.interp(ft - d, mt, mv)) ** 2) for d in ds])
        j = int(np.argmin(cost))
        # a clear minimum only: the best delay must beat the zero-delay fit and the worst fit by a margin
        if cost[j] > 0.5 * cost.max():
            return
        if 0 < j < len(ds) - 1:                       # sub-step refinement by a parabola through the minimum
            a, b, c = cost[j - 1], cost[j], cost[j + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if den > 1e-12 else 0.0
            d = ds[j] + max(-1.0, min(1.0, off)) * self.step
        else:
            d = ds[j]
        self.tested += 1
        self.delay = d if self.last_good is None else 0.6 * self.last_good + 0.4 * d
        self.last_good = self.delay

    @property
    def excess(self):
        """Delay above the budgeted one, 0..cap (s)."""
        return float(min(self.cap, max(0.0, self.delay - self.nominal)))
