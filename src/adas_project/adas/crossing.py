"""Predictive speed choice for MOVING obstacles (user, 28 Sep: "speed up to go past it, slow down to let it pass, and
move away if no other option").

Path-velocity decomposition (Kant & Zucker, IJRR 1986): the car keeps its path (the driver's steering arc, or the
planned leg in autonomy) and only its speed along the path is chosen. Each moving obstacle (adas/tracking.py: position
and its own velocity, the car's motion removed) is predicted at constant velocity; the times at which it occupies the
car's corridor, and the stretch of the path it occupies then, form forbidden regions in the space-time (s-t) plane.
Candidate speed profiles - accelerate or brake (within the car's limits) to a target speed and hold it - are checked
against those regions, and the one closest to what the driver / planner wants is taken:

  pass    faster than wanted: the car clears the crossing before the obstacle gets there
  yield   slower than wanted (down to a stop): the obstacle crosses first
  away    no profile along the path is safe, not even stopping (it is coming at the car): back off, if the way
          behind is free - else stop
  clear   nothing moving conflicts: the wanted speed
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CrossingConfig:
    horizon_s: float = 3.0         # s predicted ahead
    dt: float = 0.1
    margin: float = 0.10           # m kept between the body and a moving object (people / cars move unpredictably)
    time_margin: float = 0.4       # s: pass only if clear of the crossing this long before the object arrives
    accel: float = 0.8             # m/s^2 the car speeds up with
    decel: float = 2.5             # m/s^2 it slows down with (comfortable, well under the ~4 it can)
    v_max: float = 0.8
    pass_gain_max: float = 0.35    # m/s: never more than this above the wanted speed to get past
    path_len: float = 3.0


def arc_path(kappa, direction=1, length=3.0, ds=0.05):
    """Poses (N, 3) along a constant-curvature arc from the rear axle (vehicle frame)."""
    s = np.arange(0.0, length + ds, ds) * direction
    if abs(kappa) < 1e-6:
        return np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
    th = kappa * s
    return np.column_stack([np.sin(th) / kappa, (1 - np.cos(th)) / kappa, th])


def _profile(v0, v_target, cfg, times):
    """Distance travelled at each time: accelerate / brake at the limit to v_target, then hold it."""
    a = cfg.accel if v_target > v0 else -cfg.decel
    t_c = abs(v_target - v0) / max(abs(a), 1e-6)
    s = np.where(times <= t_c, v0 * times + 0.5 * a * times ** 2, v0 * t_c + 0.5 * a * t_c ** 2 + v_target * (times - t_c))
    return np.maximum(s, 0.0)


def forbidden(tracks, path, p, cfg):
    """For every predicted time step, the path stretch [s_lo, s_hi] the rear axle must stay out of (NaN = none).
    tracks: [(x, y, vx, vy, radius)] in the vehicle frame now; path: poses (N, 3) along the car's path."""
    times = np.arange(0.0, cfg.horizon_s + 1e-9, cfg.dt)
    s_path = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1])))])
    lo = np.full(len(times), np.nan)
    hi = np.full(len(times), np.nan)
    c, sn = np.cos(path[:, 2]), np.sin(path[:, 2])
    for (x, y, vx, vy, r) in tracks:
        qx, qy = x + vx * times, y + vy * times                     # the object's predicted centre
        dx = qx[:, None] - path[None, :, 0]
        dy = qy[:, None] - path[None, :, 1]
        lx = c[None, :] * dx + sn[None, :] * dy                      # the object in the car's frame at each s
        ly = -sn[None, :] * dx + c[None, :] * dy
        m = r + cfg.margin
        hit = (lx >= p.rear_x - m) & (lx <= p.front_x + m) & (np.abs(ly) <= p.width / 2 + m)
        for k in np.flatnonzero(hit.any(axis=1)):
            idx = np.flatnonzero(hit[k])
            a, b = s_path[idx[0]], s_path[idx[-1]]
            lo[k] = a if np.isnan(lo[k]) else min(lo[k], a)
            hi[k] = b if np.isnan(hi[k]) else max(hi[k], b)
    return times, lo, hi


def decide(tracks, path, v_now, v_want, p, cfg=None, rear_free=0.0):
    """-> (action, v_target, info). v_want: the speed the driver / planner asks for (m/s, >= 0 along the path);
    rear_free: free distance behind the car (m), for 'away'."""
    cfg = cfg or CrossingConfig()
    moving = [t for t in tracks if math.hypot(t[2], t[3]) > 0.05]
    if not moving or v_want <= 0.0:
        return "clear", v_want, {}
    times, lo, hi = forbidden(moving, path, p, cfg)
    busy = ~np.isnan(lo)
    if not busy.any():
        return "clear", v_want, {}
    t_first = float(times[busy][0])
    v_now = max(0.0, v_now)

    def safe(v_target, time_margin=0.0):
        s = _profile(v_now, v_target, cfg, times)
        # with a time margin: the car must also be clear of the stretch the object occupies a little earlier/later
        k = int(round(time_margin / cfg.dt))
        for i in np.flatnonzero(busy):
            j0, j1 = max(0, i - k), min(len(times) - 1, i + k)
            if np.any((s[j0:j1 + 1] >= lo[i]) & (s[j0:j1 + 1] <= hi[i])):
                return False
        return True
    info = {"conflict_in_s": round(t_first, 2), "conflict_at_m": round(float(np.nanmin(lo)), 2)}
    cand = np.round(np.arange(0.0, cfg.v_max + 1e-9, 0.05), 3)
    ok = [v for v in cand if safe(v, cfg.time_margin if v > v_want else 0.0)]
    if not ok:
        if rear_free > 0.30:
            return "away", -0.15, dict(info, why="it is coming at the car - backing away")
        return "stop", 0.0, dict(info, why="no safe speed and no room behind - stopped")
    best = min(ok, key=lambda v: abs(v - v_want))
    if best > v_want + 1e-6:
        if best - v_want > cfg.pass_gain_max:                        # would need too big a burst: yield instead
            slower = [v for v in ok if v < v_want]
            if slower:
                best = max(slower)
            else:
                return "pass", min(best, v_want + cfg.pass_gain_max), dict(info, why="speeding up to get past first")
        else:
            return "pass", best, dict(info, why="speeding up to get past first")
    if best < v_want - 1e-6:
        return ("wait" if best == 0.0 else "yield"), best, dict(info, why="letting it pass" if best > 0 else
                                                             "stopping to let it pass")
    return "clear", v_want, info
