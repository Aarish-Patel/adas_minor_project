"""Virtual 360-degree 2D LiDAR (modelled on the RPLIDAR A3M1)."""

import math

import numpy as np


class LidarSim:
    def __init__(self, n_points=1600, rate_hz=10.0, min_range=0.2, max_range=8.0,
                 noise_std=0.01, dropout=0.01, seed=0):
        self.n = n_points
        self.period = 1.0 / rate_hz
        self.min_range = min_range
        self.max_range = max_range
        self.noise_std = noise_std
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)
        self.angles = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)

    def scan(self, world, x, y, theta, p):
        """Returns (angles, ranges, valid). Angles are in the vehicle frame."""
        c, s = math.cos(theta), math.sin(theta)
        ox = x + c * p.lidar_x - s * p.lidar_y
        oy = y + s * p.lidar_x + c * p.lidar_y

        world_ang = self.angles + theta
        d = np.column_stack([np.cos(world_ang), np.sin(world_ang)])
        best = np.full(self.n, np.inf)

        seg = world.segments()
        if len(seg):
            ex, ey = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
            wx, wy = seg[:, 0] - ox, seg[:, 1] - oy
            denom = d[:, 0, None] * ey[None, :] - d[:, 1, None] * ex[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (wx * ey - wy * ex)[None, :] / denom
                u = (wx[None, :] * d[:, 1, None] - wy[None, :] * d[:, 0, None]) / denom
            hit = (np.abs(denom) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
            best = np.minimum(best, np.where(hit, t, np.inf).min(axis=1))

        circ = world.circles()
        if len(circ):
            fx, fy = ox - circ[:, 0], oy - circ[:, 1]
            b = d[:, 0, None] * fx[None, :] + d[:, 1, None] * fy[None, :]
            cc = (fx * fx + fy * fy - circ[:, 2] ** 2)[None, :]
            disc = b * b - cc
            with np.errstate(invalid="ignore"):
                t = -b - np.sqrt(disc)
            hit = (disc >= 0) & (t > 0)
            best = np.minimum(best, np.where(hit, t, np.inf).min(axis=1))

        ranges = best + self.rng.normal(0.0, self.noise_std, self.n)
        valid = np.isfinite(best) & (ranges >= self.min_range) & (ranges <= self.max_range)
        valid &= self.rng.random(self.n) > self.dropout
        return self.angles, np.where(valid, ranges, np.nan), valid
