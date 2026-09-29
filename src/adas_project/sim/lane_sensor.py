"""Simulated lane camera: the floor points of lane lines that the real camera would see."""

import math

import numpy as np

from adas.markers import Camera, project


class LaneSensor:
    def __init__(self, camera=None, rate_hz=15.0, noise=0.006, dropout=0.08, seed=0):
        self.cam = camera or Camera()
        self.period = 1.0 / rate_hz
        self.noise, self.dropout = noise, dropout
        self.rng = np.random.default_rng(seed)
        self.next_time = 0.0

    def sense(self, world, car, t):
        """Ground points (N, 2) in the vehicle frame, or None if it is not time for a frame."""
        if t < self.next_time:
            return None
        self.next_time = t + self.period
        c, s = math.cos(car.theta), math.sin(car.theta)
        cam = self.cam
        out = []
        for o in world.objects:
            if not getattr(o, "lane", False):
                continue
            pts = np.asarray(o.pts, dtype=float)
            for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
                n = max(2, int(math.hypot(x2 - x1, y2 - y1) / 0.02))
                xs, ys = np.linspace(x1, x2, n), np.linspace(y1, y2, n)
                dx, dy = xs - car.x, ys - car.y
                out.append(np.column_stack([c * dx + s * dy, -s * dx + c * dy]))
        if not out:
            return np.empty((0, 2))
        g = np.vstack(out)
        g = g[(g[:, 0] > 0.15) & (g[:, 0] < 1.6) & (np.abs(g[:, 1]) < 0.7)]
        if not len(g):
            return g
        px, depth = project(cam, np.column_stack([g, np.zeros(len(g))]))
        vis = (depth > 0.02) & (px[:, 0] > 0) & (px[:, 0] < cam.width) & (px[:, 1] > cam.height / 2) & (px[:, 1] < cam.height)
        g = g[vis]
        g = g[self.rng.random(len(g)) > self.dropout]
        return g + self.rng.normal(0.0, self.noise, g.shape)
