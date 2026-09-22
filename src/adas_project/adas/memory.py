"""Short-term obstacle memory for the LiDAR blind ring.

The LiDAR cannot see anything closer than its minimum range, and it sits on
the car's centre line, so anything within about 13 cm of the car's side (or
just in front of the bumper) vanishes from the scan. This keeps obstacles seen
a moment ago and dead-reckons them as the car moves, then hands back only
those that are currently inside the blind ring.

Dead reckoning uses the estimated speed and steering, so it is only trusted
for a few seconds and only inside the blind ring.
"""

import math

import numpy as np


class ObstacleMemory:
    def __init__(self, p, blind_radius=0.24, keep_radius=1.2, max_age=3.0, max_points=400):
        self.p = p
        self.blind_radius = blind_radius
        self.keep_radius = keep_radius
        self.max_age = max_age
        self.max_points = max_points
        self.x = self.y = self.theta = 0.0            # dead-reckoned pose in a local frame
        self.pts = np.empty((0, 2))                   # remembered points, local frame
        self.age = np.empty(0)

    def advance(self, dt, v, delta):
        omega = v * math.tan(delta) / self.p.wheelbase
        mid = self.theta + 0.5 * omega * dt
        self.x += v * math.cos(mid) * dt
        self.y += v * math.sin(mid) * dt
        self.theta += omega * dt
        self.age += dt

    def _to_local(self, pts):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.column_stack([self.x + c * pts[:, 0] - s * pts[:, 1],
                                self.y + s * pts[:, 0] + c * pts[:, 1]])

    def _to_vehicle(self, pts):
        c, s = math.cos(self.theta), math.sin(self.theta)
        dx, dy = pts[:, 0] - self.x, pts[:, 1] - self.y
        return np.column_stack([c * dx + s * dy, -s * dx + c * dy])

    def add_scan(self, points, exclude=None):
        """points: the new scan (vehicle frame). exclude: (x, y, radius) of moving objects, not remembered."""
        # 1) forget remembered points that this scan can see are no longer there
        if len(self.pts) and len(points):
            from scipy.spatial import cKDTree
            v = self._to_vehicle(self.pts)
            d_lidar = np.hypot(v[:, 0] - self.p.lidar_x, v[:, 1] - self.p.lidar_y)
            dist, _ = cKDTree(points).query(v)
            visible = d_lidar > self.blind_radius + 0.03
            keep = ~visible | (dist < 0.10)
            self.pts, self.age = self.pts[keep], self.age[keep]

        # 2) remember the near part of the new scan, except moving objects
        if len(points):
            d = np.hypot(points[:, 0] - self.p.lidar_x, points[:, 1] - self.p.lidar_y)
            near = points[d < self.keep_radius]
            if exclude and len(near):
                ok = np.ones(len(near), dtype=bool)
                for ex, ey, er in exclude:
                    ok &= np.hypot(near[:, 0] - ex, near[:, 1] - ey) > er + 0.08
                near = near[ok]
            if len(near):
                self.pts = np.vstack([self.pts, self._to_local(near)])
                self.age = np.concatenate([self.age, np.zeros(len(near))])

        keep = self.age < self.max_age
        self.pts, self.age = self.pts[keep], self.age[keep]
        if len(self.pts) > self.max_points:
            idx = np.linspace(0, len(self.pts) - 1, self.max_points).astype(int)
            self.pts, self.age = self.pts[idx], self.age[idx]

    def blind_points(self):
        """Remembered points that lie inside the blind ring right now, in the vehicle frame."""
        if not len(self.pts):
            return np.empty((0, 2))
        v = self._to_vehicle(self.pts)
        d = np.hypot(v[:, 0] - self.p.lidar_x, v[:, 1] - self.p.lidar_y)
        return v[d < self.blind_radius]
