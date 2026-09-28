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

try:                                   # imported once at start-up: importing inside the scan callback cost ~0.1 s
    from scipy.spatial import cKDTree
except ImportError:                    # pragma: no cover
    cKDTree = None

VOXEL = 0.02                           # m: remembered points are de-duplicated on this grid


class ObstacleMemory:
    def __init__(self, p, blind_radius=0.24, keep_radius=1.2, max_age=3.0, max_points=400, max_travel=None):
        """max_travel (m): if set, a point is also kept until the car has driven this far since seeing it -
        a parked car next to a wall must not forget the wall just because time passed."""
        self.p = p
        self.max_travel = max_travel
        self.travel = 0.0                             # distance driven so far (m)
        self.seen_at = np.empty(0)                    # self.travel when each point was stored
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
        self.travel += abs(v) * dt

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
        if len(self.pts) and len(points) and cKDTree is not None:
            v = self._to_vehicle(self.pts)
            d_lidar = np.hypot(v[:, 0] - self.p.lidar_x, v[:, 1] - self.p.lidar_y)
            visible = d_lidar > self.blind_radius + 0.03
            keep = ~visible
            if visible.any():
                dist, _ = cKDTree(points).query(v[visible], distance_upper_bound=0.10)
                keep[visible] = np.isfinite(dist)
            self.pts, self.age, self.seen_at = self.pts[keep], self.age[keep], self.seen_at[keep]

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
                self.seen_at = np.concatenate([self.seen_at, np.full(len(near), self.travel)])
                # one point per 2 cm cell, the newest (the scan repeats the same wall ten times a second)
                cells = np.floor(self.pts / VOXEL).astype(np.int64)
                key = cells[:, 0] * 1_000_003 + cells[:, 1]
                _, last = np.unique(key[::-1], return_index=True)
                idx = len(key) - 1 - last
                self.pts, self.age, self.seen_at = self.pts[idx], self.age[idx], self.seen_at[idx]

        if self.max_travel is None:
            keep = self.age < self.max_age
        else:
            keep = (self.travel - self.seen_at < self.max_travel) & (self.age < self.max_age)
        self.pts, self.age, self.seen_at = self.pts[keep], self.age[keep], self.seen_at[keep]
        if len(self.pts) > self.max_points:
            # thin the points far from the car first; the ones in and near the blind ring matter most
            v = self._to_vehicle(self.pts)
            d = np.hypot(v[:, 0] - self.p.lidar_x, v[:, 1] - self.p.lidar_y)
            order = np.argsort(d)[: self.max_points]
            self.pts, self.age, self.seen_at = self.pts[order], self.age[order], self.seen_at[order]

    def prune_contradicted(self, scan_pts, beam_deg=4.0, slack=0.05):
        """Drop remembered points the current scan proves are not there: at that bearing (from the LiDAR) the
        sensor sees something FURTHER away, so the space in between is empty. Dead reckoning from an estimated
        speed drifts; without this a remembered wall can "approach" the car and block it with nothing there.
        A bearing with no return at all keeps its points (that is what a truly blind-close object looks like)."""
        if not len(self.pts) or not len(scan_pts):
            return
        lx, ly = self.p.lidar_x, self.p.lidar_y
        v = self._to_vehicle(self.pts)
        mb = np.degrees(np.arctan2(v[:, 1] - ly, v[:, 0] - lx))
        md = np.hypot(v[:, 0] - lx, v[:, 1] - ly)
        sb = np.degrees(np.arctan2(scan_pts[:, 1] - ly, scan_pts[:, 0] - lx))
        sd = np.hypot(scan_pts[:, 0] - lx, scan_pts[:, 1] - ly)
        # farthest return per 1-degree bearing bin, then the max over +-beam_deg (wrapping round): vectorised, no
        # Python loop over the remembered points
        far = np.full(360, -1.0)
        np.maximum.at(far, np.floor(sb).astype(int) % 360, sd)
        w = int(math.ceil(beam_deg))
        win = far.copy()
        for k in range(1, w + 1):
            win = np.maximum(win, np.maximum(np.roll(far, k), np.roll(far, -k)))
        keep = win[np.floor(mb).astype(int) % 360] <= md + slack
        self.pts, self.age, self.seen_at = self.pts[keep], self.age[keep], self.seen_at[keep]

    def blind_points(self):
        """Remembered points that lie inside the blind ring right now, in the vehicle frame."""
        if not len(self.pts):
            return np.empty((0, 2))
        v = self._to_vehicle(self.pts)
        d = np.hypot(v[:, 0] - self.p.lidar_x, v[:, 1] - self.p.lidar_y)
        return v[d < self.blind_radius]
