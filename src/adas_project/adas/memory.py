"""Short-term obstacle memory for the LiDAR blind ring.

The LiDAR cannot see anything closer than its minimum range, and it sits on
the car's centre line, so anything within about 13 cm of the car's side (or
just in front of the bumper) vanishes from the scan. This keeps obstacles seen
a moment ago and dead-reckons them as the car moves, then hands back only
those that are currently inside the blind ring.

Dead reckoning uses the estimated speed and steering, so it is only trusted
for a few seconds and only inside the blind ring. Where a scan-matched pose exists
(the relay's LocalMapOdometry, ~1 % drift against 5-10 % for speed x steering),
`correct_pose` puts the memory on it after every scan and dead reckoning only
bridges the 0.1 s between scans.
"""

import math

import numpy as np

from adas.visibility import evidence, width_from_neighbours

try:                                   # imported once at start-up: importing inside the scan callback cost ~0.1 s
    from scipy.spatial import cKDTree
except ImportError:                    # pragma: no cover
    cKDTree = None

VOXEL = 0.02                           # m: remembered points are de-duplicated on this grid
CONTACT_TRAVEL_M = 20.0                # contact points (virtual bumper) are kept for this much driving
SCORE_MAX = 8.0                        # clamped evidence score per point (adas/visibility.py): hits add 1, see-through misses subtract


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
        self.score = np.empty(0)                      # evidence per point (adas/visibility.py)
        self.contact = np.empty(0, bool)              # from the virtual bumper (adas/contact.py): an object the LiDAR cannot see
        self._judged = False                         # this scan's evidence already applied (prune_contradicted before add_scan)

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
        # 1) evidence from this scan: seen again / seen through (thin objects far away are not judged - adas/visibility.py)
        if not self._judged:
            self.judge(points)
        self._judged = False

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
                self.score = np.concatenate([self.score, np.ones(len(near))])
                self.contact = np.concatenate([self.contact, np.zeros(len(near), bool)])
                # one point per 2 cm cell, the newest (the scan repeats the same wall ten times a second), with the cell's best score
                cells = np.floor(self.pts / VOXEL).astype(np.int64)
                key = cells[:, 0] * 1_000_003 + cells[:, 1]
                uk, inv = np.unique(key, return_inverse=True)
                best = np.zeros(len(uk))
                np.maximum.at(best, inv, self.score)
                cbest = np.zeros(len(uk), bool)
                np.logical_or.at(cbest, inv, self.contact)
                _, last = np.unique(key[::-1], return_index=True)
                idx = len(key) - 1 - last
                self.score, self.contact = best[inv], cbest[inv]
                self._take(idx)

        if self.max_travel is None:
            keep = self.age < self.max_age
        else:
            keep = (self.travel - self.seen_at < self.max_travel) & (self.age < self.max_age)
        keep |= self.contact & (self.travel - self.seen_at < CONTACT_TRAVEL_M)     # invisible to the LiDAR: kept much longer
        self._take(keep)
        if len(self.pts) > self.max_points:
            # thin the points far from the car first; the ones in and near the blind ring matter most
            v = self._to_vehicle(self.pts)
            d = np.hypot(v[:, 0] - self.p.lidar_x, v[:, 1] - self.p.lidar_y)
            order = np.argsort(np.where(self.contact, -1.0, d))[: self.max_points]     # contact points first, then the nearest
            self._take(order)

    def _take(self, idx):
        """Keep the points idx (index array or mask) in every per-point array."""
        self.pts, self.age, self.seen_at, self.score, self.contact = (self.pts[idx], self.age[idx], self.seen_at[idx], self.score[idx],
                                                                      self.contact[idx])

    def prune_contradicted(self, scan_pts, **_):
        """The scan's evidence against the memory (judge); the add_scan for the same scan then does not apply it again."""
        self.judge(scan_pts)
        self._judged = True

    def judge(self, scan_pts):
        """Apply one scan's evidence (adas/visibility.py): +1 for a point seen again, minus the miss weight for a point the scan saw
        through - only beams within the point's own angular size count, and thin points far away are not judged; a point is dropped
        at zero. Dead reckoning drifts, so a remembered wall can "approach" the car - the scan seeing past it removes it. A bearing
        with no return keeps its points (that is what a truly close object looks like)."""
        if not len(self.pts) or scan_pts is None or not len(scan_pts):
            return
        lx, ly = self.p.lidar_x, self.p.lidar_y
        v = self._to_vehicle(self.pts)
        mb = np.degrees(np.arctan2(v[:, 1] - ly, v[:, 0] - lx))
        md = np.hypot(v[:, 0] - lx, v[:, 1] - ly)
        sb = np.degrees(np.arctan2(scan_pts[:, 1] - ly, scan_pts[:, 0] - lx))
        sd = np.hypot(scan_pts[:, 0] - lx, scan_pts[:, 1] - ly)
        hit, miss = evidence(md, mb, sd, sb, min_range=self.blind_radius, half_width=width_from_neighbours(self.pts, VOXEL))
        self.score = np.minimum(SCORE_MAX, self.score + hit) - miss
        keep = self.score > 0
        self._take(keep)

    def correct_pose(self, pose, max_jump=0.25, max_turn_deg=15.0, stamp=None):
        """Take a better pose (x, y, theta; the scan-matched world pose) for the car in the memory's frame. A jump larger than a scan's
        worth of motion is a frame change (the world origin was reset, a relocalisation): then the remembered points keep their place
        around the car instead. `stamp`: the pose's update counter - the same stamp again (no new scan matched) is ignored, as a stale
        pose would pull the car back."""
        if stamp is not None:
            if stamp == getattr(self, "_pose_stamp", None):
                return
            self._pose_stamp = stamp
        x, y, th = (float(v) for v in pose)
        jump = math.hypot(x - self.x, y - self.y) > max_jump or abs(math.remainder(th - self.theta, 2 * math.pi)) > math.radians(max_turn_deg)
        if jump and len(self.pts):
            v = self._to_vehicle(self.pts)
            self.x, self.y, self.theta = x, y, th
            self.pts = self._to_local(v)
        else:
            self.x, self.y, self.theta = x, y, th

    def clear(self):
        """Forget everything (the car was moved by hand / the simulator teleported it: nothing remembered is where it was)."""
        self.pts, self.age, self.seen_at, self.score, self.contact = np.empty((0, 2)), np.empty(0), np.empty(0), np.empty(0), np.empty(0, bool)

    def add_bumper(self, direction, depth=0.02, n=9):
        """A contact the LiDAR cannot see (adas/contact.ContactDetector): remember the bumper line just ahead of the front (direction
        +1) or behind the rear (-1), at full score - nothing returned there, so only a scan that sees through it can remove it."""
        x = self.p.front_x + depth if direction > 0 else self.p.rear_x - depth
        hw = self.p.width / 2
        pts = np.column_stack([np.full(n, x), np.linspace(-hw, hw, n)])
        self.pts = np.vstack([self.pts, self._to_local(pts)])
        self.age = np.concatenate([self.age, np.zeros(n)])
        self.seen_at = np.concatenate([self.seen_at, np.full(n, self.travel)])
        self.score = np.concatenate([self.score, np.full(n, SCORE_MAX)])
        self.contact = np.concatenate([self.contact, np.ones(n, bool)])

    def blind_points(self):
        """Remembered points the LiDAR cannot see right now, in the vehicle frame: inside the blind ring, and contact points (virtual
        bumper) at any distance up to 3 m."""
        if not len(self.pts):
            return np.empty((0, 2))
        v = self._to_vehicle(self.pts)
        d = np.hypot(v[:, 0] - self.p.lidar_x, v[:, 1] - self.p.lidar_y)
        return v[(d < self.blind_radius) | (self.contact & (d < 3.0))]      # contact points: invisible at any distance
