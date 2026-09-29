"""Moving-object tracking and collision prediction from LiDAR scans.

Each scan is split into compact clusters (pedestrians, cones, boxes, other
cars; long walls are ignored). Clusters are followed from scan to scan, and a
straight-line fit over the last ~0.5 s gives each one's velocity as seen from
the car. Subtracting the motion the car itself causes (from its speed and
steering) leaves the object's own velocity. Objects that really move are then
projected forward in time and checked against the car's footprint moving along
its steering arc.

Only relative motion is used, so no wheel encoder is needed. The car's speed
only has to be roughly right, and the check tolerates +/-30 % error in it.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .geometry import path_pose
from .vehicle_params import VehicleParams


@dataclass
class Track:
    id: int
    hist: deque = field(default_factory=lambda: deque(maxlen=8))   # (t, x, y) in each scan's vehicle frame
    radius: float = 0.03
    missed: int = 0
    vel_rel: tuple = (0.0, 0.0)     # velocity seen from the car
    vel_obj: tuple = (0.0, 0.0)     # the object's own velocity (car motion removed)
    moving: bool = False
    moving_count: int = 0

    @property
    def pos(self):
        return self.hist[-1][1], self.hist[-1][2]


def cluster_points(points, gap=0.08, min_pts=3, max_extent=0.45):
    """Split ordered scan points into compact clusters. Returns [(centroid, radius)]."""
    if len(points) < min_pts:
        return []
    steps = np.hypot(*np.diff(points, axis=0).T)
    groups = np.split(np.arange(len(points)), np.flatnonzero(steps > gap) + 1)

    out = []
    for g in groups:
        if len(g) < min_pts:
            continue
        pts = points[g]
        centre = pts.mean(axis=0)
        radius = float(np.hypot(*(pts - centre).T).max())
        if 2.0 * radius > max_extent:
            continue
        out.append((centre, max(radius, 0.02)))
    return out


def static_velocity(x, y, v, omega):
    """Velocity, in the vehicle frame, of a point that is fixed in the world."""
    return (-v + omega * y, -omega * x)


class Tracker:
    def __init__(self, gate=0.30, min_samples=4, max_missed=3, move_threshold=0.10, persist=3):
        self.gate = gate
        self.min_samples = min_samples
        self.max_missed = max_missed
        self.move_threshold = move_threshold
        self.persist = persist
        self.tracks = []
        self._next_id = 1

    def update(self, points, t, v, omega):
        """v: signed speed estimate (m/s), omega: yaw rate (rad/s). Returns the track list."""
        clusters = cluster_points(points)
        unmatched = list(range(len(clusters)))

        for tr in self.tracks:
            tr.missed += 1
            if not unmatched:
                continue
            dt = t - tr.hist[-1][0]
            px = tr.pos[0] + tr.vel_rel[0] * dt
            py = tr.pos[1] + tr.vel_rel[1] * dt
            dists = [math.hypot(clusters[i][0][0] - px, clusters[i][0][1] - py) for i in unmatched]
            k = int(np.argmin(dists))
            if dists[k] < self.gate:
                i = unmatched.pop(k)
                (cx, cy), r = clusters[i]
                tr.hist.append((t, float(cx), float(cy)))
                tr.radius = 0.7 * tr.radius + 0.3 * r
                tr.missed = 0

        for i in unmatched:
            (cx, cy), r = clusters[i]
            tr = Track(self._next_id, radius=r)
            tr.hist.append((t, float(cx), float(cy)))
            self._next_id += 1
            self.tracks.append(tr)

        self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed]
        for tr in self.tracks:
            while len(tr.hist) > 2 and t - tr.hist[0][0] > 0.8:
                tr.hist.popleft()
            self._estimate(tr, v, omega)
        return self.tracks

    def _estimate(self, tr, v, omega):
        tr.moving = False
        tr.vel_obj = (0.0, 0.0)
        if len(tr.hist) < self.min_samples:
            return
        ts = np.array([h[0] for h in tr.hist])
        if ts[-1] - ts[0] < 0.25:
            tr.moving_count = 0
            return
        vx = float(np.polyfit(ts, [h[1] for h in tr.hist], 1)[0])
        vy = float(np.polyfit(ts, [h[2] for h in tr.hist], 1)[0])
        tr.vel_rel = (vx, vy)

        x, y = tr.pos
        # The car's speed and turn rate are only approximate (both scale together
        # with speed), so let a single factor k in [0.7, 1.3] explain as much of the
        # motion as possible as a static object. Only what is left over counts.
        ux, uy = static_velocity(x, y, v, omega)
        norm2 = ux * ux + uy * uy
        k = 1.0 if norm2 < 1e-9 else min(max((vx * ux + vy * uy) / norm2, 0.7), 1.3)
        rx, ry = vx - k * ux, vy - k * uy
        tr.vel_obj = (rx, ry)

        if math.hypot(rx, ry) > self.move_threshold:
            tr.moving_count += 1
        else:
            tr.moving_count = 0
        tr.moving = tr.moving_count >= self.persist


def moving_object_contact(tracks, delta, direction, v, p, horizon=2.0, dt=0.02, margin=0.02):
    """Earliest predicted collision with a MOVING object if the car keeps its speed and steering.

    Returns (travel_distance, time, track_id). Distance is how far the car
    travels before contact; inf if no collision is predicted within `horizon`.
    """
    best = (math.inf, math.inf, None)
    movers = [t for t in tracks if t.moving]
    if not movers or v < 0.03:
        return best

    tau = np.arange(0.0, horizon + dt, dt)
    ex, ey, epsi = path_pose(delta, direction * v * tau, p)
    cos_p, sin_p = np.cos(epsi), np.sin(epsi)

    for tr in movers:
        ox = tr.pos[0] + tr.vel_obj[0] * tau
        oy = tr.pos[1] + tr.vel_obj[1] * tau
        dx, dy = ox - ex, oy - ey
        lx = cos_p * dx + sin_p * dy
        ly = -sin_p * dx + cos_p * dy
        gx = np.maximum(np.maximum(p.rear_x - lx, 0.0), lx - p.front_x)
        gy = np.maximum(np.abs(ly) - p.width / 2.0, 0.0)
        hit = np.flatnonzero(np.hypot(gx, gy) <= margin + tr.radius)
        if len(hit) and tau[hit[0]] < best[1]:
            best = (float(v * tau[hit[0]]), float(tau[hit[0]]), tr.id)
    return best
