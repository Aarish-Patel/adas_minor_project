"""Room exploration: the car maps an unknown room by itself (TODO F1 / C6).

Frontier-based exploration (Yamauchi, "A frontier-based approach for autonomous exploration", CIRA 1997): keep an occupancy
grid, find the frontier - known-free cells that touch unknown space - and drive to the nearest reachable one; repeat until no
reachable frontier is left. Occupancy from LiDAR by ray casting into a log-odds grid (Moravec & Elfes 1985; Thrun, Burgard, Fox,
Probabilistic Robotics 2005, ch. 9). Driving to a frontier is the existing click-to-go (Hybrid A*, adas/autonav.py), so
exploration inherits its collision handling and the operator's dead-man throttle.

    grid = OccupancyGrid()                       # world frame = the start pose (x ahead, y left)
    grid.update(pose, lidar_xy_vehicle, lidar_x) # every scan, pose from the scan-matched odometry
    goal = pick_frontier(grid, pose)             # (x, y) world or None when everything reachable is explored
"""
import collections
import math

import numpy as np

UNKNOWN, FREE, OCC = 0, 1, 2


class OccupancyGrid:
    def __init__(self, half=8.0, res=0.05, max_range=4.0):
        self.res, self.half, self.max_range = res, half, max_range
        self.n = int(2 * half / res)
        self.logodds = np.zeros((self.n, self.n), np.float32)     # 0 = unknown; > 0 occupied, < 0 free
        self.seen = np.zeros((self.n, self.n), bool)
        self.hits = np.zeros((self.n, self.n), np.uint8)

    def cell(self, x, y):
        return int((x + self.half) / self.res), int((y + self.half) / self.res)

    def world(self, i, j):
        return (i + 0.5) * self.res - self.half, (j + 0.5) * self.res - self.half

    def update(self, pose, pts_vehicle, lidar_x=0.12, stride=1):
        """pts_vehicle: LiDAR returns in the vehicle frame (N, 2). Free space along every stride-th ray, occupied at its end."""
        if len(pts_vehicle) == 0:
            return
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        ox, oy = x + lidar_x * c, y + lidar_x * s
        p = np.asarray(pts_vehicle, float)[::stride]
        wx = x + c * p[:, 0] - s * p[:, 1]
        wy = y + s * p[:, 0] + c * p[:, 1]
        d = np.hypot(wx - ox, wy - oy)
        keep = (d > 0.15) & (d < self.max_range)
        wx, wy, d = wx[keep], wy[keep], d[keep]
        steps = max(1, int(self.max_range / self.res))
        t = np.linspace(0.0, 1.0, steps)[None, :]
        fx = ox + (wx - ox)[:, None] * t
        fy = oy + (wy - oy)[:, None] * t
        valid = t * d[:, None] < d[:, None] - 0.5 * self.res          # stop short of the hit cell
        i = ((fx + self.half) / self.res).astype(int)
        j = ((fy + self.half) / self.res).astype(int)
        ok = valid & (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        self.logodds[i[ok], j[ok]] -= 0.35
        self.seen[i[ok], j[ok]] = True
        hi = ((wx + self.half) / self.res).astype(int)
        hj = ((wy + self.half) / self.res).astype(int)
        okh = (hi >= 0) & (hi < self.n) & (hj >= 0) & (hj < self.n)
        self.logodds[hi[okh], hj[okh]] += 0.9
        np.add.at(self.hits, (hi[okh], hj[okh]), 1)
        self.hits = np.minimum(self.hits, 250)
        self.seen[hi[okh], hj[okh]] = True
        np.clip(self.logodds, -4.0, 4.0, out=self.logodds)

    def state(self):
        g = np.full((self.n, self.n), UNKNOWN, np.uint8)
        g[self.seen & (self.logodds < -0.2)] = FREE
        g[self.seen & (self.hits > 0) & (self.logodds >= -0.2)] = OCC
        return g

    def known_free_area(self):
        return float(np.count_nonzero(self.state() == FREE)) * self.res ** 2


def _inflate(mask, cells):
    """Grow a boolean mask by `cells` (square structuring element, separable shifts)."""
    out = mask.copy()
    for _ in range(cells):
        g = out.copy()
        g[1:, :] |= out[:-1, :]
        g[:-1, :] |= out[1:, :]
        g[:, 1:] |= out[:, :-1]
        g[:, :-1] |= out[:, 1:]
        out = g
    return out


def frontiers(grid, robot_radius=0.16, min_cells=4):
    """(labels of frontier clusters, free-and-safe mask). A frontier cell is free, safe (not within the robot's radius of an
    obstacle) and has an unknown 4-neighbour."""
    from scipy import ndimage
    st = grid.state()
    occ = st == OCC
    safe = (st == FREE) & ~_inflate(occ, int(round(robot_radius / grid.res)))
    unk = st == UNKNOWN
    near_unknown = np.zeros_like(unk)
    near_unknown[1:, :] |= unk[:-1, :]
    near_unknown[:-1, :] |= unk[1:, :]
    near_unknown[:, 1:] |= unk[:, :-1]
    near_unknown[:, :-1] |= unk[:, 1:]
    front = safe & near_unknown
    lab, n = ndimage.label(front, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(front, lab, index=np.arange(1, n + 1))
        for k, sz in enumerate(sizes, start=1):
            if sz < min_cells:
                lab[lab == k] = 0
    return lab, safe


def pick_frontier(grid, pose, blacklist=(), min_dist=0.5, robot_radius=0.16):
    """The nearest reachable frontier cell (BFS distance over free, safe cells), as world (x, y), or None."""
    lab, safe = frontiers(grid, robot_radius)
    if not lab.any():
        return None
    start = grid.cell(pose[0], pose[1])
    n = grid.n
    if not (0 <= start[0] < n and 0 <= start[1] < n):
        return None
    # the car's own neighbourhood (0.3 m) counts as safe: its body and the LiDAR's blind ring must not trap the search
    r = int(0.3 / grid.res)
    lo_i, hi_i, lo_j, hi_j = max(0, start[0] - r), min(n, start[0] + r + 1), max(0, start[1] - r), min(n, start[1] + r + 1)
    ii, jj = np.meshgrid(np.arange(lo_i, hi_i), np.arange(lo_j, hi_j), indexing="ij")
    disk = (ii - start[0]) ** 2 + (jj - start[1]) ** 2 <= r * r
    st_ = grid.state()
    safe = safe.copy()
    sub = safe[lo_i:hi_i, lo_j:hi_j]
    sub |= disk & (st_[lo_i:hi_i, lo_j:hi_j] != OCC)
    dist = np.full((n, n), -1, np.int32)
    q = collections.deque()
    for di in range(-4, 5):
        for dj in range(-4, 5):
            i, j = start[0] + di, start[1] + dj
            if 0 <= i < n and 0 <= j < n and safe[i, j] and dist[i, j] < 0:
                dist[i, j] = 0
                q.append((i, j))
    if not q:
        return None
    bl = [grid.cell(*b) for b in blacklist]
    best = None
    while q:
        i, j = q.popleft()
        if lab[i, j] and math.hypot(*(np.array(grid.world(i, j)) - np.array(pose[:2]))) >= min_dist and \
                all(abs(i - bi) > 6 or abs(j - bj) > 6 for bi, bj in bl):
            best = (i, j)
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < n and safe[a, b] and dist[a, b] < 0:
                dist[a, b] = dist[i, j] + 1
                q.append((a, b))
    return None if best is None else grid.world(*best)


class Explorer:
    """Drives an exploration: call `step(pose, pts_vehicle, nav_active, goto)` on every scan; `goto(x_vehicle, y_vehicle)`
    starts click-to-go and returns True when a path was found. Finishes when no reachable frontier is left."""

    def __init__(self, lidar_x=0.12, replan_s=1.0, blacklist_s=60.0):
        self.grid = OccupancyGrid()
        self.lidar_x = lidar_x
        self.replan_s, self.blacklist_s = replan_s, blacklist_s
        self.blacklist = []                          # (x, y, until)
        self.done, self.msg, self.goal, self._t_plan, self.goals_sent = False, "exploring", None, -9.0, 0

    def step(self, t, pose, pts_vehicle, nav_active, goto):
        self.grid.update(pose, pts_vehicle, self.lidar_x)
        if self.done or nav_active or t - self._t_plan < self.replan_s:
            return
        self._t_plan = t
        self.blacklist = [b for b in self.blacklist if b[2] > t]
        g = pick_frontier(self.grid, pose, [(b[0], b[1]) for b in self.blacklist])
        if g is None:
            self.done, self.msg, self.goal = True, "explored: no reachable frontier left", None
            return
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        dx, dy = g[0] - x, g[1] - y
        gv = (c * dx + s * dy, -s * dx + c * dy)     # world -> vehicle
        self.goal = g
        # every goal is blacklisted for a while once tried: a frontier that stays a frontier after the car has been there (a
        # sliver at a wall the LiDAR cannot resolve) must not be driven to again and again
        self.blacklist.append((g[0], g[1], t + self.blacklist_s))
        if goto(gv[0], gv[1]):
            self.goals_sent += 1
            self.msg = f"exploring: frontier {g[0]:.1f}, {g[1]:.1f}"
