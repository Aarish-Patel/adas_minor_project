"""Hybrid A* (Dolgov, Thrun, Montemerlo, Diebel, IJRR 2010) for the car's evasive and autonomous manoeuvres.

Frame: the "line frame" - x along the driver's desired path, y left, the car starting at (0, 0, heading 0) unless
given otherwise. Obstacles are LiDAR points (+ remembered points) in that frame.

  grid        occupancy at RES metres, Euclidean distance transform to the nearest obstacle
  body        three overlapping circles along the car's length (exact enough, and a single lookup per circle)
  search      continuous (x, y, heading) nodes, closed set on a (x, y, heading) lattice; motion primitives are arcs
              the car can drive (curvatures up to KAPPA_MAX), forward - and backward only in "unstuck" mode
  cost        path length, steering effort and steering changes, distance from the desired line, and a penalty for
              low clearance - so the chosen path is the smallest safe detour, not just any path
  heuristic   max(Euclidean, obstacle-aware grid distance) to the goal region
  goal        back on the desired line (|y| small, heading along it) beyond the obstacle, with the line ahead free;
              reached by an analytic expansion: a smooth rejoin curve tried from nodes, accepted if collision free
"""
import heapq
import math

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:              # pragma: no cover
    distance_transform_edt = None

RES = 0.05
# closed-set resolution. Forward swerves need 7.5 deg heading bins (each step turns only 4-13 deg: coarser bins
# merge the states a doorway swerve needs). Searches WITH reversing use 15 deg: on 31 back-off jobs captured from
# Monte Carlo drives they then find 26 paths instead of 15 with a third of the nodes (0.2 s instead of 0.54 s).
HEADING_BIN_DEG = 7.5
REVERSE_HEADING_BIN_DEG = 15.0
CELL_M = 0.10
# searches WITH reversing (backing off from something close): heuristic inflation (weighted A*, as in ARA*), a
# longer step and fewer steering primitives - these searches only need a feasible way out, not the shortest.
# On 31 back-off jobs recorded in Monte Carlo drives: same 26 paths found, median 189 -> 100 ms, 95th percentile
# 586 -> 148 ms on a laptop (sweep of eps 1.2-3, step 0.15-0.25 m, 5-7 primitives)
REVERSE_EPS = 3.0
REVERSE_STEP_M = 0.25
REVERSE_N_KAPPA = 5


class Grid:
    def __init__(self, points, x0, x1, y0, y1, res=RES):
        self.x0, self.y0, self.res = x0, y0, res
        self.nx, self.ny = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
        occ = np.zeros((self.nx, self.ny), bool)
        if len(points):
            ix = ((points[:, 0] - x0) / res).astype(int)
            iy = ((points[:, 1] - y0) / res).astype(int)
            ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
            occ[ix[ok], iy[ok]] = True
        self.occ = occ
        if distance_transform_edt is not None:
            self.dist = distance_transform_edt(~occ) * res
        else:                    # brute force fallback
            pts = np.argwhere(occ) * res
            gx, gy = np.meshgrid(np.arange(self.nx) * res, np.arange(self.ny) * res, indexing="ij")
            if len(pts):
                d = np.sqrt((gx[..., None] - pts[:, 0]) ** 2 + (gy[..., None] - pts[:, 1]) ** 2).min(-1)
            else:
                d = np.full(gx.shape, 9.0)
            self.dist = d

    def lookup(self, x, y):
        ix = np.clip(((np.asarray(x) - self.x0) / self.res).astype(int), 0, self.nx - 1)
        iy = np.clip(((np.asarray(y) - self.y0) / self.res).astype(int), 0, self.ny - 1)
        inside = (np.asarray(x) >= self.x0) & (np.asarray(y) >= self.y0) & \
                 (np.asarray(x) < self.x0 + self.nx * self.res) & (np.asarray(y) < self.y0 + self.ny * self.res)
        return np.where(inside, self.dist[ix, iy], 9.0)


class HybridAStar:
    def __init__(self, params, kappa_max=1.5, margin=0.06, step=0.15, clear_wish=0.15,
                 w_offset=0.6, w_steer=0.15, w_change=0.4, w_clear=4.0, w_reverse=3.0):
        self.p = params
        self.kappa_max, self.margin, self.step = kappa_max, margin, step
        self.clear_wish = clear_wish
        self.w_offset, self.w_steer, self.w_change, self.w_clear, self.w_reverse = w_offset, w_steer, w_change, w_clear, w_reverse
        L = params.front_x - params.rear_x
        hw = params.width / 2
        seg = L / 3.0
        self.circle_x = np.array([params.rear_x + seg / 2, params.rear_x + 1.5 * seg, params.rear_x + 2.5 * seg])
        self.circle_r = math.hypot(seg / 2, hw)
        self.expanded = 0

    # ------------------------------------------------------------------ body checks
    def clearance(self, grid, poses):
        """Smallest clearance of the body (circle model) over poses (N,3); negative = collision."""
        poses = np.atleast_2d(poses)
        c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
        cx = poses[:, 0][:, None] + c * self.circle_x[None, :]
        cy = poses[:, 1][:, None] + s * self.circle_x[None, :]
        return float((grid.lookup(cx, cy) - self.circle_r).min())

    @staticmethod
    def arc(pose, kappa, length, n):
        x, y, th = pose
        s = np.linspace(length / n, length, n)
        if abs(kappa) < 1e-6:
            return np.column_stack([x + s * math.cos(th), y + s * math.sin(th), np.full(n, th)])
        ths = th + kappa * s
        return np.column_stack([x + (np.sin(ths) - math.sin(th)) / kappa, y - (np.cos(ths) - math.cos(th)) / kappa, ths])

    def rejoin(self, pose, line_len=0.45, tau=0.25, max_deg=30.0, ds=0.04, max_len=3.0):
        """Analytic expansion: the damped heading law onto y = 0 (as tracked on the car). Returns poses or None."""
        x, y, th = pose
        out = []
        for _ in range(int(max_len / ds)):
            th_des = -math.atan(y / line_len)
            th_des = max(-math.radians(max_deg), min(math.radians(max_deg), th_des))
            k = max(-self.kappa_max, min(self.kappa_max, (th_des - th) / tau))
            th += k * ds
            x += ds * math.cos(th)
            y += ds * math.sin(th)
            out.append((x, y, th))
            if abs(y) < 0.03 and abs(th) < math.radians(3):
                return np.array(out)
        return None

    # ------------------------------------------------------------------ the search
    def plan(self, points, start=(0.0, 0.0, 0.0), x_goal=1.0, allow_reverse=False, max_nodes=2500,
             line_free_ahead=0.4, budget_s=None):
        """Path (N,4: x, y, heading, direction +1/-1) from start back onto the line y = 0 at x >= x_goal, or None."""
        x0 = min(start[0], 0.0) - 1.0
        grid = Grid(points, x0, max(x_goal, start[0]) + 3.0, -2.0, 2.0)
        m = self.margin
        gx0 = int((x_goal - grid.x0) / grid.res)
        iy0 = int((0.0 - grid.y0) / grid.res)
        seeds = [(ix, iy0 + dy) for ix in range(max(0, gx0), grid.nx) for dy in (-1, 0, 1) if 0 <= iy0 + dy < grid.ny]

        def analytic(pose, d_prev):
            if d_prev < 0 or pose[0] <= x_goal - 1.2:
                return None
            rj = self.rejoin(pose)
            if rj is not None and rj[-1, 0] >= x_goal and self.clearance(grid, rj[::2]) >= m and \
                    self._line_free(grid, rj[-1, 0], line_free_ahead):
                return rj, 1
            return None

        return self._search(grid, start, seeds, analytic,
                            lambda e: math.hypot(max(0.0, x_goal - e[0]), e[1]), allow_reverse, max_nodes, self.w_offset,
                            budget_s)

    def plan_to_point(self, points, start, goal, allow_reverse=False, max_nodes=4000, pad=1.5, budget_s=None,
                      goal_heading=None, w_reverse=None, coarse=None):
        """Path to a goal POSITION, e.g. a point picked on the map. goal_heading None: any final heading - the
        analytic expansion is the single circular arc from a node through the goal (what pure pursuit would drive).
        goal_heading (rad): the car must also arrive pointing that way - the analytic expansion is the Dubins path
        to the goal pose (forward, or driven backwards when reversing is allowed), as Dolgov et al. use
        Reeds-Shepp curves. w_reverse: cost factor for reversing (click-to-go prefers driving forward).
        coarse: search on the coarser lattice (15 deg, 0.25 m steps, weighted heuristic) - default with reversing."""
        coarse = allow_reverse if coarse is None else coarse
        gx, gy = goal
        grid = Grid(points, min(start[0], gx) - pad, max(start[0], gx) + pad, min(start[1], gy) - pad,
                    max(start[1], gy) + pad)
        m = self.margin
        if float(grid.lookup(gx, gy)) < self.p.width / 2 + m:
            self.grid = grid
            self.gave_up = "goal unreachable"             # the goal itself is too close to an obstacle
            return None
        seeds = [(int((gx - grid.x0) / grid.res), int((gy - grid.y0) / grid.res))]
        saved_w = self.w_reverse
        if w_reverse is not None:
            self.w_reverse = w_reverse
        try:
            if goal_heading is not None:
                return self._plan_to_pose(grid, start, (gx, gy, goal_heading), seeds, allow_reverse, max_nodes,
                                          budget_s, coarse)
            return self._plan_to_position(grid, start, (gx, gy), seeds, allow_reverse, max_nodes, budget_s, coarse)
        finally:
            self.w_reverse = saved_w

    def _plan_to_pose(self, grid, start, goal, seeds, allow_reverse, max_nodes, budget_s, coarse):
        from . import dubins
        gx, gy, gth = goal
        m = self.margin
        radius = 1.0 / self.kappa_max

        def analytic(pose, d_prev):
            if math.hypot(gx - pose[0], gy - pose[1]) > 3.0:
                return None
            best = None
            for d, fn in ((1, dubins.sample), (-1, dubins.sample_reverse)):
                if d < 0 and not allow_reverse:
                    continue
                P = fn(pose, goal, radius)
                if P is None or len(P) * 0.04 > 4.0:
                    continue
                # no loops: an approach that turns more than 270 deg in total (a Dubins word with a full circle in
                # it, seen on the car, 28 Sep) is refused - the search goes on and finds a sensible way in
                turn = np.abs(np.diff(np.unwrap(np.concatenate([[pose[2]], P[:, 2]])))).sum()
                if turn > math.radians(270):
                    continue
                if self.clearance(grid, P[::2]) >= m and self.clearance(grid, P[-1:]) >= m:
                    cost = len(P) * (1.0 if d > 0 else 1.0 + self.w_reverse)
                    if best is None or cost < best[0]:
                        best = (cost, P, d)
            return None if best is None else (best[1], best[2])

        def h(e):
            # (the obstacle-free Dubins length was tried as a tighter heuristic: 25 % slower overall - dropped)
            return math.hypot(gx - e[0], gy - e[1])

        return self._search(grid, start, seeds, analytic, h, allow_reverse, max_nodes, 0.0, budget_s, coarse)

    def _plan_to_position(self, grid, start, goal, seeds, allow_reverse, max_nodes, budget_s, coarse):
        gx, gy = goal
        m = self.margin

        def analytic(pose, d_prev):
            x, y, th = pose
            if math.hypot(gx - x, gy - y) > 2.0:
                return None
            c, s = math.cos(th), math.sin(th)
            lx, ly = c * (gx - x) + s * (gy - y), -s * (gx - x) + c * (gy - y)
            if lx > 0.05:
                d = 1
            elif lx < -0.05 and allow_reverse:
                d = -1                                    # goal behind: back into it (Reeds-Shepp style)
            else:
                return None
            dd = lx * lx + ly * ly
            k = 2.0 * ly / dd                             # the circle tangent to the heading through the goal
            if abs(k) > self.kappa_max:
                return None
            length = math.sqrt(dd) if abs(k) < 1e-6 else 2.0 * math.atan2(abs(ly), abs(lx)) / abs(k)
            arc = self.arc(pose, k, d * length, max(2, int(length / 0.04)))
            return (arc, d) if self.clearance(grid, arc[::2]) >= m else None

        return self._search(grid, start, seeds, analytic, lambda e: math.hypot(gx - e[0], gy - e[1]),
                            allow_reverse, max_nodes, 0.0, budget_s, coarse)

    def _search(self, grid, start, seeds, analytic, h_euclid, allow_reverse, max_nodes, w_offset, budget_s=None,
                coarse=None):
        """budget_s: give up after this much computing time (the relay's planner must answer in time on the Pi).
        coarse: the coarse lattice (default: with reversing)."""
        import time
        t_start = time.perf_counter()
        self.grid = grid
        self.expanded = 0
        self.gave_up = None
        m = self.margin
        if self.clearance(grid, np.array(start)) < 0.0:
            return None                                   # already touching: nothing to plan
        # obstacle-aware 2D heuristic: Dijkstra from the goal region over free cells (inflated by half the width)
        free = grid.dist > (self.p.width / 2 + m)
        h2d = self._dijkstra(grid, free, seeds)
        # the 2D heuristic says no free corridor reaches the goal from anywhere near the car: the kinematic search
        # cannot succeed either - answer at once instead of exhausting max_nodes (seconds on the Pi)
        r = int(0.45 / grid.res)
        ix0 = int((start[0] - grid.x0) / grid.res)
        iy0 = int((start[1] - grid.y0) / grid.res)
        near = h2d[max(0, ix0 - r):ix0 + r + 1, max(0, iy0 - r):iy0 + r + 1]
        if not near.size or near.min() >= 1e8:
            self.gave_up = "goal unreachable"
            return None
        kappas = np.linspace(-self.kappa_max, self.kappa_max, REVERSE_N_KAPPA if allow_reverse else 7)
        dirs = (1, -1) if allow_reverse else (1,)
        coarse = allow_reverse if coarse is None else coarse
        if coarse and not allow_reverse:
            kappas = np.linspace(-self.kappa_max, self.kappa_max, REVERSE_N_KAPPA)
        hbin = math.radians(REVERSE_HEADING_BIN_DEG if coarse else HEADING_BIN_DEG)
        eps = REVERSE_EPS if coarse else 1.2
        step_saved = self.step
        if coarse:
            self.step = REVERSE_STEP_M
        try:
            return self._search_loop(grid, start, h2d, analytic, h_euclid, max_nodes, w_offset, budget_s, t_start,
                                     kappas, dirs, hbin, eps)
        finally:
            self.step = step_saved

    def _search_loop(self, grid, start, h2d, analytic, h_euclid, max_nodes, w_offset, budget_s, t_start, kappas, dirs,
                     hbin, eps):
        import time
        m = self.margin
        start = tuple(float(v) for v in start)
        open_heap = [(0.0, 0, start, 0.0, 0.0, 1)]
        parents = {0: (None, None)}
        nodes = {0: start}
        closed = set()
        nid = 0
        while open_heap and self.expanded < max_nodes:
            f, i, pose, g, k_prev, d_prev = heapq.heappop(open_heap)
            key = (int(pose[0] / CELL_M), int(pose[1] / CELL_M), int(round(pose[2] / hbin)))
            if key in closed:
                continue
            closed.add(key)
            self.expanded += 1
            if budget_s is not None and self.expanded % 32 == 0 and time.perf_counter() - t_start > budget_s:
                self.gave_up = "time budget"
                return None
            # analytic expansion to the goal: every node close in, every 4th further out (Dolgov et al.: the
            # analytic expansion is tried more often as the heuristic gets small)
            h_node = (f - g) / eps
            if h_node < 0.8 or self.expanded % 4 == 0:
                fin = analytic(pose, d_prev)
                if fin is not None:
                    rj, d_fin = fin
                    return self._assemble(parents, nodes, i, np.column_stack([rj, np.full(len(rj), d_fin)]))
            for d in dirs:
                segs = self._arcs(pose, kappas, d)                        # all primitives at once
                clears = self._clearances(grid, segs)
                for j in np.flatnonzero(clears >= m):
                    k, seg, clear = kappas[j], segs[j], clears[j]
                    end = tuple(seg[-1])
                    cost = self.step * (1.0 + self.w_steer * abs(k) + (self.w_reverse if d < 0 else 0.0)) \
                        + self.w_change * abs(k - k_prev) * self.step + w_offset * abs(end[1]) * self.step \
                        + self.w_clear * max(0.0, self.clear_wish - clear) * self.step \
                        + (1.0 if d != d_prev else 0.0) * 0.5
                    g2 = g + cost
                    h = max(h_euclid(end), self._h(grid, h2d, end))
                    nid += 1
                    nodes[nid] = end
                    parents[nid] = (i, np.column_stack([seg, np.full(len(seg), d)]))
                    heapq.heappush(open_heap, (g2 + eps * h, nid, end, g2, k, d))
        self.gave_up = "node limit" if open_heap else "no way through"
        return None

    def _arcs(self, pose, kappas, d, n=4):
        """Poses along every motion primitive from pose: (len(kappas), n, 3), forward (d=1) or backward (d=-1)."""
        x, y, th = pose
        s = d * np.linspace(self.step / n, self.step, n)[None, :]
        k = np.asarray(kappas, float)[:, None]
        ths = th + k * s
        straight = np.abs(k) < 1e-6
        ks = np.where(straight, 1.0, k)
        px = np.where(straight, x + s * math.cos(th), x + (np.sin(ths) - math.sin(th)) / ks)
        py = np.where(straight, y + s * math.sin(th), y - (np.cos(ths) - math.cos(th)) / ks)
        return np.stack([px, py, np.broadcast_to(ths, px.shape)], axis=-1)

    def _clearances(self, grid, segs):
        """Smallest body clearance along each primitive: (P,)."""
        th = segs[..., 2][..., None]
        cx = segs[..., 0][..., None] + np.cos(th) * self.circle_x
        cy = segs[..., 1][..., None] + np.sin(th) * self.circle_x
        return (grid.lookup(cx, cy) - self.circle_r).reshape(len(segs), -1).min(axis=1)

    def _reverse_arc(self, pose, k):
        x, y, th = pose
        s = -np.linspace(self.step / 4, self.step, 4)
        if abs(k) < 1e-6:
            return np.column_stack([x + s * math.cos(th), y + s * math.sin(th), np.full(4, th)])
        ths = th + k * s
        return np.column_stack([x + (np.sin(ths) - math.sin(th)) / k, y - (np.cos(ths) - math.cos(th)) / k, ths])

    def _line_free(self, grid, x, ahead):
        xs = np.arange(x, x + ahead, 0.05)
        return bool((grid.lookup(xs, np.zeros_like(xs)) > self.p.width / 2 + self.margin).all())

    def _dijkstra(self, grid, free, seeds):
        """Grid distance (8-connected) from the goal cells over free cells; 1e9 where unreachable. Uses scipy's
        compiled Dijkstra (~10 ms for the whole grid; the pure-Python version below took 0.1-0.2 s)."""
        INF = 1e9
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import dijkstra
        except ImportError:                              # pragma: no cover
            return self._dijkstra_py(grid, free, seeds)
        nx, ny = free.shape
        idx = np.arange(nx * ny).reshape(nx, ny)
        rows, cols, wts = [], [], []
        for dx, dy, w in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
            a = (slice(0, nx - dx), slice(max(0, -dy), ny - max(0, dy)))
            b = (slice(dx, nx), slice(max(0, dy), ny - max(0, -dy)))
            both = free[a] & free[b]
            rows.append(idx[a][both])
            cols.append(idx[b][both])
            wts.append(np.full(int(both.sum()), w * grid.res))
        src = [int(idx[ix, iy]) for ix, iy in seeds if 0 <= ix < nx and 0 <= iy < ny and free[ix, iy]]
        if not src:
            return np.full(free.shape, INF)
        G = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny, nx * ny))
        dist = dijkstra(G.tocsr(), directed=False, indices=src, min_only=True)
        dist = dist.reshape(nx, ny)
        dist[~np.isfinite(dist)] = INF
        return dist

    def _dijkstra_py(self, grid, free, seeds):
        INF = 1e9
        d = np.full(free.shape, INF)
        heap = []
        for ix, iy in seeds:
            if 0 <= ix < grid.nx and 0 <= iy < grid.ny and free[ix, iy]:
                d[ix, iy] = 0.0
                heap.append((0.0, ix, iy))
        heapq.heapify(heap)
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while heap:
            c, ix, iy = heapq.heappop(heap)
            if c > d[ix, iy]:
                continue
            for dx, dy, w in steps:
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < grid.nx and 0 <= jy < grid.ny and free[jx, jy]:
                    nc = c + w * grid.res
                    if nc < d[jx, jy]:
                        d[jx, jy] = nc
                        heapq.heappush(heap, (nc, jx, jy))
        return d

    @staticmethod
    def _h(grid, h2d, pose):
        ix = int(np.clip((pose[0] - grid.x0) / grid.res, 0, grid.nx - 1))
        iy = int(np.clip((pose[1] - grid.y0) / grid.res, 0, grid.ny - 1))
        v = h2d[ix, iy]
        return v if v < 1e8 else 50.0

    @staticmethod
    def _assemble(parents, nodes, i, tail):
        segs = [tail]
        while i is not None:
            parent, seg = parents[i]
            if seg is not None:
                segs.append(seg)
            i = parent
        return np.vstack(segs[::-1])
