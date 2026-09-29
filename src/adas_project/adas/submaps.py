"""Submaps and loop closure: a drift-free map and pose for long drives and for 'return to start' (TODO W1/W2).

The problem. Scan-to-scan matching (pi/scanmatch.py, RF2O) gives a good pose for a few metres and then drifts (1.6-1.8 % of the
distance in the twin, more on the car); a route that comes back to where it began ends up somewhere else, and 'go home' drives to
the wrong place.

The method - the structure of Google's Cartographer (Hess, Kohler, Rapp, Andor, "Real-time loop closure in 2D LiDAR SLAM", ICRA
2016), in a compact form for a 720-beam LiDAR on a Pi:
  * SUBMAPS. Consecutive scans are inserted at their estimated poses into a small local map (a submap); a new one is started every
    `submap_scans` nodes. Inside a submap drift is negligible, and a finished submap never changes shape - only its pose in the
    world is adjusted later.
  * NODES. One node (pose) per `node_dist` metres / `node_deg` degrees of motion, with its scan.
  * LOOP CLOSURE. Each new scan is matched against every finished submap whose origin is near the current pose (not the recent
    ones): a correlative scan matcher (Olson, "Real-time correlative scan matching", ICRA 2009) evaluates all poses in a window
    against a likelihood field of the submap (a distance transform of its points), the best is refined by ICP; a high score with a
    clear margin over the runner-up becomes a constraint.
  * POSE GRAPH. Nodes are variables; consecutive nodes are tied by their measured relative pose (odometry edges), loop closures by
    the matched relative pose. Sparse pose adjustment (Konolige et al., ICRA 2010): robust nonlinear least squares over all
    node poses, node 0 fixed at the start. The corrected pose of the newest node is what 'return to start' drives home with.

    g = PoseGraphMap()
    pose = g.add_scan(pose_estimate, points_vehicle)      # (x, y, th) in the start frame; pose = corrected estimate
    g.correct(pose_estimate)                              # the same correction applied to any later estimate
"""
import math

import numpy as np

try:
    from scipy import ndimage
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
except ImportError:                                                   # pragma: no cover
    ndimage = least_squares = cKDTree = None


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def compose(a, b):
    """a (+) b: pose b expressed in a's frame -> world."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], _wrap(a[2] + b[2])])


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-(c * a[0] + s * a[1]), s * a[0] - c * a[1], _wrap(-a[2])])


def between(a, b):
    """The pose of b in a's frame."""
    return compose(inverse(a), b)


def transform(pose, pts):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.column_stack([pose[0] + c * pts[:, 0] - s * pts[:, 1], pose[1] + s * pts[:, 0] + c * pts[:, 1]])


class Submap:
    def __init__(self, idx, origin):
        self.idx, self.origin = idx, np.array(origin, float)      # origin: a node pose (x, y, th)
        self.pts = []                                             # scans in the submap frame
        self.nodes = []
        self.finished = False
        self.field = None
        self.res = 0.05
        self.lo = None

    def add(self, node_idx, node_pose, pts_vehicle, max_pts=6000):
        rel = between(self.origin, node_pose)
        self.pts.append(transform(rel, pts_vehicle))
        self.nodes.append(node_idx)

    def finish(self, sigma=0.08):
        """Freeze the submap: build the likelihood field (a distance transform of its occupied cells) for scan matching."""
        allp = np.vstack(self.pts)
        self.lo = allp.min(axis=0) - 0.6
        hi = allp.max(axis=0) + 0.6
        shape = np.ceil((hi - self.lo) / self.res).astype(int) + 1
        occ = np.ones(tuple(shape), bool)
        idx = ((allp - self.lo) / self.res).astype(int)
        occ[idx[:, 0], idx[:, 1]] = False
        dist = ndimage.distance_transform_edt(occ) * self.res
        self.field = np.exp(-dist ** 2 / (2 * sigma ** 2)).astype(np.float32)
        # thinned points for the ICP refinement
        keep = allp[:: max(1, len(allp) // 2500)]
        self.tree = cKDTree(keep)
        self.icp_pts = keep
        self.finished = True

    def score(self, pts):
        """Mean likelihood of points (in the submap frame) under the field."""
        idx = ((pts - self.lo) / self.res).astype(int)
        ok = (idx[:, 0] >= 0) & (idx[:, 0] < self.field.shape[0]) & (idx[:, 1] >= 0) & (idx[:, 1] < self.field.shape[1])
        if not ok.any():
            return 0.0
        return float(self.field[idx[ok, 0], idx[ok, 1]].sum() / len(pts))


class PoseGraphMap:
    def __init__(self, submap_scans=14, node_dist=0.20, node_deg=12.0, loop_radius=2.2, min_score=0.62, margin=0.08,
                 search_xy=0.45, search_deg=14.0, min_gap_nodes=12):
        self.submap_scans, self.node_dist, self.node_rad = submap_scans, node_dist, math.radians(node_deg)
        self.loop_radius, self.min_score, self.margin = loop_radius, min_score, margin
        self.search_xy, self.search_rad, self.min_gap = search_xy, math.radians(search_deg), min_gap_nodes
        self.est = []                  # node poses as estimated by the front end (never changed)
        self.nodes = []                # node poses, optimised
        self.scans = []                # scans (vehicle frame) per node
        self.submaps = []
        self.node_submap = []          # the submap each node was inserted into
        self.edges = []                # (i, j, rel pose measured, sigma_xy, sigma_th, robust)
        self.loops = []                # (node, submap idx, score)
        self.corrections = 0

    # ------------------------------------------------------------------ front end
    def add_scan(self, pose_est, pts_vehicle):
        pose_est = np.array(pose_est, float)
        pts = np.asarray(pts_vehicle, float)
        if len(pts) < 40:
            return self.correct(pose_est)
        if self.est:
            d = between(self.est[-1], pose_est)
            if math.hypot(d[0], d[1]) < self.node_dist and abs(d[2]) < self.node_rad:
                return self.correct(pose_est)
        i = len(self.est)
        self.est.append(pose_est)
        pose = pose_est.copy() if i == 0 else compose(self.nodes[-1], between(self.est[-2], pose_est))
        self.nodes.append(pose)
        self.scans.append(pts)
        if i > 0:
            self.edges.append((i - 1, i, between(self.est[-2], pose_est), 0.03 + 0.03 * math.hypot(*between(self.est[-2], pose_est)[:2]),
                               0.02, False))
        if not self.submaps or len(self.submaps[-1].nodes) >= self.submap_scans:
            if self.submaps:
                self.submaps[-1].finish()
            self.submaps.append(Submap(len(self.submaps), pose))
        sm = self.submaps[-1]
        sm.add(i, pose, pts)
        self.node_submap.append(sm.idx)
        if self._close_loops(i):
            self._optimise()
        return self.correct(pose_est)

    def correct(self, pose_est):
        """The optimised pose for a front-end estimate: the newest node's correction applied to the estimate."""
        if not self.est:
            return np.array(pose_est, float)
        rel = between(self.est[-1], np.array(pose_est, float))
        return compose(self.nodes[-1], rel)

    # ------------------------------------------------------------------ loop closure
    def _close_loops(self, i):
        added = False
        here = self.nodes[i]
        for sm in self.submaps[:-1]:
            if not sm.finished or i - sm.nodes[-1] < self.min_gap:
                continue
            if any(l[0] == i and l[1] == sm.idx for l in self.loops):
                continue
            if math.hypot(*(here[:2] - sm.origin[:2])) > self.loop_radius:
                continue
            pred = between(sm.origin, here)
            m = self._match(sm, self.scans[i], pred)
            if m is None:
                continue
            rel, score = m
            self.edges.append((sm.nodes[0], i, rel, 0.03, 0.025, True))
            self.loops.append((i, sm.idx, score))
            added = True
        return added

    def _match(self, sm, pts, pred):
        """Correlative scan matching of pts (vehicle frame) against a finished submap around the predicted pose (submap frame)."""
        sub = pts[:: max(1, len(pts) // 260)]
        xs = np.arange(-self.search_xy, self.search_xy + 1e-9, 0.05)
        ths = np.arange(-self.search_rad, self.search_rad + 1e-9, math.radians(1.5))
        best, second = (0.0, None), 0.0
        scores = []
        for dth in ths:
            th = pred[2] + dth
            c, s = math.cos(th), math.sin(th)
            base = np.column_stack([c * sub[:, 0] - s * sub[:, 1], s * sub[:, 0] + c * sub[:, 1]])
            for dx in xs:
                for dy in xs:
                    p = base + np.array([pred[0] + dx, pred[1] + dy])
                    sc = sm.score(p)
                    scores.append((sc, pred[0] + dx, pred[1] + dy, th))
        scores.sort(key=lambda t: -t[0])
        sc0, x0, y0, th0 = scores[0]
        if sc0 < self.min_score:
            return None
        # the runner-up must be a genuinely different pose (not a neighbour of the best)
        for sc, x, y, th in scores[1:]:
            if math.hypot(x - x0, y - y0) > 0.15 or abs(th - th0) > math.radians(6):
                second = sc
                break
        if sc0 - second < self.margin * 0.5:
            return None
        rel = self._icp_refine(sm, pts, np.array([x0, y0, th0]))
        return rel, sc0

    def _icp_refine(self, sm, pts, pose, iters=8):
        pose = pose.copy()
        sub = pts[:: max(1, len(pts) // 400)]
        for _ in range(iters):
            w = transform(pose, sub)
            d, j = sm.tree.query(w, distance_upper_bound=0.25)
            ok = np.isfinite(d)
            if ok.sum() < 30:
                break
            a, b = w[ok], sm.icp_pts[j[ok]]
            ca, cb = a.mean(axis=0), b.mean(axis=0)
            h = (a - ca).T @ (b - cb)
            dth = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
            c, s = math.cos(dth), math.sin(dth)
            t = cb - np.array([c * ca[0] - s * ca[1], s * ca[0] + c * ca[1]])
            pose = np.array([c * pose[0] - s * pose[1] + t[0], s * pose[0] + c * pose[1] + t[1], _wrap(pose[2] + dth)])
        return pose

    # ------------------------------------------------------------------ back end (sparse pose adjustment)
    def _optimise(self):
        n = len(self.nodes)
        x0 = np.array(self.nodes, float).ravel()
        edges = self.edges
        I = np.array([e[0] for e in edges])
        J = np.array([e[1] for e in edges])
        R = np.array([e[2] for e in edges])
        sxy = np.array([e[3] for e in edges])
        sth = np.array([e[4] for e in edges])

        def resid(x):
            p = x.reshape(n, 3)
            a, b = p[I], p[J]
            c, s = np.cos(a[:, 2]), np.sin(a[:, 2])
            dx, dy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
            rel = np.column_stack([c * dx + s * dy, -s * dx + c * dy, _wrap(b[:, 2] - a[:, 2])])
            r = rel - R
            r[:, 2] = _wrap(r[:, 2])
            r[:, 0] /= sxy
            r[:, 1] /= sxy
            r[:, 2] /= sth
            return np.concatenate([r.ravel(), (p[0] - self.nodes[0]) * 1e3])        # node 0 anchored at the start

        from scipy.sparse import lil_matrix
        m = 3 * len(edges) + 3
        sp = lil_matrix((m, 3 * n), dtype=int)
        for k, (i, j) in enumerate(zip(I, J)):
            for r_ in range(3):
                for c_ in range(3):
                    sp[3 * k + r_, 3 * i + c_] = 1
                    sp[3 * k + r_, 3 * j + c_] = 1
        for c_ in range(3):
            sp[3 * len(edges) + c_, c_] = 1
        sol = least_squares(resid, x0, jac_sparsity=sp, loss="soft_l1", f_scale=3.0, max_nfev=25, x_scale=1.0)
        new = sol.x.reshape(n, 3)
        new[:, 2] = _wrap(new[:, 2])
        self.nodes = [row.copy() for row in new]
        for sm in self.submaps:                                   # a submap sits at its first node
            sm.origin = self.nodes[sm.nodes[0]].copy()
        self.corrections += 1


class SlamService:
    """Runs the pose graph in a worker thread (a match + optimisation costs tens of milliseconds on a laptop, more on the Pi, and
    must never sit in the control loop). The relay submits (front-end pose, scan) for every scan; only the newest waits. The result
    is one transform, `correction`, applied to any front-end pose: corrected = correction (+) pose."""

    def __init__(self, **kw):
        import threading
        self.kw = kw
        self.graph = PoseGraphMap(**kw)
        self.correction = np.zeros(3)
        self.loops = 0
        self._slot, self._stop = None, False
        self._cv = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, pose_est, pts_vehicle):
        with self._cv:
            self._slot = (np.array(pose_est, float), np.asarray(pts_vehicle, float))
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while self._slot is None and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                pose, pts = self._slot
                self._slot = None
            try:
                self.graph.add_scan(pose, pts)
                g = self.graph
                if g.est:
                    self.correction = compose(g.nodes[-1], inverse(g.est[-1]))
                    self.loops = len(g.loops)
            except Exception:                                   # a failed match must never take the relay down
                pass

    def corrected(self, pose_est):
        return compose(self.correction, np.array(pose_est, float))

    def reset(self):
        with self._cv:
            self._slot = None
            self.graph = PoseGraphMap(**self.kw)
            self.correction = np.zeros(3)
            self.loops = 0

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
