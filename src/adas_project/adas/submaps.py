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

  * RELOCALISATION (TODO P2-2). Loop closure needs a pose guess; a car lifted and put down elsewhere, or switched on inside a room
    whose map was saved, has none. `PoseGraphMap.relocalise` searches EVERY submap over its whole extent and all headings: a coarse
    pass (0.2 m cells on the max-pooled likelihood field, 3 deg steps; the multi-resolution correlative matching of Olson 2009 and
    the pooled-field pyramid of Hess et al. 2016) proposes candidate poses, each is refined on the fine field and by ICP; the
    result is accepted only if the score is high, enough of the scan lies on the map, and the best pose beats every genuinely
    different pose (another place, or the room's mirror image) by a margin - a symmetric room gives no fix instead of a wrong one.
    `Relocaliser` also wants two consecutive scans to agree after the front end's motion between them. `rebase` then joins the new
    stretch to the map through that fix. `kidnap_score` spots the lift-and-drop from two consecutive scans.

    g = PoseGraphMap()
    pose = g.add_scan(pose_estimate, points_vehicle)      # (x, y, th) in the start frame; pose = corrected estimate
    g.correct(pose_estimate)                              # the same correction applied to any later estimate
    fix = g.relocalise(points_vehicle)                    # None, or Fix(pose in the map frame, score, margin, ...)
    g.save(path); PoseGraphMap.load(path)                 # a mapped room outlives the session
"""
import math
from dataclasses import dataclass

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


@dataclass
class Fix:
    """A global localisation of one scan: the vehicle pose in the map frame."""
    pose: np.ndarray            # (x, y, th) world
    score: float                # mean likelihood of the scan under the submap's field (0-1)
    margin: float               # score minus the best score of a genuinely different pose
    inliers: float              # fraction of scan points within 10 cm of a submap point
    submap: int
    rel: np.ndarray             # the pose in that submap's frame
    tried: int = 0              # candidate poses examined


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

    COARSE = 4                                                    # coarse cell = 4 fine cells = 0.20 m
    PAD = 24                                                      # zero border (cells) so shifted points never need a bounds test

    def _padded(self):
        if getattr(self, "_fpad", None) is None:
            self._fpad = np.pad(self.field, self.PAD).astype(np.float32)
        return self._fpad

    def score_cells(self, cells, shifts, fpad=None):
        """Mean likelihood of a scan whose points sit in field cells `cells` (n, 2, int) when shifted by each of `shifts` (m, 2, int
        cells) -> (m,). All shifts at once: one gather instead of a Python loop per pose (the correlative search of Olson 2009 is a
        sum over exactly this table). Cells outside the field read the zero border."""
        fpad = self._padded() if fpad is None else fpad
        h, w = fpad.shape
        ix = np.clip(cells[None, :, 0] + shifts[:, None, 0] + self.PAD, 0, h - 1)
        iy = np.clip(cells[None, :, 1] + shifts[:, None, 1] + self.PAD, 0, w - 1)
        return fpad.ravel()[ix * w + iy].sum(axis=1, dtype=np.float32) / len(cells)

    def score_shifts(self, pts, offsets, res=None, fpad=None):
        """The same for metric offsets (m, 2): the points' cells are found per shift (used by the coarse relocalisation search)."""
        res = self.res if res is None else res
        cells = np.floor((pts[None, :, :] + offsets[:, None, :] - self.lo) / res).astype(np.int32)
        fpad = self._padded() if fpad is None else fpad
        h, w = fpad.shape
        ix = np.clip(cells[..., 0] + self.PAD, 0, h - 1)
        iy = np.clip(cells[..., 1] + self.PAD, 0, w - 1)
        return fpad.ravel()[ix * w + iy].sum(axis=1, dtype=np.float32) / len(pts)

    def coarse_field(self, k=4):
        """The likelihood field max-pooled over k x k cells (an upper bound of the fine field in each block: the pyramid of
        Cartographer's fast correlative matcher, Hess et al. 2016) padded like the fine one, and the pooled cells where the car's
        origin can stand (no map point within ~0.2 m at the block centre)."""
        cache = getattr(self, "_coarse", None)
        if cache is None:
            cache = self._coarse = {}
        if k not in cache:
            h, w = self.field.shape
            ph, pw = (-h) % k, (-w) % k
            f = np.pad(self.field, ((0, ph), (0, pw)))
            pooled = f.reshape(f.shape[0] // k, k, f.shape[1] // k, k).max(axis=(1, 3))
            centre = f[k // 2::k, k // 2::k]
            cache[k] = (np.pad(pooled, self.PAD).astype(np.float32), centre < 0.05, pooled.shape)
        return cache[k]


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
        self.breaks = set()            # node indices whose predecessor belongs to another session (kidnap / restart): no odometry edge
        self._pending = None           # (submap idx, rel pose) - the relocalisation edge for the next node
        self._anchor = None            # front-end frame -> map frame for the first node after a break
        self.reloc = dict(coarse_deg=3.0, coarse_range=5.0, coarse_pts=60, coarse_min=0.30, per_submap=4, refine=6, level2_keep=24,
                          min_score=0.60, margin=0.06, min_inliers=0.55, min_pts=60)

    # ------------------------------------------------------------------ front end
    def add_scan(self, pose_est, pts_vehicle):
        pose_est = np.array(pose_est, float)
        pts = np.asarray(pts_vehicle, float)
        if len(pts) < 40:
            return self.correct(pose_est)
        brk = self._anchor is not None
        if self.est and not brk:
            d = between(self.est[-1], pose_est)
            if math.hypot(d[0], d[1]) < self.node_dist and abs(d[2]) < self.node_rad:
                return self.correct(pose_est)
        i = len(self.est)
        self.est.append(pose_est)
        if brk:                                                    # first node of a new session: placed by the fix, not by odometry
            pose = compose(self._anchor, pose_est)
            self.breaks.add(i)
            self._anchor = None
        else:
            pose = pose_est.copy() if i == 0 else compose(self.nodes[-1], between(self.est[-2], pose_est))
        self.nodes.append(pose)
        self.scans.append(pts)
        if i > 0 and not brk:
            self.edges.append((i - 1, i, between(self.est[-2], pose_est), 0.03 + 0.03 * math.hypot(*between(self.est[-2], pose_est)[:2]),
                               0.02, False))
        if brk and self._pending is not None:
            sm_idx, rel, score = self._pending
            self.edges.append((self.submaps[sm_idx].nodes[0], i, rel, 0.03, 0.025, True))
            self.loops.append((i, sm_idx, score))
            self._pending = None
        if not self.submaps or brk or len(self.submaps[-1].nodes) >= self.submap_scans:
            if self.submaps and not self.submaps[-1].finished:
                self.submaps[-1].finish()
            self.submaps.append(Submap(len(self.submaps), pose))
        sm = self.submaps[-1]
        sm.add(i, pose, pts)
        self.node_submap.append(sm.idx)
        if self._close_loops(i) or (brk and self.loops and self.loops[-1][0] == i):
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

    def _grid(self, sm, sub, pred, half_xy, half_th, step_th):
        """Correlative search (Olson 2009) of the scan `sub` (vehicle frame) around `pred` (submap frame) on the field's own 5 cm
        grid. Returns scores (n_th, n_x, n_y) and the offset axes."""
        res = sm.res
        n = int(round(half_xy / res))
        xs = np.arange(-n, n + 1) * res
        ths = np.arange(-half_th, half_th + 1e-9, step_th)
        gi, gj = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
        shifts = np.column_stack([gi.ravel(), gj.ravel()]).astype(np.int32)
        out = np.empty((len(ths), len(xs), len(xs)))
        for k, dth in enumerate(ths):
            th = pred[2] + dth
            c, s = math.cos(th), math.sin(th)
            base = np.column_stack([c * sub[:, 0] - s * sub[:, 1], s * sub[:, 0] + c * sub[:, 1]]) + pred[:2]
            cells = np.floor((base - sm.lo) / res).astype(np.int32)
            out[k] = sm.score_cells(cells, shifts).reshape(len(xs), len(xs))
        return out, xs, ths

    def _match(self, sm, pts, pred):
        """Correlative scan matching of pts (vehicle frame) against a finished submap around the predicted pose (submap frame)."""
        sub = pts[:: max(1, len(pts) // 260)]
        sc, xs, ths = self._grid(sm, sub, pred, self.search_xy, self.search_rad, math.radians(1.5))
        k, i, j = np.unravel_index(int(np.argmax(sc)), sc.shape)
        sc0 = float(sc[k, i, j])
        if sc0 < self.min_score:
            return None
        # the runner-up must be a genuinely different pose (not a neighbour of the best)
        far = ((xs[:, None] - xs[i]) ** 2 + (xs[None, :] - xs[j]) ** 2 > 0.15 ** 2)[None] | (np.abs(ths - ths[k]) > math.radians(6))[:, None, None]
        second = float(sc[far].max()) if far.any() else 0.0
        if sc0 - second < self.margin * 0.5:
            return None
        x0, y0, th0 = pred[0] + xs[i], pred[1] + xs[j], pred[2] + ths[k]
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

    # ------------------------------------------------------------------ relocalisation (no pose guess)
    def _view(self, sm):
        """A submap that can be searched now: finished ones as they are, the growing one as a frozen copy (cached per size)."""
        if sm.finished:
            return sm
        cache = getattr(self, "_views", {})
        if cache.get(sm.idx, (None, None))[0] != len(sm.nodes):
            v = Submap(sm.idx, sm.origin)
            v.pts, v.nodes = list(sm.pts), list(sm.nodes)
            v.finish()
            cache[sm.idx] = (len(sm.nodes), v)
            self._views = cache
        return cache[sm.idx][1]

    def _coarse(self, sm, pts):
        """Candidate poses of the scan in one submap over its whole extent and all headings: [(score, x, y, th)] best first.
        Two levels: 0.4 m cells and coarse headings over everything, then 0.2 m / 3 deg around the survivors."""
        r = self.reloc
        rng = np.hypot(pts[:, 0] - 0.12, pts[:, 1])
        use = pts[rng <= r["coarse_range"]]
        if len(use) < 30:
            use = pts
        use = use[:: max(1, len(use) // r["coarse_pts"])]
        # level 2: everything
        k2 = 2 * sm.COARSE
        pooled, standable, shape = sm.coarse_field(k2)
        ii, jj = np.nonzero(standable[: shape[0], : shape[1]])
        if not len(ii):
            return []
        rc2 = sm.res * k2
        cand = sm.lo + (np.column_stack([ii, jj]) + 0.5) * rc2
        step = r["coarse_deg"] * 2.0
        yaws = np.radians(np.arange(0.0, 360.0, step))
        best = np.empty((len(yaws), len(cand)), np.float32)
        for a, th in enumerate(yaws):
            best[a] = sm.score_shifts(self._rot(use, th), cand, res=rc2, fpad=pooled)
        flat = np.argsort(-best, axis=None)[: r["level2_keep"] * 20]
        keep = []
        for f in flat:
            a, m = divmod(int(f), len(cand))
            if len(keep) >= r["level2_keep"] or best[a, m] < r["coarse_min"]:
                break
            x, y, th = cand[m, 0], cand[m, 1], float(yaws[a])
            if all(math.hypot(x - o[0], y - o[1]) > 0.55 or abs(_wrap(th - o[2])) > math.radians(14) for o in keep):
                keep.append((x, y, th))
        # level 1: around each survivor, 0.2 m cells and 3 deg
        pooled1, _, _ = sm.coarse_field(sm.COARSE)
        rc1 = sm.res * sm.COARSE
        g = np.arange(-2, 3) * rc1
        offs = np.array([(dx, dy) for dx in g for dy in g])
        out = []
        for x, y, th in keep:
            tries = []
            for dth in np.radians(np.arange(-r["coarse_deg"] * 2, r["coarse_deg"] * 2 + 0.1, r["coarse_deg"])):
                sc = sm.score_shifts(self._rot(use, th + dth), offs + np.array([x, y]), res=rc1, fpad=pooled1)
                m = int(np.argmax(sc))
                tries.append((float(sc[m]), x + offs[m, 0], y + offs[m, 1], th + dth))
            out.append(max(tries))
        out = [o for o in sorted(out, reverse=True) if o[0] >= r["coarse_min"]]
        res = []
        for o in out:
            if all(math.hypot(o[1] - q[1], o[2] - q[2]) > 0.4 or abs(_wrap(o[3] - q[3])) > math.radians(10) for q in res):
                res.append(o)
        return res[: r["per_submap"]]

    @staticmethod
    def _rot(pts, th):
        c, s_ = math.cos(th), math.sin(th)
        return np.column_stack([c * pts[:, 0] - s_ * pts[:, 1], s_ * pts[:, 0] + c * pts[:, 1]])

    def _refine(self, sm, pts, cand):
        """Fine correlative search around a coarse candidate, then ICP; returns (score, pose in submap frame, inlier fraction)."""
        sub = pts[:: max(1, len(pts) // 260)]
        sc, xs, ths = self._grid(sm, sub, np.array(cand[1:]), 0.15, math.radians(4.0), math.radians(1.0))
        k, i, j = np.unravel_index(int(np.argmax(sc)), sc.shape)
        pose = self._icp_refine(sm, pts, np.array([cand[1] + xs[i], cand[2] + xs[j], cand[3] + ths[k]]))
        w = transform(pose, sub)
        d, _ = sm.tree.query(w, distance_upper_bound=0.10)
        return sm.score(w), pose, float(np.isfinite(d).mean())

    def relocalise(self, pts_vehicle):
        """Where is the car, given one scan and no idea? A `Fix` in the map frame, or None when the scan does not identify a place
        (unmapped area, too few points) or identifies two (symmetric room: the answer needs the car to move or turn)."""
        r = self.reloc
        pts = np.asarray(pts_vehicle, float)
        if len(pts) < r["min_pts"] or not self.submaps:
            return None
        hyps = []
        tried = 0
        for sm0 in self.submaps:
            if len(sm0.nodes) < 3:
                continue
            sm = self._view(sm0)
            for cand in self._coarse(sm, pts)[: r["refine"]]:
                tried += 1
                score, rel, inl = self._refine(sm, pts, cand)
                hyps.append((score, compose(sm.origin, rel), rel, inl, sm.idx))
        if not hyps:
            return None
        hyps.sort(key=lambda h: -h[0])
        best = hyps[0]
        second = 0.0
        for h in hyps[1:]:                                             # genuinely different: another place or heading
            if math.hypot(*(h[1][:2] - best[1][:2])) > 0.30 or abs(_wrap(h[1][2] - best[1][2])) > math.radians(8):
                second = h[0]
                break
        margin = best[0] - second
        if best[0] < r["min_score"] or best[3] < r["min_inliers"] or margin < r["margin"]:
            return None
        return Fix(best[1], float(best[0]), float(margin), best[3], best[4], best[2], tried)

    def rebase(self, fix, pose_est):
        """Continue the map from a fix: the front-end pose `pose_est` (the scan the fix came from) is at `fix.pose`. The next
        `add_scan` starts a new session there, tied to the fixed submap by an edge instead of an odometry step."""
        self._anchor = compose(fix.pose, inverse(np.array(pose_est, float)))
        self._pending = (fix.submap, fix.rel, fix.score)
        return self._anchor

    def start_new_session(self, pose_corrected, pose_est):
        """No fix (a new area): carry on from the corrected pose as it is, without a link to the old nodes."""
        self._anchor = compose(np.array(pose_corrected, float), inverse(np.array(pose_est, float)))
        self._pending = None

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        """The map (nodes, scans, edges) as a .npz: a room mapped once can be relocalised in after a restart."""
        off = np.cumsum([0] + [len(s) for s in self.scans])
        e = self.edges
        np.savez_compressed(
            path, nodes=np.array(self.nodes), est=np.array(self.est), scan_pts=np.vstack(self.scans).astype(np.float32), scan_off=off,
            node_submap=np.array(self.node_submap), breaks=np.array(sorted(self.breaks), int),
            edge_ij=np.array([[a[0], a[1]] for a in e], int).reshape(-1, 2), edge_rel=np.array([a[2] for a in e]).reshape(-1, 3),
            edge_sig=np.array([[a[3], a[4], float(a[5])] for a in e]).reshape(-1, 3),
            cfg=np.array([self.submap_scans, self.node_dist, math.degrees(self.node_rad), self.loop_radius, self.min_score, self.margin,
                          self.search_xy, math.degrees(self.search_rad), self.min_gap]))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        c = z["cfg"]
        g = cls(submap_scans=int(c[0]), node_dist=c[1], node_deg=c[2], loop_radius=c[3], min_score=c[4], margin=c[5], search_xy=c[6],
                search_deg=c[7], min_gap_nodes=int(c[8]))
        g.nodes = [row.copy() for row in z["nodes"]]
        g.est = [row.copy() for row in z["est"]]
        off = z["scan_off"]
        g.scans = [z["scan_pts"][off[k]: off[k + 1]].astype(float) for k in range(len(g.nodes))]
        g.node_submap = [int(v) for v in z["node_submap"]]
        g.breaks = set(int(v) for v in z["breaks"])
        g.edges = [(int(i), int(j), rel.copy(), float(sg[0]), float(sg[1]), bool(sg[2]))
                   for (i, j), rel, sg in zip(z["edge_ij"], z["edge_rel"], z["edge_sig"])]
        for k, idx in enumerate(g.node_submap):
            while len(g.submaps) <= idx:
                g.submaps.append(Submap(len(g.submaps), g.nodes[k]))
            g.submaps[idx].add(k, g.nodes[k], g.scans[k])
        for sm in g.submaps[:-1]:
            sm.finish()
        return g

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


class Relocaliser:
    """Confirms fixes over consecutive scans: a fix counts once a second, independent scan gives the same place after the motion
    the front end measured in between (guards against one bad scan - a person in view, a vibration glitch - putting the car in
    the wrong room). Feed it every scan while lost."""

    def __init__(self, graph, need=2, tol_xy=0.15, tol_deg=6.0, keep_s=3.0):
        self.graph, self.need, self.tol_xy, self.tol_th, self.keep = graph, need, tol_xy, math.radians(tol_deg), keep_s
        self.seen = []                                   # (t, front-end pose, fix)

    def update(self, t, pose_est, pts):
        fix = self.graph.relocalise(pts)
        pose_est = np.array(pose_est, float)
        self.seen = [h for h in self.seen if t - h[0] <= self.keep]
        if fix is None:
            return None
        agree = 1
        for _, e0, f0 in self.seen:
            carried = compose(f0.pose, between(e0, pose_est))         # the older fix moved along by the front end's motion
            d = fix.pose - carried
            if math.hypot(d[0], d[1]) <= self.tol_xy and abs(_wrap(d[2])) <= self.tol_th:
                agree += 1
        self.seen.append((t, pose_est, fix))
        return fix if agree >= self.need else None


def kidnap_score(prev_pts, pts, n_bins=180, max_shift_deg=25.0):
    """How well two consecutive scans agree, 0-1: the fraction of angular bins whose range differs by < 0.15 m after the best turn
    of up to +-25 deg. Driving, turning and people walking past leave this high; lifting the car and putting it down leaves ~0.
    Inputs are (n, 2) vehicle-frame points."""
    def image(p):
        a = np.arctan2(p[:, 1], p[:, 0] - 0.12)
        d = np.hypot(p[:, 1], p[:, 0] - 0.12)
        b = ((a + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
        img = np.full(n_bins, np.nan)
        order = np.argsort(-d)                                       # nearest last, so it wins the bin
        img[b[order]] = d[order]
        return img
    a, b = image(np.asarray(prev_pts, float)), image(np.asarray(pts, float))
    best = 0.0
    for sh in range(-int(max_shift_deg / 360 * n_bins), int(max_shift_deg / 360 * n_bins) + 1):
        bb = np.roll(b, sh)
        both = np.isfinite(a) & np.isfinite(bb)
        if both.sum() < 20:
            continue
        best = max(best, float((np.abs(a[both] - bb[both]) < 0.15).sum() / max(np.isfinite(a).sum(), 1)))
    return best


class SlamService:
    """Runs the pose graph in a worker thread (a match + optimisation costs tens of milliseconds on a laptop, more on the Pi, and
    must never sit in the control loop). The relay submits (front-end pose, scan) for every scan; only the newest waits. The result
    is one transform, `correction`, applied to any front-end pose: corrected = correction (+) pose.

    Lost mode: after `relocalise()` (a map was loaded, the operator asks) or when two consecutive scans disagree the way a lifted and
    dropped car's do (`kidnap_score` below `kidnap_below`), scans go to a `Relocaliser` instead of the graph until it has a
    confirmed fix; the graph is then rebased onto the fix and mapping goes on. With no fix after `give_up_s` seconds (a new area)
    it carries on from the pose it had, as a new session. `lost` tells the relay the corrected pose is not to be trusted."""

    def __init__(self, kidnap_below=0.30, give_up_s=8.0, inline=False, **kw):
        """inline=True (simulations): every scan is processed at once in the caller, in order - the worker thread's timing made the
        same simulated drive end differently from run to run (which scans it skipped, when a loop closure landed)."""
        import threading
        self.kw, self.inline = kw, inline
        self.graph = PoseGraphMap(**kw)
        self.correction = np.zeros(3)
        self.loops = 0
        self.kidnap_below, self.give_up_s = kidnap_below, give_up_s
        self.lost, self.lost_reason, self.last_fix = False, "", None
        self.scans_seen, self.relocalisations = 0, 0
        self._reloc, self._lost_since, self._prev_pts, self._prev_t = None, None, None, 0.0
        self._slot, self._stop, self._cmd = None, False, None
        self._cv = threading.Condition()
        if not inline:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def submit(self, pose_est, pts_vehicle, t=None):
        import time
        if self.inline:
            try:
                self._step(np.array(pose_est, float), np.asarray(pts_vehicle, float), time.time() if t is None else t)
            except Exception:                                   # as in the worker: a failed match must never stop the caller
                pass
            self.scans_seen += 1
            return
        with self._cv:
            self._slot = (np.array(pose_est, float), np.asarray(pts_vehicle, float), time.time() if t is None else t)
            self._cv.notify()

    def relocalise(self, reason="requested"):
        """Declare the pose lost (a map was loaded, the operator lifted the car): the next scans are used to find it again."""
        with self._cv:
            self.lost, self.lost_reason, self._lost_since = True, reason, None
            self._reloc = Relocaliser(self.graph)

    def save(self, path, timeout=10.0):
        """Write the map to `path` (.npz). Runs in the worker between scans, so the graph is not being changed while it is read."""
        return self._command(("save", path), timeout)

    def load(self, path, timeout=10.0):
        """Replace the map by a saved one and start looking for the car in it (`lost` until the first confirmed fix)."""
        return self._command(("load", path), timeout)

    def _command(self, cmd, timeout):
        import threading
        done = threading.Event()
        box = {}
        if self.inline:
            self._cmd = (cmd, done, box)
            self._do_command()
            return box.get("ok", False)
        with self._cv:
            self._cmd = (cmd, done, box)
            self._cv.notify()
        if not done.wait(timeout):
            return False
        return box.get("ok", False)

    def _do_command(self):
        (op, path), done, box = self._cmd
        self._cmd = None
        try:
            if op == "save":
                self.graph.save(path)
            else:
                self.graph = PoseGraphMap.load(path)
                self.correction = np.zeros(3)
                self.loops = len(self.graph.loops)
                self._prev_pts = None
                self.relocalise("map loaded")
            box["ok"] = True
        except Exception as e:                                  # a missing or damaged file must not stop the service
            box["ok"], box["error"] = False, str(e)
        done.set()

    def _run(self):
        while True:
            with self._cv:
                while self._slot is None and self._cmd is None and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                if self._cmd is not None:
                    pass
                else:
                    pose, pts, t = self._slot
                    self._slot = None
            if self._cmd is not None:
                self._do_command()
                continue
            try:
                self._step(pose, pts, t)
            except Exception:                                   # a failed match must never take the relay down
                pass
            self.scans_seen += 1

    def _step(self, pose, pts, t):
        g = self.graph
        if not self.lost and self._prev_pts is not None and len(pts) > 40 and g.est and t - self._prev_t < 0.35:   # consecutive scans only
            if kidnap_score(self._prev_pts, pts) < self.kidnap_below:
                self.relocalise("scans changed abruptly")
        self._prev_pts, self._prev_t = pts, t
        found = False
        if self.lost:
            if self._lost_since is None:
                self._lost_since = t
            fix = self._reloc.update(t, pose, pts)
            if fix is not None:
                g.rebase(fix, pose)
                self.last_fix = fix
                self.relocalisations += 1
                found = True
            elif t - self._lost_since > self.give_up_s:         # nothing matches: a new area, carry on
                g.start_new_session(self.corrected(pose), pose)
                found = True
            else:
                return
        g.add_scan(pose, pts)
        if g.est:
            self.correction = compose(g.nodes[-1], inverse(g.est[-1]))
            self.loops = len(g.loops)
        if found:
            self.lost = False                                    # only once the corrected pose is the new one

    def corrected(self, pose_est):
        return compose(self.correction, np.array(pose_est, float))

    def reset(self):
        with self._cv:
            self._slot = None
            self.graph = PoseGraphMap(**self.kw)
            self.correction = np.zeros(3)
            self.loops = 0
            self.lost, self._reloc, self._prev_pts = False, None, None

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
