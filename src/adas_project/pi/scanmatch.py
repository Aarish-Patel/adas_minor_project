"""Scan registration (2D ICP) - numpy only, so it runs on the Pi and in the laptop simulator.

Frame convention everywhere: x = forward, y = toward POSITIVE bearing (the GUI's right-hand
side), so a point at bearing a, range d is (d*cos a, d*sin a). Positive rotation turns the car
toward +y."""
import numpy as np


def polar_to_xy(points, max_n=190, dmin=0.25, dmax=6.0):
    """max_n=None keeps every point (use that for obstacle detection; thinning is only for ICP speed)."""
    arr = np.array([(a, d) for a, d in points if dmin <= d <= dmax], dtype=float)
    if len(arr) == 0:
        return np.zeros((0, 2))
    arr = arr[np.argsort(arr[:, 0])]
    if max_n is not None and len(arr) > max_n:
        arr = arr[np.linspace(0, len(arr) - 1, max_n).astype(int)]
    r = np.radians(arr[:, 0])
    return np.stack([arr[:, 1] * np.cos(r), arr[:, 1] * np.sin(r)], axis=1)


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


try:
    from scipy.spatial import cKDTree
except ImportError:                    # pragma: no cover
    cKDTree = None


def _nearest(A, tree, P):
    """Nearest point of A for each row of P: (distance, index). k-d tree (as KISS-ICP / libpointmatcher) - the
    brute-force distance matrix it replaces cost ~10 ms per match on the Pi and stalled the control loop."""
    if tree is not None:
        return tree.query(P)
    d2 = ((P[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    j = d2.argmin(1)
    return np.sqrt(d2[np.arange(len(P)), j]), j


def icp(A, B, init=(0.0, 0.0, 0.0), iters=18, gates=(0.35, 0.18, 0.10), tree=None):
    """Find (R, t) with A ~= B @ R.T + t  (maps the NEWER scan B onto the OLDER scan A).
    Returns (R, t, rot_rad, mean_resid, n_inliers) or None. `init` = (dx, dy, dth) guess of the
    car's motion between the scans (the pose of the new scan's origin in the old frame). `tree`: a cKDTree of A,
    if the caller keeps one (A is matched against repeatedly)."""
    if len(A) < 50 or len(B) < 50:
        return None
    if tree is None and cKDTree is not None:
        tree = cKDTree(A)
    th = init[2]
    R = rot(th)
    t = np.array([init[0], init[1]], float)
    n_g = len(gates)
    for k in range(iters):
        gate = gates[min(k * n_g // iters, n_g - 1)]
        Bt = B @ R.T + t
        dist, j = _nearest(A, tree, Bt)
        keep = dist < gate
        if keep.sum() < 30:
            return None
        P, Q = Bt[keep], A[j[keep]]
        pc, qc = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
        Ri = Vt.T @ U.T
        if np.linalg.det(Ri) < 0:
            Vt[-1] *= -1
            Ri = Vt.T @ U.T
        ti = qc - Ri @ pc
        R, t = Ri @ R, Ri @ t + ti
    Bt = B @ R.T + t
    dist, _ = _nearest(A, tree, Bt)
    ok = dist < gates[-1] * 1.5
    if ok.sum() < 30:
        return None
    return R, t, float(np.arctan2(R[1, 0], R[0, 0])), float(dist[ok].mean()), int(ok.sum())


def match_residual(A, B, dx, dy, th, gate=0.15, tree=None):
    """Mean nearest-neighbour distance of scan B placed at (dx, dy, th) in A's frame, over points within `gate`,
    and how many points that was. Same measure icp() reports, but at a fixed pose (no iterations)."""
    Bt = B @ rot(th).T + np.array([dx, dy])
    if tree is None and cKDTree is not None:
        tree = cKDTree(A)
    d, _ = _nearest(A, tree, Bt)
    ok = d < gate
    return (float(d[ok].mean()) if ok.any() else 9.0), int(ok.sum())


class Odometry:
    """Integrates scan-to-scan ICP into a pose (x, y, th) in the START frame. Falls back to a
    kinematic prediction when a registration is poor, so one bad scan doesn't lose the pose."""

    def __init__(self):
        self.pose = np.zeros(3)      # x, y, th in the start frame
        self.prev = None
        self.prev_t = None
        self.fallbacks = 0
        self.n = 0
        self.ref = None              # the first scan: static reference that stops drift building up
        self.ref_fixes = 0
        self.along_fallbacks = 0     # scans where the along-corridor distance came from the speed prediction
        self.last_source = None      # 'icp', 'along_pred' (dx predicted) or 'pred' (whole step predicted)

    def _tree(self, pts):
        return cKDTree(pts) if cKDTree is not None and len(pts) else None

    def update(self, xy, t, v_pred, kappa_pred):
        if self.prev is None or self.prev_t is None:
            self.prev, self.prev_t = xy, t
            self.prev_tree = self._tree(xy)
            self.ref = xy
            self.ref_tree = self.prev_tree              # the start scan's tree is built once for the manoeuvre
            return self.pose
        dt = max(t - self.prev_t, 1e-3)
        dth = kappa_pred * v_pred * dt
        init = (v_pred * dt, 0.0, dth)
        res = icp(self.prev, xy, init, tree=self.prev_tree)
        ok = res is not None and res[3] < 0.05 and res[4] >= 45 and abs(res[2] - dth) < np.radians(6)
        if ok:
            R, tt, th, resid, n_in = res
            dx, dy, dth_used = float(tt[0]), float(tt[1]), th
            self.last_source = "icp"
            # corridor ambiguity: parallel walls pin down sideways position and heading but not how far the car
            # moved along them, and ICP then slides. If ICP disagrees with the speed prediction and the scans fit
            # just as well at the predicted distance, the scan cannot tell - trust the prediction for dx.
            step = v_pred * dt
            if step > 0.01 and abs(dx - step) > 0.4 * step:
                r_pred, n_pred = match_residual(self.prev, xy, step, dy, dth_used, tree=self.prev_tree)
                if r_pred <= resid * 1.15 and n_pred >= 0.9 * n_in:
                    dx = step
                    self.along_fallbacks += 1
                    self.last_source = "along_pred"
        else:
            dx, dy, dth_used = init
            self.fallbacks += 1
            self.last_source = "pred"
        x, y, thw = self.pose
        c, s = np.cos(thw), np.sin(thw)
        self.pose = np.array([x + c * dx - s * dy, y + s * dx + c * dy, thw + dth_used])
        self.prev, self.prev_t = xy, t
        self.prev_tree = self._tree(xy)
        self.n += 1
        # scan-to-reference correction: the start scan (walls + the obstacle) is static, so
        # registering against it gives the pose directly with no accumulated drift
        if self.n % 2 == 0 and self.ref is not None:
            x, y, thw = self.pose
            res = icp(self.ref, xy, (x, y, thw), iters=16, gates=(0.25, 0.12, 0.08), tree=self.ref_tree)
            # strict gate while the scene still looks like the start scan; a looser one (still needing a
            # decent inlier count and a modest correction) once the overlap has shrunk - that is exactly
            # when the scan-to-scan drift needs correcting most
            if res is not None and res[3] < 0.055 and res[4] >= 55:
                R, tt, th, resid, n_in = res
                jump = np.hypot(tt[0] - x, tt[1] - y)
                strict = resid < 0.04 and n_in >= 75
                if (jump < (0.20 if strict else 0.25)) and abs(th - thw) < np.radians(10 if strict else 8):
                    # with little overlap the match can slide along parallel walls, so a loose match only
                    # corrects sideways position and heading (what rejoining the line needs), not x
                    self.pose = np.array([tt[0] if strict else x, tt[1], th])
                    self.ref_fixes += 1
        return self.pose


class LocalMapOdometry:
    """Scan-to-local-map odometry: each scan is registered to a map of the recent keyframes, not to the previous scan, so errors do not
    add up scan after scan (KISS-ICP: Vizzo, Guadagnino, Mersch, Wiesmann, Behley, Stachniss, "KISS-ICP: In Defense of Point-to-Point
    ICP", RA-L 2023 - local voxel map, constant-velocity prediction). The registration minimises point-to-line distances to the map
    (the map's local normals; Censi, "An ICP variant using a point-to-line metric", ICRA 2008) by Gauss-Newton with a Huber weight.
    Degeneracy (a corridor: parallel walls say nothing about the distance along them) is read off the Gauss-Newton matrix itself: a
    direction with too little information keeps the motion prediction instead of sliding (Zhang, Kaess & Singh, "On degeneracy of
    optimization-based state estimation problems", ICRA 2016, solution remapping).

    Same interface as `Odometry`: update(xy, t, v_pred, kappa_pred) -> pose (x, y, th) in the START frame (the first scan's), xy in the
    same convention as Odometry (pi frame). `last_source` is 'icp', 'along_pred' (a degenerate direction kept from the prediction) or
    'pred' (the whole step predicted)."""

    VOXEL = 0.03            # m: one map point per cell (the newest)
    MAP_RADIUS = 4.0        # m: map points further than this from the car are dropped
    MAP_MAX = 5000
    KF_DIST, KF_DEG = 0.08, 3.0     # a scan joins the map after this much motion
    GATES = (0.30, 0.15, 0.08)      # m: correspondence distance, shrinking over the iterations
    ITERS = 10
    HUBER = 0.02
    CONST_VEL = False
    DEGEN_MIN = 25.0        # smallest eigenvalue of the (normalised) information matrix still trusted (inliers along a direction)

    def __init__(self):
        self.pose = np.zeros(3)
        self.prev_pose = None
        self.prev_t = None
        self.map_pts = np.zeros((0, 2))
        self.map_nrm = np.zeros((0, 2))
        self.tree = None
        self.kf_pose = None
        self.n = self.fallbacks = self.along_fallbacks = self.ref_fixes = 0
        self.last_source = None

    # ------------------------------------------------------------------ map
    def _normals(self, pts, tree, k=6):
        """Unit normal of the local line through each point's k nearest map neighbours (PCA)."""
        if tree is None or len(pts) < k:
            return np.tile([0.0, 1.0], (len(pts), 1))
        _, idx = tree.query(pts, k=k)
        nb = self.map_pts[idx]                                   # (N, k, 2)
        c = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", c, c)
        w, v = np.linalg.eigh(cov)
        return v[:, :, 0]                                        # eigenvector of the smallest eigenvalue

    def _add_to_map(self, xy):
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        w = xy @ np.array([[c, s], [-s, c]]) + self.pose[:2]
        pts = np.vstack([self.map_pts, w])
        cells = np.floor(pts / self.VOXEL).astype(np.int64)
        key = cells[:, 0] * 1_000_003 + cells[:, 1]
        _, last = np.unique(key[::-1], return_index=True)
        keep = np.sort(len(key) - 1 - last)
        pts = pts[keep]
        near = np.hypot(pts[:, 0] - self.pose[0], pts[:, 1] - self.pose[1]) < self.MAP_RADIUS
        pts = pts[near]
        if len(pts) > self.MAP_MAX:
            pts = pts[-self.MAP_MAX:]
        self.map_pts = pts
        self.tree = cKDTree(pts) if cKDTree is not None and len(pts) else None
        self.map_nrm = self._normals(pts, self.tree)
        self.kf_pose = self.pose.copy()

    # ------------------------------------------------------------------ registration
    def _register(self, xy, init):
        """Point-to-line Gauss-Newton of scan xy (vehicle) onto the map from pose `init`. Returns (pose, n_inliers, rms, degenerate
        direction or None)."""
        x, y, th = init
        n_in, rms, degen = 0, 9.0, None
        for k in range(self.ITERS):
            gate = self.GATES[min(k * len(self.GATES) // self.ITERS, len(self.GATES) - 1)]
            c, s = np.cos(th), np.sin(th)
            w = xy @ np.array([[c, s], [-s, c]]) + (x, y)
            d, j = self.tree.query(w, distance_upper_bound=gate)
            ok = np.isfinite(d)
            n_in = int(ok.sum())
            if n_in < 30:
                return None
            p, q, nr = w[ok], self.map_pts[j[ok]], self.map_nrm[j[ok]]
            r = ((p - q) * nr).sum(1)
            wt = np.where(np.abs(r) <= self.HUBER, 1.0, self.HUBER / np.maximum(np.abs(r), 1e-9))
            lever = np.column_stack([-(p[:, 1] - y), p[:, 0] - x])
            J = np.column_stack([nr, (nr * lever).sum(1)])
            H = (J * wt[:, None]).T @ J
            g = (J * wt[:, None]).T @ r
            try:
                u = -np.linalg.solve(H + 1e-6 * np.eye(3), g)
            except np.linalg.LinAlgError:
                return None
            # degeneracy: the translation block's weak direction (information per inlier, so the test does not depend on the count)
            ev, evec = np.linalg.eigh(H[:2, :2])
            if ev[0] < self.DEGEN_MIN:
                dvec = evec[:, 0]
                u[:2] -= dvec * (dvec @ u[:2])                   # no update along the direction the scene does not constrain
                degen = dvec
            else:
                degen = None
            x, y, th = x + u[0], y + u[1], th + u[2]
            rms = float(np.sqrt(np.mean(r ** 2)))
            if abs(u[0]) + abs(u[1]) < 1e-4 and abs(u[2]) < 1e-5:
                break
        return np.array([x, y, th]), n_in, rms, degen

    def update(self, xy, t, v_pred, kappa_pred):
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self.tree is None:
            if len(xy) >= 40:
                self._add_to_map(xy)
                self.prev_t = t
            return self.pose
        dt = max(t - self.prev_t, 1e-3) if self.prev_t is not None else 0.1
        # prediction: the speed / steering model (the relay's estimate) for the step, heading from the constant-velocity history
        x, y, th = self.pose
        step, dth = v_pred * dt, kappa_pred * v_pred * dt
        pred = np.array([x + step * np.cos(th + dth / 2), y + step * np.sin(th + dth / 2), th + dth])
        if self.CONST_VEL and self.prev_pose is not None and self.prev_t is not None:
            vel = self.pose - self.prev_pose                  # the matched motion of the last step (constant velocity, KISS-ICP)
            if np.hypot(vel[0], vel[1]) < 0.3:
                pred = self.pose + vel
        res = self._register(xy, pred) if len(xy) >= 40 else None
        self.prev_pose = self.pose.copy()
        if res is not None and res[2] < 0.05 and abs(res[0][2] - pred[2]) < np.radians(8) and np.hypot(*(res[0][:2] - pred[:2])) < 0.25:
            self.pose = res[0]
            self.last_source = "along_pred" if res[3] is not None else "icp"
            if res[3] is not None:
                self.along_fallbacks += 1
        else:
            self.pose = pred
            self.fallbacks += 1
            self.last_source = "pred"
        self.prev_t = t
        self.n += 1
        moved = np.hypot(*(self.pose[:2] - self.kf_pose[:2]))
        if moved > self.KF_DIST or abs(self.pose[2] - self.kf_pose[2]) > np.radians(self.KF_DEG):
            self._add_to_map(xy)
        return self.pose
