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


def icp(A, B, init=(0.0, 0.0, 0.0), iters=18, gates=(0.35, 0.18, 0.10)):
    """Find (R, t) with A ~= B @ R.T + t  (maps the NEWER scan B onto the OLDER scan A).
    Returns (R, t, rot_rad, mean_resid, n_inliers) or None. `init` = (dx, dy, dth) guess of the
    car's motion between the scans (the pose of the new scan's origin in the old frame)."""
    if len(A) < 50 or len(B) < 50:
        return None
    th = init[2]
    R = rot(th)
    t = np.array([init[0], init[1]], float)
    n_g = len(gates)
    for k in range(iters):
        gate = gates[min(k * n_g // iters, n_g - 1)]
        Bt = B @ R.T + t
        d2 = ((Bt[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(B)), j])
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
    d2 = ((Bt[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    dist = np.sqrt(d2.min(1))
    ok = dist < gates[-1] * 1.5
    if ok.sum() < 30:
        return None
    return R, t, float(np.arctan2(R[1, 0], R[0, 0])), float(dist[ok].mean()), int(ok.sum())


def match_residual(A, B, dx, dy, th, gate=0.15):
    """Mean nearest-neighbour distance of scan B placed at (dx, dy, th) in A's frame, over points within `gate`,
    and how many points that was. Same measure icp() reports, but at a fixed pose (no iterations)."""
    Bt = B @ rot(th).T + np.array([dx, dy])
    d = np.sqrt(((Bt[:, None, :] - A[None, :, :]) ** 2).sum(-1).min(1))
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

    def update(self, xy, t, v_pred, kappa_pred):
        if self.prev is None or self.prev_t is None:
            self.prev, self.prev_t = xy, t
            self.ref = xy
            return self.pose
        dt = max(t - self.prev_t, 1e-3)
        dth = kappa_pred * v_pred * dt
        init = (v_pred * dt, 0.0, dth)
        res = icp(self.prev, xy, init)
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
                r_pred, n_pred = match_residual(self.prev, xy, step, dy, dth_used)
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
        self.n += 1
        # scan-to-reference correction: the start scan (walls + the obstacle) is static, so
        # registering against it gives the pose directly with no accumulated drift
        if self.n % 2 == 0 and self.ref is not None:
            x, y, thw = self.pose
            res = icp(self.ref, xy, (x, y, thw), iters=16, gates=(0.25, 0.12, 0.08))
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
