"""Scan registration (2D ICP) - numpy only, so it runs on the Pi and in the laptop simulator.

Frame convention everywhere: x = forward, y = toward POSITIVE bearing (the GUI's right-hand
side), so a point at bearing a, range d is (d*cos a, d*sin a). Positive rotation turns the car
toward +y."""
import numpy as np


def polar_to_xy(points, max_n=190, dmin=0.25, dmax=4.0):
    arr = np.array([(a, d) for a, d in points if dmin <= d <= dmax], dtype=float)
    if len(arr) == 0:
        return np.zeros((0, 2))
    arr = arr[np.argsort(arr[:, 0])]
    if len(arr) > max_n:
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
            R, tt, th, _, _ = res
            dx, dy, dth_used = float(tt[0]), float(tt[1]), th
        else:
            dx, dy, dth_used = init
            self.fallbacks += 1
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
            if res is not None and res[3] < 0.04 and res[4] >= 75:
                R, tt, th, _, _ = res
                jump = np.hypot(tt[0] - x, tt[1] - y)
                if jump < 0.20 and abs(th - thw) < np.radians(10):
                    self.pose = np.array([tt[0], tt[1], th])
                    self.ref_fixes += 1
        return self.pose
