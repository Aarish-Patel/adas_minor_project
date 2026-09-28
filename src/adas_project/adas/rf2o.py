"""Planar LiDAR velocity from range flow - RF2O (Jaimez, Monroy, Gonzalez-Jimenez, "Planar odometry from a radial
laser scanner. A range flow-based approach", ICRA 2016). numpy only; runs on the Pi.

For a static scene and a sensor moving with (vx, vy, w) in its own frame (x forward, y left, angles CCW), the range
R(a) seen along beam angle a changes over time as

    R_t = vx * (-cos a - R_a sin a / R) + vy * (-sin a + R_a cos a / R) + w * R_a            (range flow constraint)

with R_a = dR/da. Every valid beam gives one linear equation in (vx, vy, w): solved by iteratively re-weighted least
squares (Cauchy weights, as in the paper) with beams at depth discontinuities left out. Large motions break the
linearisation, so - like the paper's coarse-to-fine scheme - the new scan is warped by the current estimate and the
remaining motion is solved again, a few times.
"""
import math

import numpy as np


class RangeFlow:
    def __init__(self, n_bins=720, r_min=0.2, r_max=8.0, iters=3, jump_m=0.06):
        # iters: warp-and-solve rounds; from the EKF's guess it has converged after 2-3 on the real drive log
        # (0.04 cm/s from the 8-round answer), and each round costs ~0.2 ms on a laptop
        self.n = n_bins
        self.da = 2 * math.pi / n_bins
        self.alpha = (np.arange(n_bins) + 0.5) * self.da - math.pi
        self.r_min, self.r_max, self.iters, self.jump = r_min, r_max, iters, jump_m
        self.prev = None           # (xy points, range image) of the previous scan
        self.prev_t = None

    # ------------------------------------------------------------------ range images
    def image(self, xy):
        """Points (sensor frame, y left) -> range per angle bin (nearest return; NaN where nothing)."""
        r = np.hypot(xy[:, 0], xy[:, 1])
        ok = (r >= self.r_min) & (r <= self.r_max)
        a = np.arctan2(xy[ok, 1], xy[ok, 0])
        b = np.clip(((a + math.pi) / self.da).astype(int), 0, self.n - 1)
        img = np.full(self.n, np.inf)
        np.minimum.at(img, b, r[ok])
        img[~np.isfinite(img)] = np.nan
        return img

    @staticmethod
    def transform(xy, dx, dy, dth):
        """Points of the NEW scan into the OLD frame, given the new frame's pose (dx, dy, dth) in the old one."""
        c, s = math.cos(dth), math.sin(dth)
        return np.column_stack([dx + c * xy[:, 0] - s * xy[:, 1], dy + s * xy[:, 0] + c * xy[:, 1]])

    # ------------------------------------------------------------------ one solve
    def _solve(self, R0, R1, dt):
        """Remaining motion between range images R0 (old) and R1 (new, already warped into the old frame)."""
        Ra = (np.roll(R0, -1) - np.roll(R0, 1)) / (2 * self.da)
        jump = np.abs(np.roll(R0, -1) - R0) + np.abs(R0 - np.roll(R0, 1))
        ok = np.isfinite(R0) & np.isfinite(R1) & np.isfinite(Ra) & (jump < self.jump * (1 + R0))
        ok &= np.abs(R1 - R0) < 0.15 + 0.05 * R0           # different surfaces, not motion
        if ok.sum() < 30:
            return None
        a, R, Rt, Ra = self.alpha[ok], R0[ok], (R1[ok] - R0[ok]) / dt, Ra[ok]
        A = np.column_stack([-np.cos(a) - Ra * np.sin(a) / R, -np.sin(a) + Ra * np.cos(a) / R, Ra])
        w = np.ones(len(Rt))
        for _ in range(4):                                   # IRLS, Cauchy weights
            Aw = A * w[:, None]
            try:
                xi = np.linalg.solve(Aw.T @ A + 1e-6 * np.eye(3), Aw.T @ Rt)
            except np.linalg.LinAlgError:
                return None
            res = Rt - A @ xi
            k = 2.0 * max(np.median(np.abs(res)) * 1.4826, 1e-3)
            w = 1.0 / (1.0 + (res / k) ** 2)
        sigma2 = float((w * res ** 2).sum() / max(w.sum() - 3, 1.0))
        cov = np.linalg.inv((A * w[:, None]).T @ A + 1e-9 * np.eye(3)) * sigma2
        return xi, cov, int(ok.sum())

    # ------------------------------------------------------------------ scan to scan
    def update(self, xy, t, guess=None):
        """New scan (points in the sensor frame, x forward, y left) at time t. Returns (vx, vy, w, cov, n_beams) of
        the sensor over the last scan interval, or None (first scan, or too little structure). guess: (vx, vy, w)
        prior, e.g. from the car model - speeds up and steadies the warping."""
        R1 = self.image(xy)
        prev, prev_t = self.prev, self.prev_t
        self.prev, self.prev_t = (xy, R1), t
        if prev is None or t - prev_t < 1e-3:
            return None
        dt = t - prev_t
        R0 = prev[1]
        xi = np.zeros(3) if guess is None else np.asarray(guess, float).copy()
        out = None
        for _ in range(self.iters):
            dx, dy, dth = xi * dt
            R1w = self.image(self.transform(xy, dx, dy, dth))
            sol = self._solve(R0, R1w, dt)
            if sol is None:
                break
            dxi, cov, n = sol
            xi = xi + dxi
            out = (float(xi[0]), float(xi[1]), float(xi[2]), cov, n)
            if np.abs(dxi * dt).max() < 1e-4:
                break
        return out
